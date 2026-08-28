#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py - PyTorch Dataset for Myanmar Sign Language Recognition

Supports:
  - Loading pre-extracted keypoint sequences (.npy)
  - On-the-fly augmentation (for training)
  - Padding / truncation to fixed length
  - Returning variable-length sequences with masks
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from augment import RandomAugmentor


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MSLDataset(Dataset):
    """
    Myanmar Sign Language keypoint dataset.

    Each item is a dict:
      {
        'keypoints': FloatTensor (T, 75, 3)  – or flattened (T, 225),
        'label':     LongTensor  ()
        'length':    int  – original (unpadded) sequence length,
        'mask':      BoolTensor  (max_seq_len,)  – True where padded,
        'idx':       int  – annotation index,
      }
    """

    def __init__(
        self,
        records:      List[Dict],
        label2idx:    Dict[str, int],
        max_seq_len:  int  = 200,
        flatten:      bool = True,     # (T,75,3) → (T,225)
        augmentor:    Optional[RandomAugmentor] = None,
        min_seq_len:  int  = 5,
    ):
        """
        Args:
            records:     list of dicts with 'keypoint_path' and 'label' keys.
            label2idx:   mapping from label string to class index.
            max_seq_len: sequences longer than this are centre-cropped;
                         shorter ones are zero-padded.
            flatten:     if True, return (T, 225) instead of (T, 75, 3).
            augmentor:   RandomAugmentor instance for on-the-fly augmentation.
            min_seq_len: samples with fewer than this many frames are skipped.
        """
        self.label2idx   = label2idx
        self.max_seq_len = max_seq_len
        self.flatten     = flatten
        self.augmentor   = augmentor
        self.min_seq_len = min_seq_len

        # Filter out records with missing keypoint files
        self.samples = []
        missing = 0
        for rec in records:
            kp = rec.get('keypoint_path')
            if kp and Path(kp).exists():
                self.samples.append(rec)
            else:
                missing += 1

        if missing:
            print(f"[MSLDataset] Warning: {missing} samples have no keypoint file.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        rec    = self.samples[idx]
        seq    = np.load(rec['keypoint_path']).astype(np.float32)  # (T, 75, 3)
        label  = self.label2idx[rec['label']]

        # On-the-fly augmentation
        if self.augmentor is not None:
            seq = self.augmentor(seq)

        # Ensure minimum length
        T = seq.shape[0]
        if T < self.min_seq_len:
            # Repeat sequence until long enough
            reps = int(np.ceil(self.min_seq_len / T))
            seq  = np.tile(seq, (reps, 1, 1))[:self.min_seq_len]
            T    = self.min_seq_len

        # Centre-crop if too long
        if T > self.max_seq_len:
            start = (T - self.max_seq_len) // 2
            seq   = seq[start: start + self.max_seq_len]
            T     = self.max_seq_len

        length = T  # true length before padding

        # Zero-pad to max_seq_len
        pad_len = self.max_seq_len - T
        if pad_len > 0:
            pad_shape = (pad_len, seq.shape[1], seq.shape[2])
            seq = np.concatenate([seq, np.zeros(pad_shape, dtype=np.float32)], axis=0)

        # Padding mask: True where padded (for Transformer key_padding_mask)
        mask = np.zeros(self.max_seq_len, dtype=bool)
        mask[length:] = True

        if self.flatten:
            seq = seq.reshape(self.max_seq_len, -1)  # (T, 225)

        return {
            'keypoints': torch.from_numpy(seq),
            'label':     torch.tensor(label, dtype=torch.long),
            'length':    torch.tensor(length, dtype=torch.long),
            'mask':      torch.from_numpy(mask),
            'idx':       rec['idx'],
        }


class MSLDatasetGCN(MSLDataset):
    """
    Variant that returns keypoints in ST-GCN format: (C, T, V)
      C = coordinate channels (3)
      T = time steps
      V = nodes (75)
    """

    def __getitem__(self, idx: int) -> Dict:
        item = super().__getitem__(idx)
        # Reshape from (T, 225) → (T, 75, 3) → (3, T, 75)
        kp = item['keypoints'].reshape(self.max_seq_len, 75, 3)
        kp = kp.permute(2, 0, 1)              # (3, T, 75)
        item['keypoints'] = kp
        return item


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    train_records: List[Dict],
    val_records:   List[Dict],
    test_records:  List[Dict],
    label2idx:     Dict[str, int],
    cfg:           dict,
    augmentor:     Optional[RandomAugmentor] = None,
    model_type:    str = 'bilstm',            # 'bilstm' | 'transformer' | 'stgcn'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    For training:  uses augmentor if provided.
    For val/test:  no augmentation, deterministic order.
    """
    dcfg   = cfg['data']
    tcfg   = cfg['training']
    max_T  = dcfg['max_seq_len']

    DatasetClass = MSLDatasetGCN if model_type == 'stgcn' else MSLDataset
    flatten      = (model_type != 'stgcn')

    train_ds = DatasetClass(
        train_records, label2idx,
        max_seq_len = max_T,
        flatten     = flatten,
        augmentor   = augmentor,
    )
    val_ds = DatasetClass(
        val_records, label2idx,
        max_seq_len = max_T,
        flatten     = flatten,
        augmentor   = None,
    )
    test_ds = DatasetClass(
        test_records, label2idx,
        max_seq_len = max_T,
        flatten     = flatten,
        augmentor   = None,
    )

    loader_kwargs = dict(
        batch_size  = tcfg['batch_size'],
        num_workers = tcfg.get('num_workers', 4),
        pin_memory  = tcfg.get('pin_memory', True),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **loader_kwargs)

    print(f"\n[DataLoaders]")
    print(f"  Train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_ds)}   samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_ds)}  samples, {len(test_loader)} batches\n")

    return train_loader, val_loader, test_loader
