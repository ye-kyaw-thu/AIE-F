
"""
Neural unit->syllable pipeline for Myanmar handwriting.

Design
------
1) Build per-sample padded stroke-embedding sequences from train/test_emb_df.
2) Train a small BiLSTM encoder with a CTC unit head to predict variable-length unit sequences.
3) Freeze (or optionally partially unfreeze) the encoder.
4) Train a syllable classifier on top of the frozen encoder using a BiLSTM head + masked pooling.
5) Evaluate, save, and reload the full pipeline.

Why CTC for the unit stage?
---------------------------
Stroke count T and unit count U do not align 1:1. CTC lets us train a unit bottleneck
without hand-crafted alignment while still using padded batches.

Expected inputs
---------------
- train_emb_df / test_emb_df: long-form per-stroke embedding DataFrames with at least
    [sample_index, stroke_index, embedding]
  and optionally repeated columns like [syllable, align_unit_array].
- train_meta_df / test_meta_df (optional): one-row-per-sample DataFrames providing
    [sample_index, syllable, align_unit_array].
  If omitted, the loader will take syllable / unit labels from the first stroke row of each sample.

Typical usage
-------------
>>> from nn_unit_syllable_pipeline import run_full_pipeline, DataPrepConfig, ModelConfig, TrainConfig
>>> result = run_full_pipeline(
...     train_emb_df=train_emb_df,
...     test_emb_df=test_emb_df,
...     train_meta_df=train_sample_df,   # optional
...     test_meta_df=test_sample_df,     # optional
...     data_cfg=DataPrepConfig(sample_index_col="sample_index", stroke_index_col="stroke_index",
...                             emb_col="embedding", syllable_col="syllable", unit_array_col="align_unit_array"),
...     model_cfg=ModelConfig(hidden_dim=128, enc_layers=1, enc_dropout=0.10,
...                           head_hidden_dim=128, head_layers=1, head_dropout=0.10),
...     unit_train_cfg=TrainConfig(batch_size=32, epochs=20, lr=1e-3, weight_decay=1e-4,
...                                grad_clip=5.0, patience=5, device="cuda"),
...     syll_train_cfg=TrainConfig(batch_size=32, epochs=20, lr=1e-3, weight_decay=1e-4,
...                                grad_clip=5.0, patience=5, device="cuda"),
...     save_path="unit_syllable_pipeline.pt",
... )
>>> print(result["syll_test_metrics"])  # top-1 / top-k metrics
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import time


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Repro / config
# ============================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class DataPrepConfig:
    sample_index_col: str = "sample_index"
    stroke_index_col: str = "stroke_index"
    emb_col: str = "embedding"
    syllable_col: str = "syllable"
    unit_array_col: str = "align_unit_array"
    blank_token: str = "<BLANK>"
    unk_token: str = "<UNK>"


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    enc_layers: int = 1
    enc_dropout: float = 0.10
    bidirectional: bool = True
    # Syllable head
    head_hidden_dim: int = 128
    head_layers: int = 1
    head_dropout: float = 0.10


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 5
    device: str = "cpu"
    num_workers: int = 0
    log_every: int = 50


# ============================================================
# Data preparation
# ============================================================


def _safe_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if pd.isna(x):
        return []
    # fall back to singleton string token only if clearly non-empty
    return list(x) if isinstance(x, str) else [x]


def build_sample_table(
    stroke_emb_df: pd.DataFrame,
    *,
    sample_meta_df: Optional[pd.DataFrame] = None,
    cfg: DataPrepConfig,
    require_units: bool = True,
) -> pd.DataFrame:
    """
    Convert long-form stroke-embedding dataframe into one-row-per-sample table.

    Output columns:
      - sample_index
      - emb_seq        : np.ndarray [T, D]
      - seq_len        : int
      - syllable       : str
      - unit_seq       : list[str]
    """
    s_col = cfg.sample_index_col
    t_col = cfg.stroke_index_col
    e_col = cfg.emb_col
    y_col = cfg.syllable_col
    u_col = cfg.unit_array_col

    if s_col not in stroke_emb_df.columns:
        raise KeyError(f"Missing {s_col!r} in stroke_emb_df")
    if t_col not in stroke_emb_df.columns:
        raise KeyError(f"Missing {t_col!r} in stroke_emb_df")
    if e_col not in stroke_emb_df.columns:
        raise KeyError(f"Missing {e_col!r} in stroke_emb_df")

    meta_lookup = None
    if sample_meta_df is not None:
        if s_col not in sample_meta_df.columns:
            raise KeyError(f"Missing {s_col!r} in sample_meta_df")
        meta_lookup = sample_meta_df.set_index(s_col).to_dict("index")

    rows = []
    for sid, g in stroke_emb_df.groupby(s_col):
        g = g.sort_values(t_col)
        emb_list = [np.asarray(v, dtype=np.float32) for v in g[e_col].tolist()]
        if len(emb_list) == 0:
            continue
        emb_seq = np.vstack(emb_list).astype(np.float32)

        if meta_lookup is not None:
            meta = meta_lookup.get(sid, None)
            if meta is None:
                continue
            syll = str(meta.get(y_col, ""))
            unit_seq = _safe_list(meta.get(u_col, []))
        else:
            syll = str(g[y_col].iloc[0]) if y_col in g.columns else ""
            unit_seq = _safe_list(g[u_col].iloc[0]) if u_col in g.columns else []

        if require_units and len(unit_seq) == 0:
            continue
        if syll == "":
            continue

        rows.append({
            s_col: int(sid),
            "emb_seq": emb_seq,
            "seq_len": int(len(emb_seq)),
            y_col: syll,
            "unit_seq": [str(u) for u in unit_seq],
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No valid samples were built. Check unit labels / sample mapping.")
    return out.reset_index(drop=True)


def build_inference_sample_table(
    stroke_emb_df: pd.DataFrame,
    *,
    sample_meta_df: Optional[pd.DataFrame] = None,
    cfg: DataPrepConfig,
) -> pd.DataFrame:
    """
    Convert long-form stroke-embedding dataframe into one-row-per-sample table
    for inference.

    Output columns:
      - sample_index
      - emb_seq        : np.ndarray [T, D]
      - seq_len        : int
      - syllable       : optional, only if available from meta_df or emb_df

    Unlike build_sample_table(...), this does NOT require unit labels.
    """
    s_col = cfg.sample_index_col
    t_col = cfg.stroke_index_col
    e_col = cfg.emb_col
    y_col = cfg.syllable_col

    if s_col not in stroke_emb_df.columns:
        raise KeyError(f"Missing {s_col!r} in stroke_emb_df")
    if t_col not in stroke_emb_df.columns:
        raise KeyError(f"Missing {t_col!r} in stroke_emb_df")
    if e_col not in stroke_emb_df.columns:
        raise KeyError(f"Missing {e_col!r} in stroke_emb_df")

    meta_lookup = None
    if sample_meta_df is not None:
        if s_col not in sample_meta_df.columns:
            raise KeyError(f"Missing {s_col!r} in sample_meta_df")
        meta_lookup = sample_meta_df.set_index(s_col).to_dict("index")

    rows = []
    for sid, g in stroke_emb_df.groupby(s_col):
        g = g.sort_values(t_col)
        emb_list = [np.asarray(v, dtype=np.float32) for v in g[e_col].tolist()]
        if len(emb_list) == 0:
            continue
        emb_seq = np.vstack(emb_list).astype(np.float32)

        row = {
            s_col: int(sid),
            "emb_seq": emb_seq,
            "seq_len": int(len(emb_seq)),
        }

        # optional label for evaluation
        syll = None
        if meta_lookup is not None:
            meta = meta_lookup.get(sid, None)
            if meta is not None and y_col in meta:
                syll = str(meta[y_col])
        elif y_col in g.columns:
            syll = str(g[y_col].iloc[0])

        if syll is not None:
            row[y_col] = syll

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No valid inference samples were built.")
    return out.reset_index(drop=True)


class InferenceDataset(Dataset):
    """
    One-row-per-sample dataset for inference.

    sample_table must contain:
      - sample_index
      - emb_seq
      - seq_len

    Optional:
      - syllable
    """
    def __init__(self, sample_table: pd.DataFrame, cfg: DataPrepConfig):
        self.df = sample_table.reset_index(drop=True)
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            "sample_index": int(row[self.cfg.sample_index_col]),
            "x": torch.tensor(row["emb_seq"], dtype=torch.float32),
            "x_len": int(row["seq_len"]),
        }
        if self.cfg.syllable_col in row.index and pd.notna(row[self.cfg.syllable_col]):
            item["syllable"] = str(row[self.cfg.syllable_col])
        return item


def collate_infer(batch: List[dict]):
    xs = [b["x"] for b in batch]
    x_lens = torch.tensor([b["x_len"] for b in batch], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)

    out = {
        "x": x_pad,
        "x_lens": x_lens,
        "sample_index": [b["sample_index"] for b in batch],
    }

    if "syllable" in batch[0]:
        out["syllable"] = [b["syllable"] for b in batch]

    return out


def build_unit_vocab(sample_table: pd.DataFrame, cfg: DataPrepConfig):
    units = sorted({u for seq in sample_table["unit_seq"] for u in seq})
    unit2idx = {cfg.blank_token: 0, cfg.unk_token: 1}
    for u in units:
        if u not in unit2idx:
            unit2idx[u] = len(unit2idx)
    idx2unit = {i: u for u, i in unit2idx.items()}
    return unit2idx, idx2unit



def build_syllable_vocab(train_sample_table: pd.DataFrame, cfg: DataPrepConfig):
    syllables = sorted(train_sample_table[cfg.syllable_col].astype(str).unique().tolist())
    syll2idx = {s: i for i, s in enumerate(syllables)}
    idx2syll = {i: s for s, i in syll2idx.items()}
    return syll2idx, idx2syll


# ============================================================
# Datasets / collate
# ============================================================


class UnitCTCDataset(Dataset):
    def __init__(self, sample_table: pd.DataFrame, unit2idx: Dict[str, int], cfg: DataPrepConfig):
        self.df = sample_table.reset_index(drop=True)
        self.unit2idx = unit2idx
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row["emb_seq"], dtype=torch.float32)
        units = [self.unit2idx.get(u, self.unit2idx[self.cfg.unk_token]) for u in row["unit_seq"]]
        y = torch.tensor(units, dtype=torch.long)
        return {
            "sample_index": int(row[self.cfg.sample_index_col]),
            "x": x,
            "x_len": int(row["seq_len"]),
            "y_units": y,
            "y_units_len": int(len(y)),
            "syllable": str(row[self.cfg.syllable_col]),
        }


class SyllableDataset(Dataset):
    def __init__(self, sample_table: pd.DataFrame, syll2idx: Dict[str, int], cfg: DataPrepConfig):
        self.df = sample_table.reset_index(drop=True)
        self.syll2idx = syll2idx
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(row["emb_seq"], dtype=torch.float32)
        y = self.syll2idx[str(row[self.cfg.syllable_col])]
        return {
            "sample_index": int(row[self.cfg.sample_index_col]),
            "x": x,
            "x_len": int(row["seq_len"]),
            "y_syll": int(y),
            "syllable": str(row[self.cfg.syllable_col]),
        }



def collate_ctc(batch: List[dict]):
    xs = [b["x"] for b in batch]
    x_lens = torch.tensor([b["x_len"] for b in batch], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)

    y_lens = torch.tensor([b["y_units_len"] for b in batch], dtype=torch.long)
    y_concat = torch.cat([b["y_units"] for b in batch], dim=0)
    y_list = [b["y_units"].tolist() for b in batch]

    return {
        "x": x_pad,
        "x_lens": x_lens,
        "y_units": y_concat,
        "y_units_lens": y_lens,
        "y_units_list": y_list,
        "sample_index": [b["sample_index"] for b in batch],
        "syllable": [b["syllable"] for b in batch],
    }



def collate_syll(batch: List[dict]):
    xs = [b["x"] for b in batch]
    x_lens = torch.tensor([b["x_len"] for b in batch], dtype=torch.long)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=0.0)
    y = torch.tensor([b["y_syll"] for b in batch], dtype=torch.long)
    return {
        "x": x_pad,
        "x_lens": x_lens,
        "y_syll": y,
        "sample_index": [b["sample_index"] for b in batch],
        "syllable": [b["syllable"] for b in batch],
    }


# ============================================================
# Models
# ============================================================


class StrokeBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: ModelConfig):
        super().__init__()
        dropout = cfg.enc_dropout if cfg.enc_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.enc_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=cfg.bidirectional,
        )
        self.output_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
        return out, out_lengths


class UnitCTCModel(nn.Module):
    def __init__(self, input_dim: int, num_units: int, model_cfg: ModelConfig):
        super().__init__()
        self.encoder = StrokeBiLSTMEncoder(input_dim, model_cfg)
        self.unit_head = nn.Linear(self.encoder.output_dim, num_units)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        enc, out_lengths = self.encoder(x, lengths)
        logits = self.unit_head(enc)                # [B, T, C]
        log_probs = F.log_softmax(logits, dim=-1)  # [B, T, C]
        return log_probs.transpose(0, 1), out_lengths, enc  # [T, B, C]


class SyllableBiLSTMClassifier(nn.Module):
    def __init__(self, encoder: StrokeBiLSTMEncoder, num_syllables: int, model_cfg: ModelConfig, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        dropout = model_cfg.head_dropout if model_cfg.head_layers > 1 else 0.0
        self.head_rnn = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=model_cfg.head_hidden_dim,
            num_layers=model_cfg.head_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        head_out_dim = model_cfg.head_hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_out_dim * 2),
            nn.Linear(head_out_dim * 2, head_out_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.head_dropout),
            nn.Linear(head_out_dim, num_syllables),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        with torch.set_grad_enabled(any(p.requires_grad for p in self.encoder.parameters())):
            enc, enc_lengths = self.encoder(x, lengths)

        packed = pack_padded_sequence(
            enc,
            enc_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.head_rnn(packed)
        out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)

        # masked mean + masked max pooling
        B, T, H = out.shape
        mask = (torch.arange(T, device=out.device)[None, :] < out_lengths[:, None]).float()
        mask_exp = mask.unsqueeze(-1)

        mean_pool = (out * mask_exp).sum(dim=1) / out_lengths.clamp(min=1).unsqueeze(-1)

        neg_inf = torch.full_like(out, -1e9)
        max_pool = torch.where(mask_exp.bool(), out, neg_inf).max(dim=1).values

        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        logits = self.classifier(pooled)
        return logits


# ============================================================
# Train / eval helpers
# ============================================================


def ctc_greedy_decode(log_probs: torch.Tensor, out_lengths: torch.Tensor, blank_idx: int = 0):
    """
    log_probs: [T, B, C]
    returns list[list[int]]
    """
    pred = log_probs.argmax(dim=-1).transpose(0, 1)  # [B, T]
    outs = []
    for b in range(pred.size(0)):
        seq = pred[b, : int(out_lengths[b])].tolist()
        collapsed = []
        prev = None
        for x in seq:
            if x != blank_idx and x != prev:
                collapsed.append(int(x))
            prev = x
        outs.append(collapsed)
    return outs



def exact_sequence_accuracy(preds: List[List[int]], targets: List[List[int]]) -> float:
    if len(preds) == 0:
        return 0.0
    hits = [p == t for p, t in zip(preds, targets)]
    return float(np.mean(hits))



def topk_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, ks=(1, 3, 5, 10)):
    max_k = min(max(ks), logits.size(1))
    topk = logits.topk(max_k, dim=1).indices
    out = {}
    for k in ks:
        k_eff = min(k, logits.size(1))
        hits = (topk[:, :k_eff] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"top{k}"] = float(hits)
    return out



def _to_device(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out



def _run_unit_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 5.0):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_n = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = _to_device(batch, device)
        log_probs, out_lengths, _ = model(batch["x"], batch["x_lens"])
        loss = criterion(
            log_probs,
            batch["y_units"],
            out_lengths,
            batch["y_units_lens"],
        )

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = batch["x"].size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        preds = ctc_greedy_decode(log_probs.detach().cpu(), out_lengths.detach().cpu(), blank_idx=0)
        all_preds.extend(preds)
        all_targets.extend(batch["y_units_list"])

    metrics = {
        "loss": total_loss / max(total_n, 1),
        "unit_seq_acc": exact_sequence_accuracy(all_preds, all_targets),
    }
    return metrics



def _run_syll_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 5.0):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_n = 0
    all_logits = []
    all_targets = []

    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["x"], batch["x_lens"])
        loss = criterion(logits, batch["y_syll"])

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = batch["x"].size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
        all_logits.append(logits.detach().cpu())
        all_targets.append(batch["y_syll"].detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    topk = topk_accuracy_from_logits(logits, targets, ks=(1, 3, 5, 10))

    metrics = {"loss": total_loss / max(total_n, 1), **topk}
    return metrics


# ============================================================
# Public train / eval API
# ============================================================


def train_unit_ctc(
    model: UnitCTCModel,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    cfg: TrainConfig,
):
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_unit_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip)
        if valid_loader is not None:
            valid_metrics = _run_unit_epoch(model, valid_loader, None, criterion, device, cfg.grad_clip)
            score = valid_metrics["unit_seq_acc"]
        else:
            valid_metrics = None
            score = train_metrics["unit_seq_acc"]

        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})

        if score > best_metric:
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, history



def evaluate_unit_ctc(model: UnitCTCModel, loader: DataLoader, cfg: TrainConfig):
    device = torch.device(cfg.device)
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    return _run_unit_epoch(model, loader, None, criterion, device, cfg.grad_clip)



def train_syllable_classifier(
    model: SyllableBiLSTMClassifier,
    train_loader: DataLoader,
    valid_loader: Optional[DataLoader],
    cfg: TrainConfig,
):
    device = torch.device(cfg.device)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg.patience
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = _run_syll_epoch(model, train_loader, optimizer, criterion, device, cfg.grad_clip)
        if valid_loader is not None:
            valid_metrics = _run_syll_epoch(model, valid_loader, None, criterion, device, cfg.grad_clip)
            score = valid_metrics["top1"]
        else:
            valid_metrics = None
            score = train_metrics["top1"]

        history.append({"epoch": epoch, "train": train_metrics, "valid": valid_metrics})

        if score > best_metric:
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, history



def evaluate_syllable_classifier(model: SyllableBiLSTMClassifier, loader: DataLoader, cfg: TrainConfig):
    device = torch.device(cfg.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    return _run_syll_epoch(model, loader, None, criterion, device, cfg.grad_clip)


# ============================================================
# Save / load
# ============================================================


def save_pipeline(
    save_path: str,
    *,
    unit_model: UnitCTCModel,
    syll_model: SyllableBiLSTMClassifier,
    unit2idx: Dict[str, int],
    idx2unit: Dict[int, str],
    syll2idx: Dict[str, int],
    idx2syll: Dict[int, str],
    data_cfg: DataPrepConfig,
    model_cfg: ModelConfig,
    unit_train_cfg: TrainConfig,
    syll_train_cfg: TrainConfig,
    input_dim: int,
):
    payload = {
        "unit_state_dict": unit_model.state_dict(),
        "syll_state_dict": syll_model.state_dict(),
        "unit2idx": unit2idx,
        "idx2unit": idx2unit,
        "syll2idx": syll2idx,
        "idx2syll": idx2syll,
        "data_cfg": asdict(data_cfg),
        "model_cfg": asdict(model_cfg),
        "unit_train_cfg": asdict(unit_train_cfg),
        "syll_train_cfg": asdict(syll_train_cfg),
        "input_dim": int(input_dim),
    }
    torch.save(payload, save_path)



def load_pipeline(save_path: str, map_location: str = "cpu"):
    ckpt = torch.load(save_path, map_location=map_location)

    data_cfg = DataPrepConfig(**ckpt["data_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    unit_train_cfg = TrainConfig(**ckpt["unit_train_cfg"])
    syll_train_cfg = TrainConfig(**ckpt["syll_train_cfg"])

    input_dim = int(ckpt["input_dim"])
    unit2idx = ckpt["unit2idx"]
    idx2unit = ckpt["idx2unit"]
    syll2idx = ckpt["syll2idx"]
    idx2syll = ckpt["idx2syll"]

    unit_model = UnitCTCModel(input_dim=input_dim, num_units=len(unit2idx), model_cfg=model_cfg)
    unit_model.load_state_dict(ckpt["unit_state_dict"])

    syll_model = SyllableBiLSTMClassifier(
        encoder=copy.deepcopy(unit_model.encoder),
        num_syllables=len(syll2idx),
        model_cfg=model_cfg,
        freeze_encoder=False,
    )
    syll_model.load_state_dict(ckpt["syll_state_dict"])

    artifacts = {
        "unit_model": unit_model,
        "syll_model": syll_model,
        "unit2idx": unit2idx,
        "idx2unit": idx2unit,
        "syll2idx": syll2idx,
        "idx2syll": idx2syll,
        "data_cfg": data_cfg,
        "model_cfg": model_cfg,
        "unit_train_cfg": unit_train_cfg,
        "syll_train_cfg": syll_train_cfg,
        "input_dim": input_dim,
    }
    return artifacts


# ============================================================
# Convenience pipeline builder / runner
# ============================================================


def make_dataloaders(
    train_sample_table: pd.DataFrame,
    test_sample_table: pd.DataFrame,
    *,
    data_cfg: DataPrepConfig,
    unit2idx: Dict[str, int],
    syll2idx: Dict[str, int],
    unit_train_cfg: TrainConfig,
    syll_train_cfg: TrainConfig,
):
    # Unit stage
    unit_train_ds = UnitCTCDataset(train_sample_table, unit2idx, data_cfg)
    unit_test_ds = UnitCTCDataset(test_sample_table, unit2idx, data_cfg)

    unit_train_loader = DataLoader(
        unit_train_ds,
        batch_size=unit_train_cfg.batch_size,
        shuffle=True,
        num_workers=unit_train_cfg.num_workers,
        collate_fn=collate_ctc,
    )
    unit_test_loader = DataLoader(
        unit_test_ds,
        batch_size=unit_train_cfg.batch_size,
        shuffle=False,
        num_workers=unit_train_cfg.num_workers,
        collate_fn=collate_ctc,
    )

    # Syllable stage
    syll_train_ds = SyllableDataset(train_sample_table, syll2idx, data_cfg)
    syll_test_ds = SyllableDataset(test_sample_table, syll2idx, data_cfg)

    syll_train_loader = DataLoader(
        syll_train_ds,
        batch_size=syll_train_cfg.batch_size,
        shuffle=True,
        num_workers=syll_train_cfg.num_workers,
        collate_fn=collate_syll,
    )
    syll_test_loader = DataLoader(
        syll_test_ds,
        batch_size=syll_train_cfg.batch_size,
        shuffle=False,
        num_workers=syll_train_cfg.num_workers,
        collate_fn=collate_syll,
    )

    return {
        "unit_train_loader": unit_train_loader,
        "unit_test_loader": unit_test_loader,
        "syll_train_loader": syll_train_loader,
        "syll_test_loader": syll_test_loader,
    }



def run_full_pipeline(
    *,
    train_emb_df: pd.DataFrame,
    test_emb_df: pd.DataFrame,
    train_meta_df: Optional[pd.DataFrame] = None,
    test_meta_df: Optional[pd.DataFrame] = None,
    data_cfg: DataPrepConfig,
    model_cfg: ModelConfig,
    unit_train_cfg: TrainConfig,
    syll_train_cfg: TrainConfig,
    freeze_encoder_for_syll: bool = True,
    save_path: Optional[str] = None,
    seed: int = 42,
):
    """
    End-to-end runner:
      1) build sample tables from train/test emb dfs
      2) train unit CTC encoder
      3) freeze encoder and train syllable classifier
      4) evaluate both stages
      5) optionally save
    """
    set_seed(seed)

    train_sample_table = build_sample_table(
        train_emb_df,
        sample_meta_df=train_meta_df,
        cfg=data_cfg,
        require_units=True,
    )
    test_sample_table = build_sample_table(
        test_emb_df,
        sample_meta_df=test_meta_df,
        cfg=data_cfg,
        require_units=True,
    )

    # input dim from first sample
    input_dim = int(train_sample_table.iloc[0]["emb_seq"].shape[1])

    unit2idx, idx2unit = build_unit_vocab(train_sample_table, data_cfg)
    syll2idx, idx2syll = build_syllable_vocab(train_sample_table, data_cfg)

    loaders = make_dataloaders(
        train_sample_table,
        test_sample_table,
        data_cfg=data_cfg,
        unit2idx=unit2idx,
        syll2idx=syll2idx,
        unit_train_cfg=unit_train_cfg,
        syll_train_cfg=syll_train_cfg,
    )

    # ----------------------------
    # Stage 1: unit CTC
    # ----------------------------
    unit_model = UnitCTCModel(input_dim=input_dim, num_units=len(unit2idx), model_cfg=model_cfg)
    unit_model, unit_train_history = train_unit_ctc(
        unit_model,
        loaders["unit_train_loader"],
        loaders["unit_test_loader"],  # use test as validation only for quick experimentation;
                                       # for production research split out a proper validation set
        unit_train_cfg,
    )

    unit_test_metrics = evaluate_unit_ctc(
        unit_model,
        loaders["unit_test_loader"],
        unit_train_cfg,
    )

    # ----------------------------
    # Stage 2: syllable classifier
    # ----------------------------
    syll_model = SyllableBiLSTMClassifier(
        encoder=copy.deepcopy(unit_model.encoder),
        num_syllables=len(syll2idx),
        model_cfg=model_cfg,
        freeze_encoder=freeze_encoder_for_syll,
    )

    syll_model, syll_train_history = train_syllable_classifier(
        syll_model,
        loaders["syll_train_loader"],
        loaders["syll_test_loader"],  # same caveat as above
        syll_train_cfg,
    )

    syll_test_metrics = evaluate_syllable_classifier(
        syll_model,
        loaders["syll_test_loader"],
        syll_train_cfg,
    )

    if save_path is not None:
        save_pipeline(
            save_path,
            unit_model=unit_model,
            syll_model=syll_model,
            unit2idx=unit2idx,
            idx2unit=idx2unit,
            syll2idx=syll2idx,
            idx2syll=idx2syll,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            unit_train_cfg=unit_train_cfg,
            syll_train_cfg=syll_train_cfg,
            input_dim=input_dim,
        )

    return {
        "train_sample_table": train_sample_table,
        "test_sample_table": test_sample_table,
        "unit2idx": unit2idx,
        "idx2unit": idx2unit,
        "syll2idx": syll2idx,
        "idx2syll": idx2syll,
        "unit_model": unit_model,
        "syll_model": syll_model,
        "unit_train_history": unit_train_history,
        "syll_train_history": syll_train_history,
        "unit_test_metrics": unit_test_metrics,
        "syll_test_metrics": syll_test_metrics,
        "input_dim": input_dim,
        "save_path": save_path,
    }


@torch.no_grad()
def predict_units(
    model: UnitCTCModel,
    loader: DataLoader,
    *,
    cfg: TrainConfig,
) -> List[Dict]:
    """
    Predict unit sequences using greedy CTC decoding.
    Returns a list of per-sample dicts.
    """
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    results = []

    for batch in loader:
        batch = _to_device(batch, device)
        log_probs, out_lengths, enc = model(batch["x"], batch["x_lens"])

        preds = ctc_greedy_decode(
            log_probs.detach().cpu(),
            out_lengths.detach().cpu(),
            blank_idx=0,
        )

        for i in range(len(preds)):
            results.append({
                "sample_index": batch["sample_index"][i],
                "pred_units_idx": preds[i],
                "syllable_gt": batch["syllable"][i],
            })

    return results



@torch.no_grad()
def predict_syllables(
    model: SyllableBiLSTMClassifier,
    loader: DataLoader,
    *,
    cfg: TrainConfig,
    k: int = 5,
) -> List[Dict]:
    """
    Predict syllables with top-k outputs.
    """
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    results = []

    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["x"], batch["x_lens"])
        probs = torch.softmax(logits, dim=-1)

        topk = probs.topk(k=min(k, probs.size(1)), dim=1)

        for i in range(probs.size(0)):
            results.append({
                "sample_index": batch["sample_index"][i],
                "syllable_gt": batch["syllable"][i],
                "topk_idx": topk.indices[i].tolist(),
                "topk_prob": topk.values[i].tolist(),
            })

    return results


@torch.no_grad()
def predict_syllable_topk(
    model: SyllableBiLSTMClassifier,
    loader: DataLoader,
    *,
    idx2syll: Dict[int, str],
    cfg: TrainConfig,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Predict Top-K syllables from a loader.

    Returns a DataFrame with:
      - sample_index
      - pred_top1
      - pred_topk
      - pred_topk_prob
      - true_syllable (if available)
    """
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    rows = []

    for batch in loader:
        batch = _to_device(batch, device)
        logits = model(batch["x"], batch["x_lens"])
        probs = torch.softmax(logits, dim=-1)

        k_eff = min(top_k, probs.size(1))
        topk = probs.topk(k_eff, dim=1)

        for i in range(probs.size(0)):
            topk_idx = topk.indices[i].detach().cpu().tolist()
            topk_prob = topk.values[i].detach().cpu().tolist()
            topk_syll = [idx2syll[int(j)] for j in topk_idx]

            row = {
                "sample_index": int(batch["sample_index"][i]),
                "pred_top1": topk_syll[0] if len(topk_syll) > 0 else None,
                "pred_topk": topk_syll,
                "pred_topk_prob": topk_prob,
            }

            if "syllable" in batch:
                row["true_syllable"] = str(batch["syllable"][i])

            rows.append(row)

    return pd.DataFrame(rows)

def predict_from_emb_df(
    emb_df: pd.DataFrame,
    *,
    syll_model: SyllableBiLSTMClassifier,
    idx2syll: Dict[int, str],
    data_cfg: DataPrepConfig,
    infer_cfg: TrainConfig,
    sample_meta_df: Optional[pd.DataFrame] = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    End-to-end prediction directly from long-form embedding dataframe.

    Works with or without labels.
    If sample_meta_df is provided (or emb_df has syllable column), returns true_syllable too.
    """
    sample_table = build_inference_sample_table(
        emb_df,
        sample_meta_df=sample_meta_df,
        cfg=data_cfg,
    )

    ds = InferenceDataset(sample_table, data_cfg)
    loader = DataLoader(
        ds,
        batch_size=infer_cfg.batch_size,
        shuffle=False,
        num_workers=infer_cfg.num_workers,
        collate_fn=collate_infer,
    )

    pred_df = predict_syllable_topk(
        syll_model,
        loader,
        idx2syll=idx2syll,
        cfg=infer_cfg,
        top_k=top_k,
    )
    return pred_df

def topk_accuracy_from_pred_df(pred_df: pd.DataFrame, k: int):
    """
    pred_df must contain:
      - true_syllable
      - pred_topk
    """
    if "true_syllable" not in pred_df.columns:
        raise KeyError("pred_df must contain 'true_syllable' for evaluation")

    hits = []
    for _, row in pred_df.iterrows():
        true = str(row["true_syllable"])
        preds = [str(x) for x in row["pred_topk"][:k]]
        hits.append(true in preds)

    return float(np.mean(hits)) if len(hits) > 0 else np.nan

def benchmark_single_sample_runtime(
    emb_df: pd.DataFrame,
    *,
    syll_model: SyllableBiLSTMClassifier,
    idx2syll: Dict[int, str],
    data_cfg: DataPrepConfig,
    infer_cfg: TrainConfig,
    sample_meta_df: Optional[pd.DataFrame] = None,
    top_k: int = 5,
    runs: int = 20,
    warmup: int = 2,
) -> pd.DataFrame:
    """
    Benchmark end-to-end prediction time for ONE sample dataframe.

    emb_df should contain one sample only.
    """
    # warmup
    for _ in range(warmup):
        _ = predict_from_emb_df(
            emb_df,
            syll_model=syll_model,
            idx2syll=idx2syll,
            data_cfg=data_cfg,
            infer_cfg=infer_cfg,
            sample_meta_df=sample_meta_df,
            top_k=top_k,
        )

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = predict_from_emb_df(
            emb_df,
            syll_model=syll_model,
            idx2syll=idx2syll,
            data_cfg=data_cfg,
            infer_cfg=infer_cfg,
            sample_meta_df=sample_meta_df,
            top_k=top_k,
        )
        times.append(time.perf_counter() - t0)

    times = np.asarray(times, dtype=np.float64)

    return pd.DataFrame([{
        "runs": int(runs),
        "mean_ms": float(times.mean() * 1000.0),
        "p50_ms": float(np.percentile(times, 50) * 1000.0),
        "p95_ms": float(np.percentile(times, 95) * 1000.0),
        "max_ms": float(times.max() * 1000.0),
    }])


def benchmark_dataset_runtime(
    emb_df: pd.DataFrame,
    *,
    syll_model: SyllableBiLSTMClassifier,
    idx2syll: Dict[int, str],
    data_cfg: DataPrepConfig,
    infer_cfg: TrainConfig,
    sample_meta_df: Optional[pd.DataFrame] = None,
    top_k: int = 5,
    sample_limit: Optional[int] = None,
    runs_per_sample: int = 5,
) -> pd.DataFrame:
    """
    Benchmark runtime over multiple samples.

    Returns one row per sample with mean/p95 timing.
    """
    s_col = data_cfg.sample_index_col
    sample_ids = emb_df[s_col].drop_duplicates().astype(int).tolist()
    if sample_limit is not None:
        sample_ids = sample_ids[:sample_limit]

    rows = []

    for sid in sample_ids:
        one_emb_df = emb_df[emb_df[s_col] == sid].copy()

        one_meta_df = None
        if sample_meta_df is not None and s_col in sample_meta_df.columns:
            one_meta_df = sample_meta_df[sample_meta_df[s_col] == sid].copy()

        bench_df = benchmark_single_sample_runtime(
            one_emb_df,
            syll_model=syll_model,
            idx2syll=idx2syll,
            data_cfg=data_cfg,
            infer_cfg=infer_cfg,
            sample_meta_df=one_meta_df,
            top_k=top_k,
            runs=runs_per_sample,
            warmup=1,
        )

        row = bench_df.iloc[0].to_dict()
        row["sample_index"] = sid
        rows.append(row)

    return pd.DataFrame(rows)


def build_infer_dataloader(
    emb_df: pd.DataFrame,
    *,
    data_cfg: DataPrepConfig,
    train_cfg: TrainConfig,
    sample_meta_df: Optional[pd.DataFrame] = None,
):
    """
    Build inference sample table + dataloader.
    """
    sample_table = build_inference_sample_table(
        emb_df,
        sample_meta_df=sample_meta_df,
        cfg=data_cfg,
    )

    infer_ds = InferenceDataset(sample_table, data_cfg)

    loader = DataLoader(
        infer_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_infer,
    )

    return sample_table, loader


__all__ = [
    "DataPrepConfig",
    "ModelConfig",
    "TrainConfig",
    "set_seed",
    "build_sample_table",
    "build_inference_sample_table",
    "build_unit_vocab",
    "build_syllable_vocab",
    "UnitCTCDataset",
    "SyllableDataset",
    "InferenceDataset",
    "collate_ctc",
    "collate_syll",
    "collate_infer",
    "StrokeBiLSTMEncoder",
    "UnitCTCModel",
    "SyllableBiLSTMClassifier",
    "train_unit_ctc",
    "evaluate_unit_ctc",
    "train_syllable_classifier",
    "evaluate_syllable_classifier",
    "save_pipeline",
    "load_pipeline",
    "make_dataloaders",
    "run_full_pipeline",

    # prediction
    "predict_units",
    "predict_syllables",
    "predict_syllable_topk",
    "build_infer_dataloader",
    "predict_from_emb_df",

    # eval
    "topk_accuracy_from_pred_df",

    # runtime benchmark
    "benchmark_single_sample_runtime",
    "benchmark_dataset_runtime",
]
