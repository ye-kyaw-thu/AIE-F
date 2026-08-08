#!/usr/bin/env python3
'''
train_extra_models.py - Standalone trainer for BiLSTM-Attention and CNN-GRU

Bug fixes (v7):
  - entry['path'] -> entry['keypoint_path']  (manifest key confirmed from augment.py)
  - entry['label'] is a string; pass label_map to Dataset and convert to int
  - assert num_classes >= 500  (detects stale 3-class label_map from keypoints dataset)
'''
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from bilstm_attention import BiLSTMAttention
from cnn_gru import CNNGRU


class KeypointDataset(Dataset):
    def __init__(self, manifest_entries, label_map, max_seq_len=200):
        self.entries   = manifest_entries
        self.label_map = label_map      # str -> int
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        seq = np.load(entry['keypoint_path'])   # FIX: was entry['path']
        seq = seq.reshape(seq.shape[0], -1)     # (T, 75, 3) -> (T, 225)

        T = seq.shape[0]
        if T > self.max_seq_len:
            seq = seq[:self.max_seq_len]
            T   = self.max_seq_len

        label_int = self.label_map[entry['label']]  # FIX: string -> int
        return torch.from_numpy(seq).float(), label_int, T


def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    max_len  = max(lengths)
    feat_dim = seqs[0].shape[1]
    padded   = torch.zeros(len(seqs), max_len, feat_dim)
    mask     = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, (seq, length) in enumerate(zip(seqs, lengths)):
        padded[i, :length] = seq
        mask[i,   :length] = True
    labels  = torch.tensor(labels,  dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, labels, lengths, mask


def build_model(model_name, num_classes, mcfg):
    if model_name == 'bilstm_attn':
        return BiLSTMAttention(
            input_dim  = mcfg.get('input_dim',  225),
            hidden_dim = mcfg.get('hidden_dim', 256),
            num_layers = mcfg.get('num_layers', 3),
            num_classes= num_classes,
            dropout    = mcfg.get('dropout', 0.4),
            num_heads  = mcfg.get('num_heads', 4),
        )
    elif model_name == 'cnn_gru':
        return CNNGRU(
            input_dim   = mcfg.get('input_dim',    225),
            cnn_channels= mcfg.get('cnn_channels', 128),
            gru_hidden  = mcfg.get('gru_hidden',   256),
            gru_layers  = mcfg.get('gru_layers',   2),
            num_classes = num_classes,
            dropout     = mcfg.get('dropout', 0.3),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    top5_correct = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, lengths, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, lengths=lengths, mask=mask)
            loss   = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

            top5 = logits.topk(min(5, logits.size(1)), dim=1).indices
            top5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    from sklearn.metrics import precision_recall_fscore_support
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)

    return {
        'loss':               total_loss / max(total, 1),
        'top1_accuracy':      correct    / max(total, 1),
        'top5_accuracy':      top5_correct / max(total, 1),
        'precision_macro':    prec_m, 'recall_macro':    rec_m, 'f1_macro':    f1_m,
        'precision_weighted': prec_w, 'recall_weighted': rec_w, 'f1_weighted': f1_w,
        'num_samples': total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  required=True, choices=['bilstm_attn', 'cnn_gru'])
    ap.add_argument('--exp',    required=True)
    ap.add_argument('--config', default='config/config.yaml')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg, tcfg = cfg['data'], cfg['training']
    mcfg = cfg['model'].get(args.model, {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(dcfg['label_map_file']) as f:
        raw_map = json.load(f)
    # Teacher's train.py calls save_label_map which writes:
    #   {'label2idx': {...}, 'idx2label': {...}, 'num_classes': N}
    # run_updated_augment.py writes flat {label: int} â€” handle both
    if 'label2idx' in raw_map:
        label_map = raw_map['label2idx']
    else:
        label_map = raw_map
    num_classes = len(label_map)
    print(f"Classes: {num_classes}")
    assert num_classes >= 500, (
        f"label_map has only {num_classes} classes after format detection. "
        f"Re-run Step 7 and teacher models first."
    )

    with open(f"{dcfg['augmented_dir']}/augmented_manifest.json") as f:
        manifest = json.load(f)

    train_ds = KeypointDataset(manifest['train'], label_map, dcfg.get('max_seq_len', 200))
    val_ds   = KeypointDataset(manifest['val'],   label_map, dcfg.get('max_seq_len', 200))
    test_ds  = KeypointDataset(manifest['test'],  label_map, dcfg.get('max_seq_len', 200))

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=tcfg.get('batch_size', 32),
                              shuffle=True, drop_last=True, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg.get('batch_size', 32),
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=tcfg.get('batch_size', 32),
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = build_model(args.model, num_classes, mcfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get('label_smoothing', 0.1))
    optimizer  = torch.optim.AdamW(
        model.parameters(),
        lr           = tcfg.get('learning_rate', 0.0005),
        weight_decay = tcfg.get('weight_decay',  0.0001),
    )
    num_epochs = tcfg.get('num_epochs', 150)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg.get('cosine_T_max', num_epochs),
        eta_min=tcfg.get('cosine_eta_min', 0.000005),
    )

    out_dir = Path('results') / args.exp
    (out_dir / 'evaluation').mkdir(parents=True, exist_ok=True)

    best_val_f1       = 0.0
    patience          = tcfg.get('patience', 25)
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_top1': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y, lengths, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths=lengths, mask=mask)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get('grad_clip', 1.0))
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        train_loss  = total_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, criterion)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_top1'].append(val_metrics['top1_accuracy'])

        print(f"Epoch {epoch+1}/{num_epochs} | train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_top1={val_metrics['top1_accuracy']:.4f} "
              f"val_f1={val_metrics['f1_macro']:.4f}")

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1       = val_metrics['f1_macro']
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_dir / 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss'); axes[0].legend()
    axes[1].plot(history['val_top1'], label='val top1 acc')
    axes[1].plot(history['val_f1'],   label='val F1 macro')
    axes[1].set_title('Validation metrics'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=120)
    print(f"Saved: {out_dir}/training_curves.png")

    model.load_state_dict(torch.load(out_dir / 'best_model.pt'))
    val_final  = evaluate(model, val_loader,  device, criterion)
    test_final = evaluate(model, test_loader, device, criterion)

    val_final['split']       = 'val'
    val_final['num_classes'] = num_classes
    test_final['split']      = 'test'
    test_final['num_classes']= num_classes

    with open(out_dir / 'evaluation' / 'metrics_val.json',  'w') as f:
        json.dump(val_final,  f, indent=2)
    with open(out_dir / 'evaluation' / 'metrics_test.json', 'w') as f:
        json.dump(test_final, f, indent=2)

    print()
    print("="*60)
    print(f"FINAL RESULTS â€” {args.exp}")
    print(f"  Val  top1={val_final['top1_accuracy']:.4f}  F1={val_final['f1_macro']:.4f}")
    print(f"  Test top1={test_final['top1_accuracy']:.4f}  F1={test_final['f1_macro']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()