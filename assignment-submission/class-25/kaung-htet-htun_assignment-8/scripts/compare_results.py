#!/usr/bin/env python3
"""
compare_results.py - Print a comparison table of Bi-LSTM, Transformer, and ST-GCN
"""

import json
from pathlib import Path

def main():
    models = [
        ("Bi-LSTM",     "exp_bilstm"),
        ("Transformer", "exp_transformer"),
        ("ST-GCN",      "exp_stgcn"),
    ]

    # Table Header
    # Using monospaced alignment
    header = (
        f"{'Model':<13} | "
        f"{'Val Top-1':>9} {'Val Top-5':>9} {'Val F1':>8} | "
        f"{'Test Top-1':>10} {'Test Top-5':>10} {'Test Prec':>10} {'Test Rec':>9} {'Test F1':>8}"
    )
    print("=" * len(header))
    print("          Model Performance Comparison (Myanmar Sign Language)")
    print("=" * len(header))
    print(header)
    print("─" * len(header))

    for name, exp_dir in models:
        base = Path("results") / exp_dir / "evaluation"
        val_m = base / "metrics_val.json"
        tst_m = base / "metrics_test.json"

        # Initialize with dashes in case files are missing
        val_top1 = val_top5 = val_f1 = "—"
        tst_top1 = tst_top5 = tst_prec = tst_rec = tst_f1 = "—"

        # Read Validation metrics
        if val_m.exists():
            with open(val_m, 'r') as f:
                d = json.load(f)
            val_top1 = f"{d.get('top1_accuracy', 0.0)*100:.2f}%"
            val_top5 = f"{d.get('top5_accuracy', 0.0)*100:.2f}%"
            val_f1   = f"{d.get('f1_macro', 0.0)*100:.2f}%"

        # Read Test metrics
        if tst_m.exists():
            with open(tst_m, 'r') as f:
                d = json.load(f)
            tst_top1 = f"{d.get('top1_accuracy', 0.0)*100:.2f}%"
            tst_top5 = f"{d.get('top5_accuracy', 0.0)*100:.2f}%"
            tst_prec = f"{d.get('precision_macro', 0.0)*100:.2f}%"
            tst_rec  = f"{d.get('recall_macro', 0.0)*100:.2f}%"
            tst_f1   = f"{d.get('f1_macro', 0.0)*100:.2f}%"

        row = (
            f"{name:<13} | "
            f"{val_top1:>9} {val_top5:>9} {val_f1:>8} | "
            f"{tst_top1:>10} {tst_top5:>10} {tst_prec:>10} {tst_rec:>9} {tst_f1:>8}"
        )
        print(row)

    print("=" * len(header))

if __name__ == "__main__":
    main()

