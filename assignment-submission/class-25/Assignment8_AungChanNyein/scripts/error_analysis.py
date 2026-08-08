#!/usr/bin/env python3
"""
error_analysis.py - Detailed error analysis across the three models
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

models = [
    ("Bi-LSTM",     "exp_bilstm"),
    ("Transformer", "exp_transformer"),
    ("ST-GCN",      "exp_stgcn"),
]

def main():
    # ------------------------------------------------------------------
    # 1. Load label map
    # ------------------------------------------------------------------
    label_map_path = Path("data/label_map.json")
    if not label_map_path.exists():
        print("[ERROR] data/label_map.json not found.")
        return

    with open(label_map_path, 'r', encoding='utf-8') as f:
        l2i = json.load(f)
    
    # idx2label has string keys like "0", "1", need to convert to int
    i2l = {int(k): v for k, v in l2i["idx2label"].items()}

    # ------------------------------------------------------------------
    # 2. Collect per-class F1 scores (Handling the BOM in CSVs)
    # ------------------------------------------------------------------
    f1_data = {}
    for name, exp_dir in models:
        csv_path = Path("results") / exp_dir / "evaluation" / "per_class_test.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=1, encoding='utf-8-sig')
            f1_data[name] = df['f1-score'].to_dict()

    all_classes = list(f1_data.get("Bi-LSTM", {}).keys())

    # ------------------------------------------------------------------
    # 3. Find Classes Failed by All Models
    # ------------------------------------------------------------------
    print("=" * 80)
    print(" ERROR ANALYSIS: Myanmar Sign Language Recognition")
    print("=" * 80)

    print("\n[1] CLASSES FAILED BY ALL 3 MODELS (F1 < 1.0 on test set)")
    print("-" * 80)
    
    failed_by_all = []
    for cls in all_classes:
        scores = [f1_data[m].get(cls, 0.0) for m, _ in models]
        if all(s < 1.0 for s in scores):
            failed_by_all.append((cls, scores))

    if failed_by_all:
        print(f"{'Class Name':<50} | {'Bi-LSTM':>7} {'Transf.':>7} {'ST-GCN':>7}")
        print("-" * 80)
        for cls, scores in failed_by_all:
            score_strs = [f"{s*100:.0f}%" for s in scores]
            print(f"{cls:<50} | {score_strs[0]:>7} {score_strs[1]:>7} {score_strs[2]:>7}")
    else:
        print("  None! All models achieved 100% F1-score on all test classes.")

    # ------------------------------------------------------------------
    # 4. Unique Strengths (Only 1 model succeeded where others failed)
    # ------------------------------------------------------------------
    print("\n[2] UNIQUE STRENGTHS (Only this model got F1=100% where others failed)")
    print("-" * 80)
    
    for m_idx, (m_name, _) in enumerate(models):
        unique_success = []
        for cls in all_classes:
            scores = [f1_data[m].get(cls, 0.0) for m, _ in models]
            
            if (scores[m_idx] == 1.0 and 
                all(scores[i] < 1.0 for i in range(len(models)) if i != m_idx)):
                unique_success.append(cls)

        if unique_success:
            print(f"\n  {m_name} ({len(unique_success)} classes):")
            for cls in unique_success[:10]:
                print(f"    - {cls}")

    # ------------------------------------------------------------------
    # 5. Detailed Error Table (True Gloss vs Predicted Gloss)
    # ------------------------------------------------------------------
    print("\n[3] DETAILED ERROR TABLE (True Gloss -> Predicted Gloss)")
    print("-" * 95)
    print(f"{'Model':<10} | {'True Gloss':<40} | {'Pred Gloss':<40} | {'Video Path':<30}")
    print("-" * 95)

    for m_name, exp_dir in models:
        pred_path = Path("results") / exp_dir / "evaluation" / "predictions_test.csv"
        if not pred_path.exists():
            continue
            
        df = pd.read_csv(pred_path, encoding='utf-8-sig')
        
        errors = df[df['correct'] == False]
        
        if errors.empty:
            print(f"\n  {m_name}: No errors on test set.")
            continue
            
        print(f"\n  {m_name} ({len(errors)} errors):")
        for _, row in errors.head(10).iterrows():
            true_g = row['true_label']
            pred_g = row['pred_label']
            conf = row['confidence']
            vid = row['video_path']
            print(f"    True: {true_g:<40} -> Pred: {pred_g:<40} | {conf*100:.1f}% | {vid}")

    # ------------------------------------------------------------------
    # 6. Top Confusion Pairs (from confusion matrix)
    # ------------------------------------------------------------------
    print("\n[4] TOP CONFUSION PAIRS PER MODEL (True -> Predicted)")
    print("-" * 80)
    
    for m_name, exp_dir in models:
        npy_path = Path("results") / exp_dir / "evaluation" / "confusion_matrix_test.npy"
        if not npy_path.exists():
            continue
            
        cm = np.load(npy_path)
        # Fix: Pass the array as the first argument
        np.fill_diagonal(cm, 0)
        
        flat_idx = np.argsort(cm, axis=None)[-5:][::-1]
        
        print(f"\n  {m_name}:")
        print(f"  {'True Class':<40} {'Predicted Class':<40} {'Count':>5}")
        print("  " + "-" * 75)
        
        for idx in flat_idx:
            true_idx = idx // cm.shape[0]
            pred_idx = idx % cm.shape[1]
            count = cm[true_idx, pred_idx]
            true_name = i2l.get(true_idx, f"Unknown_{true_idx}")
            pred_name = i2l.get(pred_idx, f"Unknown_{pred_idx}")
            print(f"    {true_name:<40} -> {pred_name:<40} | {count:.0f} times")

if __name__ == "__main__":
    main()

