## For understanding automatic learning of SCRDR rules to my AI Engineering (Fundamental) Class students
## Written by Ye Kyaw Thu, LU Lab., Myanmar
## Last Updated: 12 April 2026

import pandas as pd
import json
import argparse
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

class RDRNode:
    def __init__(self, condition=None, conclusion=None, cornerstone=None):
        self.condition = condition
        self.conclusion = self._normalize_label(conclusion)
        self.cornerstone = cornerstone
        self.if_true = None  
        self.if_false = None 

    def _normalize_label(self, label):
        if label is None: return None
        s = str(label).strip()
        if s.endswith('.0'): s = s[:-2]
        return s

    def evaluate(self, row):
        if self.condition is None: return True
        col, op, val = self.condition['col'], self.condition['op'], self.condition['val']
        try:
            target_val = row[col]
            if op == '>': return float(target_val) > float(val)
            if op == '<=': return float(target_val) <= float(val)
            if op == '==': return str(target_val) == str(val)
        except:
            return str(row[col]) == str(val)
        return False

    def to_dict(self):
        return {
            "rule": f"{self.condition['col']} {self.condition['op']} {self.condition['val']}" if self.condition else "Default",
            "conclusion": self.conclusion,
            "cornerstone_idx": int(self.cornerstone['original_idx']) if self.cornerstone else "N/A",
            "exception": self.if_true.to_dict() if self.if_true else None,
            "alternative": self.if_false.to_dict() if self.if_false else None
        }

class UniversalAutoRDR:
    def __init__(self, target_col, exclude_cols=None):
        self.target_col = target_col
        self.exclude_cols = exclude_cols if exclude_cols else []
        self.root = None
        self.categorical_cols = []

    def preprocess(self, df):
        df = df.copy()
        to_drop = []
        id_patterns = [r'id$', r'^id', r'no$', r'serial', r'index']
        
        for col in df.columns:
            if col in [self.target_col, 'original_idx']: continue
            if col in self.exclude_cols:
                to_drop.append(col)
                continue
            name_match = any(re.search(p, col.lower()) for p in id_patterns)
            unique_ratio = df[col].nunique() / len(df)
            if (name_match and unique_ratio > 0.9) or (df[col].dtype == 'object' and unique_ratio > 0.8):
                to_drop.append(col)

        df = df.drop(columns=to_drop)
        for col in df.columns:
            if col in [self.target_col, 'original_idx']: continue
            if df[col].dtype == 'object':
                df[col] = df[col].fillna("Unknown").astype(str)
                self.categorical_cols.append(col)
            else:
                df[col] = df[col].fillna(df[col].median())
        return df

    def classify(self, row):
        curr = self.root
        last_satisfied = curr
        while curr:
            if curr.evaluate(row):
                last_satisfied = curr
                curr = curr.if_true 
            else:
                curr = curr.if_false 
        return last_satisfied

    def _induce_rule(self, new_row, cornerstone_row, features):
        best_rule = None
        max_diff = -1
        for col in features:
            v_new, v_old = new_row[col], cornerstone_row[col]
            if str(v_new) == str(v_old): continue
            if col in self.categorical_cols:
                return {'col': col, 'op': '==', 'val': str(v_new)}
            else:
                diff = abs(float(v_new) - float(v_old))
                if diff > max_diff:
                    max_diff = diff
                    threshold = (float(v_new) + float(v_old)) / 2
                    op = '>' if float(v_new) > float(v_old) else '<='
                    best_rule = {'col': col, 'op': op, 'val': round(threshold, 3)}
        return best_rule if best_rule else {'col': features[0], 'op': '==', 'val': str(new_row[features[0]])}

    def train(self, df):
        features = [c for c in df.columns if c not in [self.target_col, 'original_idx']]
        first = df.iloc[0]
        self.root = RDRNode(conclusion=first[self.target_col], cornerstone=first.to_dict())
        for _, row in df.iterrows():
            match_node = self.classify(row)
            actual_label = self.root._normalize_label(row[self.target_col])
            if match_node.conclusion != actual_label:
                new_rule = self._induce_rule(row, match_node.cornerstone, features)
                new_node = RDRNode(condition=new_rule, conclusion=actual_label, cornerstone=row.to_dict())
                if match_node.evaluate(row):
                    if not match_node.if_true: match_node.if_true = new_node
                    else:
                        temp = match_node.if_true
                        while temp.if_false: temp = temp.if_false
                        temp.if_false = new_node
                else:
                    temp = match_node
                    while temp.if_false: temp = temp.if_false
                    temp.if_false = new_node

def main():
    parser = argparse.ArgumentParser(description="Universal Auto-RDR Learner")
    parser.add_argument("--input", required=True, help="CSV or Excel path")
    parser.add_argument("--target", required=True, help="Target column")
    parser.add_argument("--exclude", nargs='*', help="Columns to exclude")
    parser.add_argument("--output", default="rdr_model.json", help="JSON model output")
    parser.add_argument("--plot", default="confusion_matrix.png", help="Confusion matrix PNG output")
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()
    df = pd.read_excel(args.input) if ext in ['.xlsx', '.xls'] else pd.read_csv(args.input)
    df['original_idx'] = range(len(df))

    learner = UniversalAutoRDR(args.target, exclude_cols=args.exclude)
    clean_df = learner.preprocess(df)
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42)

    print(f"Building RDR tree on {len(train_df)} samples...")
    learner.train(train_df)

    with open(args.output, 'w') as f:
        json.dump(learner.root.to_dict(), f, indent=4)

    y_true = [learner.root._normalize_label(x) for x in test_df[args.target]]
    y_pred = [learner.classify(row).conclusion for _, row in test_df.iterrows()]

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('RDR Model Confusion Matrix')
    plt.savefig(args.plot)
    print(f"Confusion matrix saved to {args.plot}")

if __name__ == "__main__":
    main()

