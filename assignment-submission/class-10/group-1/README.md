# SCRDR Classification Experiments

This project implements **Single Classification Ripple Down Rules (SCRDR)** on two datasets:

1. Mental Health Classification
2. Banking Marketing Classification

The goal is to evaluate:
- rule-based learning
- interpretability
- generalization performance

---

# Datasets

## Data Note

Original datasets are not included to reduce repository size.

Processed datasets (train/test splits) are provided for reproducibility.

## 1. Mental Health Dataset
Source: https://www.kaggle.com/datasets/jajidhasan/mental-health

- Binary classification: `Depression (0/1)`
- Contains behavioral, lifestyle, and health-related features
- Used to predict mental health condition

## 2. Banking Marketing Dataset
Source: https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset

- Binary classification: `deposit (yes/no)`
- Data from a Portuguese bank marketing campaign
- Goal: predict whether a client subscribes to a term deposit

---

# Project Structure

```
FINAL/
│
├── banking/
│   ├── data/
│   │   ├── bank.csv
│   │   ├── bank_processed.csv
│   │   ├── train.csv
│   │   └── test.csv
│   ├── log/
│   ├── bank_rules.json
│   ├── auto_rules.json
│   ├── preprocess_bank.py
│   └── run_inter.sh
│
├── mental_health/
│   ├── data/
│   │   ├── mental_health.csv
│   │   ├── mental_health_encoded.csv
│   │   ├── train.csv
│   │   └── test.csv
│   ├── log/
│   ├── depression_rules.json
│   ├── auto_rules.json
│   ├── preprocess_mh.py
│   └── run_inter.sh
│
├── scrdr_interactive.py
├── scrdr_learner.py
├── five_ml.py
└── README.md
```

---

# Setup

Make sure Python 3 is installed.

Install dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn
```

---

# Data Preprocessing

## Mental Health

```bash
cd mental_health
python3 preprocess_mh.py
```

Creates:
- `data/train.csv` (50 rows) — balanced: 25 / 25
- `data/test.csv` (20 rows) — balanced: 10 / 10

For Auto SCRDR and ML baseline, encode the full dataset first:
```bash
python3 -c "
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('./mental_health/data/mental_health.csv')
df = df.drop(columns=['Person_ID'])
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])
df.to_csv('./mental_health/data/mental_health_encoded.csv', index=False)
"
```

## Banking

```bash
cd banking
python3 preprocess_bank.py
```

Creates:
- `data/train.csv` (50 rows)
- `data/test.csv` (20 rows)

For Auto SCRDR and ML baseline, preprocess the full dataset:
```bash
python3 -c "
import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('./banking/data/bank.csv')
df = df.drop(columns=['duration'])
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])
df.to_csv('./banking/data/bank_processed.csv', index=False)
"
```

---

# Run SCRDR (Interactive)

## Mental Health

```bash
cd mental_health
chmod +x run_inter.sh
script log/full_run.log ./run_inter.sh
```

## Banking

```bash
cd banking
chmod +x run_inter.sh
script log/full_run.log ./run_inter.sh
```

---

# Run Auto SCRDR (Automatic Rule Induction)

Trains rules automatically from data without human interaction.

## Mental Health

```bash
cd FINAL
python3 scrdr_learner.py \
  --input ./mental_health/data/mental_health_encoded.csv \
  --target Depression \
  --output ./mental_health/auto_rules.json \
  --plot ./mental_health/cf_auto.png
```

## Banking

```bash
cd FINAL
python3 scrdr_learner.py \
  --input ./banking/data/bank_processed.csv \
  --target deposit \
  --output ./banking/auto_rules.json \
  --plot ./banking/cf_auto.png
```

---

# Run ML Baseline (Random Forest)

Trains a Random Forest classifier for comparison.

## Mental Health

```bash
cd FINAL
python3 five_ml.py \
  --input ./mental_health/data/mental_health_encoded.csv \
  --target Depression \
  --method rf \
  --plot ./mental_health/cf_rf.png
```

## Banking

```bash
cd FINAL
python3 five_ml.py \
  --input ./banking/data/bank_processed.csv \
  --target deposit \
  --method rf \
  --plot ./banking/cf_rf.png
```

---

# Test on Full Dataset (Generalisation Check)

Test your interactively built rules against the full dataset to measure generalisation.

## Mental Health (2000 rows)

```bash
cd FINAL
python3 scrdr_interactive.py \
  --input ./mental_health/data/mental_health.csv \
  --target Depression \
  --exclude Person_ID \
  --tree ./mental_health/depression_rules.json \
  --mode test
```

## Banking (11162 rows)

```bash
cd FINAL
python3 scrdr_interactive.py \
  --input ./banking/data/bank_processed.csv \
  --target deposit \
  --tree ./banking/bank_rules.json \
  --mode test
```

---

# View Results WITHOUT Rerunning

## Option 1: Read saved logs

```bash
cat log/test.log
cat log/full_run.log
```

## Option 2: Re-evaluate using saved rules (small test set)

### Mental Health
```bash
python3 ../scrdr_interactive.py \
  --input ./data/test.csv \
  --target Depression \
  --tree depression_rules.json \
  --mode test
```

### Banking
```bash
python3 ../scrdr_interactive.py \
  --input ./data/test.csv \
  --target deposit \
  --tree bank_rules.json \
  --mode test
```

---

# Results Summary

## Mental Health — Depression Classification

| Method | Test Set | F1 Score |
|---|---|---|
| Interactive SCRDR (human expert) | 2000 rows | **0.55** |
| Auto SCRDR (automatic induction) | 400 rows | 0.54 |
| Random Forest (ML baseline) | 400 rows | 0.58 |

## Banking — Deposit Prediction

| Method | Test Set | F1 Score |
|---|---|---|
| Interactive SCRDR (human expert) | 11162 rows | **0.50** |
| Auto SCRDR (automatic induction) | 2233 rows | 0.53 |
| Random Forest (ML baseline) | 2233 rows | **0.72** |

---

# Key Observations

- SCRDR achieves **very high training accuracy** (1.00 for MH, 0.96 for Banking)
- Test accuracy is significantly lower — the model overfits to training cases by design
- **Mental Health**: Interactive SCRDR nearly matches Random Forest (0.55 vs 0.58) — clear domain signals (Stress_Level, Anxiety) make rule writing effective
- **Banking**: Random Forest significantly outperforms SCRDR (0.72 vs 0.50) — complex feature interactions are hard to capture with simple IF→THEN rules
- Feature correlation matters: rules targeting highly correlated features generalise better

---

# Conclusion

SCRDR provides:
- High interpretability — every decision traces back to a human-written rule
- Incremental rule learning — exceptions are added without breaking existing rules
- Competitive performance when domain signals are clear

However:
- It is sensitive to small datasets
- It overfits when many exception rules are added
- ML (e.g. Random Forest) outperforms SCRDR on complex domains like banking

---

# Author

Group 1 – SCRDR Experiments | AI Engineering (Fundamental) | 18 April 2026