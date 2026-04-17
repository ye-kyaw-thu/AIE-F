import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser(description="Five ML Baseline Classifiers with Imputation")
    parser.add_argument("--input", required=True, help="Path to the dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--method", required=True, 
                        choices=['dt', 'rf', 'svm', 'nb', 'lr'], 
                        help="dt: DecisionTree, rf: RandomForest, svm: SVM, nb: NaiveBayes, lr: LogReg")
    parser.add_argument("--plot", default="ml_confusion_matrix.png", help="PNG output filename")
    parser.add_argument("--exclude", nargs='*', help="Columns to exclude")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--kernel", default="rbf")
    
    args = parser.parse_args()

    # 1. Load Data
    ext = os.path.splitext(args.input)[1].lower()
    df = pd.read_excel(args.input) if ext in ['.xlsx', '.xls'] else pd.read_csv(args.input)
    
    # 2. Preprocess
    if args.exclude:
        to_drop = [c for c in args.exclude if c in df.columns]
        df = df.drop(columns=to_drop)
    
    # Drop rows where target itself is missing
    df = df.dropna(subset=[args.target])
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Handle Categorical Features (One-Hot Encoding)
    X = pd.get_dummies(X)
    
    # NEW: Handle Missing Values (NaN) in Features
    # Use 'median' for numeric data which is safer for outliers
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    class_names = le.classes_

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # 4. Select Method
    if args.method == 'dt':
        model = DecisionTreeClassifier(random_state=42)
    elif args.method == 'rf':
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    elif args.method == 'svm':
        model = SVC(kernel=args.kernel, probability=True)
    elif args.method == 'nb':
        model = GaussianNB()
    elif args.method == 'lr':
        model = LogisticRegression(max_iter=2000)

    # 5. Train and Predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6. Evaluation
    print(f"\n--- {args.method.upper()} REPORT | {os.path.basename(args.input)} ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{args.method.upper()} - {os.path.basename(args.input)}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(args.plot)
    plt.close()

if __name__ == "__main__":
    main()

