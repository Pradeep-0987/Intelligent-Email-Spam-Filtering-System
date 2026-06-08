"""
train.py — Train and save the spam classification model.

Usage:
    python train.py

Output:
    models/spam_model.pkl   — trained pipeline
    models/metrics.json     — performance metrics for all models
"""

import pandas as pd
import numpy as np
import joblib
import json
import re
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, 'data', 'spam.tsv')
MODEL_PATH  = os.path.join(BASE, 'models', 'spam_model.pkl')
METRIC_PATH = os.path.join(BASE, 'models', 'metrics.json')

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)


def clean(text: str) -> str:
    """Preprocess a single message."""
    text = text.lower()
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_data():
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'])
    df['label_num'] = (df['label'] == 'spam').astype(int)
    df['clean'] = df['message'].apply(clean)
    print(f"Loaded {len(df):,} samples  |  spam: {df['label_num'].sum():,}  ham: {(df['label_num']==0).sum():,}")
    return df


def build_models():
    return {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ('clf',   MultinomialNB(alpha=0.1))
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ('clf',   LogisticRegression(max_iter=1000, C=5, random_state=42))
        ]),
        'Linear SVM': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ('clf',   LinearSVC(C=1.0, max_iter=2000, random_state=42))
        ]),
    }


def train():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean'], df['label_num'],
        test_size=0.2, random_state=42, stratify=df['label_num']
    )

    models  = build_models()
    results = {}
    best_name, best_model, best_f1 = None, None, 0.0

    print("\n{'Model':<22} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'CV-F1':>8}")
    print("-" * 68)

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec  = recall_score(y_test, preds)
        f1   = f1_score(y_test, preds)
        cm   = confusion_matrix(y_test, preds).tolist()
        cv   = cross_val_score(pipe, df['clean'], df['label_num'], cv=5, scoring='f1').mean()

        print(f"{name:<22} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {cv:>8.4f}")

        results[name] = dict(
            accuracy=round(acc, 4), precision=round(prec, 4),
            recall=round(rec, 4), f1=round(f1, 4),
            cv_f1=round(cv, 4), confusion_matrix=cm
        )

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, pipe

    print(f"\n✅ Best model: {best_name}  (F1={best_f1:.4f})")
    print("\nClassification Report (best model):")
    print(classification_report(y_test, best_model.predict(X_test), target_names=['Ham', 'Spam']))

    # Save
    joblib.dump(best_model, MODEL_PATH)
    with open(METRIC_PATH, 'w') as f:
        json.dump({
            'best_model':    best_name,
            'results':       results,
            'total_samples': len(df),
            'spam_count':    int(df['label_num'].sum()),
            'ham_count':     int((df['label_num'] == 0).sum())
        }, f, indent=2)

    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {METRIC_PATH}")


if __name__ == '__main__':
    train()
