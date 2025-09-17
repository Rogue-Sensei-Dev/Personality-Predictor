import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MAPPING = {"Agree":4, "Somewhat Agree":3, "Somewhat Disagree":2, "Disagree":1, "4":4,"3":3,"2":2,"1":1}

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # assume first 20 columns are questions
    question_cols = df.columns[:20]
    for col in question_cols:
        df[col] = df[col].map(MAPPING)
    if df[question_cols].isnull().any().any():
        raise ValueError("Found unmapped responses in data. Ensure allowed responses.")
    X = df[question_cols].values
    y = df[df.columns[20]].values
    return X, y, question_cols

def train(data_path, out_dir='.', n_estimators=200):
    print("Loading data from:", data_path)
    X, y, question_cols = load_and_preprocess(data_path)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=le.classes_))
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.pkl"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    np.save(os.path.join(out_dir, "features.npy"), question_cols.values)
    print("Saved model and encoder to", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to CSV data')
    parser.add_argument('--out', default='.', help='output directory')
    args = parser.parse_args()
    train(args.data, args.out)
    