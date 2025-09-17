from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import os
import csv

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-secret"

MODEL_PATH = os.path.join("model", "model.pkl")
LE_PATH = os.path.join("model", "label_encoder.pkl")
FEATURES_PATH = os.path.join("model", "features.npy")

VALUE_MAP = {"4":4, "3":3, "2":2, "1":1, "Agree":4, "Somewhat Agree":3, "Somewhat Disagree":2, "Disagree":1}

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH) or not os.path.exists(FEATURES_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    features = np.load(FEATURES_PATH, allow_pickle=True)
    return model, le, features

def load_questions():
    # load question texts from survey_data.csv header (first row)
    qtexts = []
    with open("survey_data.csv", newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
        if rows:
            header = rows[0]
            # If header length >=20 treat as q1..q20 text
            if len(header) >= 20:
                qtexts = header[:20]
    if not qtexts:
        qtexts = [f"Question {i+1}" for i in range(20)]
    return qtexts

@app.route("/", methods=["GET"])
def index():
    questions = load_questions()
    return render_template("index.html", questions=enumerate(questions, start=1))

@app.route("/predict", methods=["POST"])
def predict():
    # load model if exists
    model, le, features = load_model()
    answers = []
    # assume 20 questions
    for i in range(1, 21):
        val = request.form.get(f"q{i}")
        if val is None:
            flash("Please answer all questions.", "danger")
            return redirect(url_for("index"))
        if val not in VALUE_MAP:
            flash("Invalid answer detected.", "danger")
            return redirect(url_for("index"))
        answers.append(VALUE_MAP[val])
    X = np.array([answers])
    if model is None:
        # If model not trained yet, show a demo prediction using simple rules
        s = X.sum()
        if s >= 70:
            pred = "Extrovert"
        elif s >= 55:
            pred = "Thinker"
        elif s >= 40:
            pred = "Feeler"
        else:
            pred = "Introvert"
        label_probs = { "Introvert":0.2, "Extrovert":0.4, "Thinker":0.2, "Feeler":0.2 }
        feature_info = [{"feature":f"q{i+1}", "importance":0.01, "answer":int(answers[i])} for i in range(20)]
        return render_template("result.html", prediction=pred, label_probs=label_probs, feature_info=feature_info)
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    label_probs = { lbl: float(probs[i]) for i,lbl in enumerate(le.classes_) }
    importances = model.feature_importances_
    feature_info=[]
    for i, feat in enumerate(features):
        feature_info.append({"feature": feat, "importance": float(importances[i]), "answer": int(answers[i])})
    feature_info = sorted(feature_info, key=lambda x: x["importance"], reverse=True)
    return render_template("result.html", prediction=pred_label, label_probs=label_probs, feature_info=feature_info)

if __name__ == '__main__':
    app.run(debug=True)
