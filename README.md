# Personality Predictor (Flask)

This is a Likert-scale based personality predictor (4 types: Introvert, Extrovert, Thinker, Feeler).
It includes:
- 20 survey questions (CSV demo data included)
- Flask app to serve the survey and show predictions
- Training script to train a RandomForest model from the CSV
- Nice UI using Bootstrap

Quick start:
1. Create & activate venv:
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model (uses included survey_data.csv):
   python model/train_model.py --data ../survey_data.csv --out .

4. Run the app:
   python app.py
   Open http://127.0.0.1:5000
