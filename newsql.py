
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def normalize_sql_log(log_text: str) -> str:
    if not isinstance(log_text, str): return ""
    log = log_text.lower()
    log = re.sub(r'\b\d+\b', 'NUM', log)
    log = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", 'STR', log)
    log = re.sub(r'0x[0-9a-f]+', 'HEX', log)
    log = re.sub(r'\s+', ' ', log).strip()
    return log

class SQLMLDetector:
    def __init__(self, model_path='sql_model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, csv_path="training_data.csv"):
        if not os.path.exists(csv_path):
            print(f"[-] Error: {csv_path} not found.")
            return False
        
        df = pd.read_csv(csv_path)
        print(f"[*] Training on {len(df)} samples...")
        df['text'] = df['text'].apply(normalize_sql_log)
        X = self.vectorizer.fit_transform(df['text'])
        self.model.fit(X, df['label'])
        self.save_model()
        return True

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"[+] Artifacts saved: {self.model_path}, {self.vectorizer_path}")

    def load_model(self):
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def _process_single_log(self, log: str):
        clean_log = normalize_sql_log(log)
        X = self.vectorizer.transform([clean_log])
        prediction = self.model.predict(X)[0]
        return "MALICIOUS" if prediction == 1 else "SAFE"

if __name__ == "__main__":
    detector = SQLMLDetector()
    if not detector.train():
        exit(1)
