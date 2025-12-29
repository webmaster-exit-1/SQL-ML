import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib # To save/load the model

def normalize_sql_log(log_text: str) -> str:
    log = log_text.lower()
    log = re.sub(r'\b\d+\b', 'NUM', log)
    log = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", 'STR', log)
    log = re.sub(r'0x[0-9a-f]+', 'HEX', log)
    log = re.sub(r'\s+', ' ', log).strip()
    return log

class SQLMLDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = RandomForestClassifier()

    def train(self, csv_path="training_data.csv"):
        df = pd.read_csv(csv_path)
        # Apply normalization to training data
        df['text'] = df['text'].apply(normalize_sql_log)
        X = self.vectorizer.fit_transform(df['text'])
        self.model.fit(X, df['label'])
        self.save_model()

    def save_model(self):
        joblib.dump(self.model, 'sql_model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')

    def load_model(self):
        self.model = joblib.load('sql_model.pkl')
        self.vectorizer = joblib.load('vectorizer.pkl')

    def _process_single_log(self, log: str):
        clean_log = normalize_sql_log(log)
        X = self.vectorizer.transform([clean_log])
        prediction = self.model.predict(X)[0]
        # Return a simple verdict for the worker
        return "MALICIOUS" if prediction == 1 else "SAFE"

if __name__ == "__main__":
    detector = SQLMLDetector()
    print("[*] Training model on training_data.csv...")
    detector.train()
    print("[+] Model saved as sql_model.pkl")

