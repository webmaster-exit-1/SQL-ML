import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
import joblib

def normalize_sql_log(log_text: str) -> str:
    """Minimal normalization to preserve special SQL/Tamper characters."""
    if not isinstance(log_text, str): return ""
    # We lowercase to handle RandomCase, but keep symbols for Tamper detection
    log = log_text.lower()
    log = re.sub(r'\s+', ' ', log).strip()
    return log

class SQLMLDetector:
    def __init__(self, model_path='sql_model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # FEATURE UNION: This is the secret sauce.
        # It looks for WORDS (like 'select') AND CHAR-SEQUENCES (like '/**/')
        self.vectorizer = FeatureUnion([
            ('word_features', TfidfVectorizer(
                ngram_range=(1, 3), 
                analyzer='word', 
                min_df=2
            )),
            ('char_features', TfidfVectorizer(
                ngram_range=(2, 5), 
                analyzer='char', 
                min_df=2
            ))
        ])
        
        # Increased estimators for better accuracy with complex tampers
        self.model = RandomForestClassifier(n_estimators=250, random_state=42)

    def train(self, csv_path="training_data.csv"):
        if not os.path.exists(csv_path):
            print(f"[!] {csv_path} not found. Run sdg.py first.")
            return False
        
        df = pd.read_csv(csv_path)
        print(f"[*] Training on {len(df)} samples (Obfuscation-Aware)...")
        
        # Apply normalization to the training set
        X = self.vectorizer.fit_transform(df['text'].apply(normalize_sql_log))
        self.model.fit(X, df['label'])
        
        self.save_model()
        print("[+] Training complete. Model saved.")
        return True

    def save_model(self):
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

    def _process_single_log(self, log: str):
        """Used by the worker to predict a single incoming log entry."""
        clean_log = normalize_sql_log(log)
        features = self.vectorizer.transform([clean_log])
        prediction = self.model.predict(features)[0]
        return int(prediction)

if __name__ == "__main__":
    detector = SQLMLDetector()
    detector.train()
