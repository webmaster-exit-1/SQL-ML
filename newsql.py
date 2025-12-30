import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

def normalize_sql_log(log_text: str) -> str:
    """
    Standardizes SQL logs by normalizing case, numbers, strings, and hex values.
    """
    if not isinstance(log_text, str):
        return ""
    
    log = log_text.lower()
    # Replace numbers with 'NUM'
    log = re.sub(r'\b\d+\b', 'NUM', log)
    # Replace string literals with 'STR'
    log = re.sub(r"(['\"])(?:(?=(\\?))\2.)*?\1", 'STR', log)
    # Replace hex values with 'HEX'
    log = re.sub(r'0x[0-9a-f]+', 'HEX', log)
    # Collapse multiple whitespaces into one
    log = re.sub(r'\s+', ' ', log).strip()
    return log

class SQLMLDetector:
    def __init__(self, model_path='sql_model.pkl', vectorizer_path='vectorizer.pkl'):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = TfidfVectorizer()
        # Using Random Forest with fixed state for reproducible CI results
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, csv_path="training_data.csv"):
        """
        Trains the model on the provided dataset and saves the results.
        """
        if not os.path.exists(csv_path):
            print(f"[-] Error: {csv_path} not found. Did you run sdg.py?")
            return False
            
        print(f"[*] Reading training data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print("[*] Pre-processing and normalizing logs...")
        df['text'] = df['text'].apply(normalize_sql_log)
        
        print("[*] Building TF-IDF feature vectors...")
        X = self.vectorizer.fit_transform(df['text'])
        y = df['label']
        
        print("[*] Training Random Forest Classifier...")
        self.model.fit(X, y)
        
        self.save_model()
        return True

    def save_model(self):
        """
        Persists the model and vectorizer to disk.
        """
        print(f"[*] Exporting model to {self.model_path}...")
        joblib.dump(self.model, self.model_path)
        
        print(f"[*] Exporting vectorizer to {self.vectorizer_path}...")
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
        print("[+] SUCCESS: All ML artifacts saved to disk.")

    def load_model(self):
        """
        Loads artifacts for inference.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError("Model artifacts missing. Run 'python newsql.py' first.")
            
        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)
        print("[+] SQL-ML Detector loaded and ready.")

    def _process_single_log(self, log: str):
        """
        Performs inference on a single log string.
        """
        clean_log = normalize_sql_log(log)
        X = self.vectorizer.transform([clean_log])
        prediction = self.model.predict(X)[0]
        return "MALICIOUS" if prediction == 1 else "SAFE"

if __name__ == "__main__":
    # This block allows the CI to run 'python newsql.py' to generate models
    print("--- SQL-ML Training Utility ---")
    detector = SQLMLDetector()
    success = detector.train()
    
    if success:
        print("[!] Ready for deployment.")
    else:
        exit(1)
