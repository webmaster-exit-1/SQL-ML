import pandas as pd
import random
import os
import sys
import importlib.util
import re
import subprocess

def find_sqlmap_tamper_path():
    """
    Git-safe search for sqlmap.
    1. Checks environment variable 'SQLMAP_HOME'
    2. Checks system PATH
    3. Performs a non-persistent regex search of common tool directories
    """
    # 1. Check for an Environment Variable (The cleanest way for Git)
    # You can set this in your .bashrc: export SQLMAP_HOME="$HOME/Tools/sqlmap"
    env_path = os.environ.get("SQLMAP_HOME")
    if env_path and os.path.exists(os.path.join(env_path, "tamper")):
        return os.path.join(env_path, "tamper")

    # 2. Check system PATH (e.g., /usr/bin/sqlmap)
    sqlmap_bin = subprocess.getoutput("which sqlmap")
    if sqlmap_bin and os.path.exists(sqlmap_bin):
        potential = os.path.dirname(os.path.realpath(sqlmap_bin))
        tp = os.path.join(potential, "tamper")
        if os.path.exists(tp): return tp

    # 3. Clever Regex Search (Generic & Non-Hardcoded)
    home = os.path.expanduser("~")
    sqlmap_regex = re.compile(r'^sqlmap', re.IGNORECASE)
    
    # We look in generic high-level folders
    for folder in ["Tools", "tools", "opt", "src"]:
        root = os.path.join(home, folder)
        if not os.path.exists(root): continue
        
        try:
            for entry in os.scandir(root):
                if entry.is_dir() and sqlmap_regex.match(entry.name):
                    tp = os.path.join(entry.path, "tamper")
                    if os.path.exists(tp): return tp
        except PermissionError:
            continue
            
    return None

def load_sqlmap_tamper(tamper_path, tamper_name):
    """Dynamically imports the actual sqlmap tamper script."""
    if not tamper_path: return None
    file_path = os.path.join(tamper_path, f"{tamper_name}.py")
    if not os.path.exists(file_path): return None
    
    try:
        spec = importlib.util.spec_from_file_location(tamper_name, file_path)
        module = importlib.util.module_from_spec(spec)
        # Add to sys.path so the tamper can find sqlmap's internal 'lib'
        parent_dir = os.path.dirname(tamper_path)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None

# --- Main Logic ---
TAMPER_DIR = find_sqlmap_tamper_path()
loaded_tampers = {}

if TAMPER_DIR:
    print(f"[*] Found sqlmap at a local path. Loading logic...")
    # These match the ones in your high-risk sqlmap command
    scripts = ["space2comment", "randomcase", "charencode", "apostrophemask"]
    for s in scripts:
        mod = load_sqlmap_tamper(TAMPER_DIR, s)
        if mod:
            loaded_tampers[s] = mod
else:
    print("[!] No local sqlmap found. Using default patterns for Git compatibility.")

def apply_real_tampers(payload):
    """Passes payload through actual sqlmap code if available."""
    processed = payload
    for name, module in loaded_tampers.items():
        if hasattr(module, 'tamper'):
            try:
                processed = module.tamper(processed)
            except:
                pass
    return processed

def generate_synthetic_data(count=5000):
    # (Existing data generation logic remains the same)
    malicious_base = [
        "SELECT * FROM users WHERE id='1' UNION SELECT 1,2,3,database(),user()--",
        "OR 1=1--",
        "WAITFOR DELAY '0:0:15'",
        "SLEEP(15)",
        "extractvalue(1,concat(0x7e,@@version))"
    ]
    
    data = []
    for _ in range(count):
        m = random.choice(malicious_base)
        data.append({"text": m, "label": 1}) # Raw
        data.append({"text": apply_real_tampers(m), "label": 1}) # Tampered
        data.append({"text": "Normal request/response data", "label": 0})

    pd.DataFrame(data).to_csv("training_data.csv", index=False)
    print(f"[+] Done. Generated {len(data)} samples.")

if __name__ == "__main__":
    generate_synthetic_data()
