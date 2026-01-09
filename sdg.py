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
    print("[DEBUG] Starting sqlmap tamper path search...")
    print(f"[DEBUG] SQLMAP_HOME env var: {os.environ.get('SQLMAP_HOME', 'Not set')}")
    print(f"[DEBUG] PATH env var: {os.environ.get('PATH', 'Not set')}")
    
    # 1. Check for an Environment Variable (The cleanest way for Git)
    # You can set this in your .bashrc: export SQLMAP_HOME="$HOME/Tools/sqlmap"
    env_path = os.environ.get("SQLMAP_HOME")
    if env_path:
        tamper_path = os.path.join(env_path, "tamper")
        print(f"[DEBUG] Checking SQLMAP_HOME path: {tamper_path}")
        if os.path.exists(tamper_path):
            print(f"[DEBUG] ✓ Found tamper directory at: {tamper_path}")
            return tamper_path
        else:
            print(f"[DEBUG] ✗ Tamper directory not found at SQLMAP_HOME location")

    # 2. Check system PATH (e.g., /usr/bin/sqlmap)
    print("[DEBUG] Checking system PATH for sqlmap binary...")
    sqlmap_bin = subprocess.getoutput("which sqlmap")
    print(f"[DEBUG] which sqlmap returned: {sqlmap_bin}")
    if sqlmap_bin and os.path.exists(sqlmap_bin):
        potential = os.path.dirname(os.path.realpath(sqlmap_bin))
        tp = os.path.join(potential, "tamper")
        print(f"[DEBUG] Checking for tamper at: {tp}")
        if os.path.exists(tp):
            print(f"[DEBUG] ✓ Found tamper directory at: {tp}")
            return tp
        else:
            print(f"[DEBUG] ✗ Tamper directory not found at binary location")

    # 3. Clever Regex Search (Generic & Non-Hardcoded)
    home = os.path.expanduser("~")
    print(f"[DEBUG] Searching common directories under home: {home}")
    sqlmap_regex = re.compile(r'^sqlmap', re.IGNORECASE)
    
    # We look in generic high-level folders
    for folder in ["Tools", "tools", "opt", "src"]:
        root = os.path.join(home, folder)
        print(f"[DEBUG] Checking directory: {root}")
        if not os.path.exists(root):
            print(f"[DEBUG] ✗ Directory does not exist")
            continue
        
        try:
            for entry in os.scandir(root):
                if entry.is_dir() and sqlmap_regex.match(entry.name):
                    tp = os.path.join(entry.path, "tamper")
                    print(f"[DEBUG] Found potential sqlmap dir: {entry.path}, checking for tamper...")
                    if os.path.exists(tp):
                        print(f"[DEBUG] ✓ Found tamper directory at: {tp}")
                        return tp
        except PermissionError:
            print(f"[DEBUG] ✗ Permission denied accessing {root}")
            continue
    
    print("[DEBUG] ✗ No sqlmap tamper directory found in any searched location")
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
    print(f"[*] Found sqlmap tamper directory at: {TAMPER_DIR}")
    print("[*] Loading sqlmap tamper scripts...")
    # These match the ones in your high-risk sqlmap command
    scripts = ["space2comment", "randomcase", "charencode", "apostrophemask"]
    for s in scripts:
        mod = load_sqlmap_tamper(TAMPER_DIR, s)
        if mod:
            loaded_tampers[s] = mod
            print(f"[DEBUG] ✓ Loaded tamper script: {s}")
        else:
            print(f"[DEBUG] ✗ Failed to load tamper script: {s}")
    print(f"[*] Successfully loaded {len(loaded_tampers)} tamper script(s)")
else:
    print("[!] No local sqlmap found. Using default patterns for Git compatibility.")
    print("[!] To use real sqlmap tampers, install sqlmap and set SQLMAP_HOME environment variable.")

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
