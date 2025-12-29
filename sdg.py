import pandas as pd
import random

def generate_synthetic_data(count=1000):
    sql_errors = [
        "SQLSTATE[42000]: Syntax error or access violation: 1064 You have an error in your SQL syntax",
        "Unclosed quotation mark after the character string",
        "The used SELECT statements have a different number of columns",
        "PostgreSQL Error: column \"STR\" does not exist",
        "ORA-00933: SQL command not properly ended"
    ]
    
    safe_logs = [
        "HTTP/1.1 200 OK - Request processed successfully",
        "User profile updated for ID NUM",
        "Search results for STR found NUM items",
        "Welcome back, STR!",
        "Page Not Found - The requested URL was not found on this server."
    ]

    data = []
    for _ in range(count):
        # Generate Malicious (Label 1)
        error = random.choice(sql_errors)
        data.append({"text": error, "label": 1})
        
        # Generate Safe (Label 0)
        safe = random.choice(safe_logs)
        data.append({"text": safe, "label": 0})

    df = pd.DataFrame(data)
    df.to_csv("training_data.csv", index=False)
    print(f"[+] Created training_data.csv with {count*2} samples.")

if __name__ == "__main__":
    generate_synthetic_data()
