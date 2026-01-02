import requests
import json
import sys

# Ensure this matches your redisql.py port (usually 5000)
API_URL = "http://localhost:5000/process"

def postprocess(page, headers=None, code=None):
    if page:
        try:
            # Send the page content to your ML Pipeline API
            payload = {"logs": [page]}
            response = requests.post(API_URL, json=payload, timeout=1)
            
            # If the API returns an error, print it to the sqlmap console
            if response.status_code != 200:
                sys.stderr.write(f"\n[!] ML Bridge API Error: Status {response.status_code}\n")
                
        except Exception as e:
            # If the bridge can't reach the API at all, this will show you why
            sys.stderr.write(f"\n[!] ML Bridge Connection Failed: {str(e)}\n")

    return page, headers, code
