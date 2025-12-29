import requests
import json

# Your API endpoint
API_URL = "http://localhost:5000/process"

def postprocess(page, headers=None, code=None):
    """
    This function is called by sqlmap for EVERY response it receives.
    """
    if page:
        try:
            # Send the page content to your ML Pipeline
            payload = {"logs": [page]}
            response = requests.post(API_URL, json=payload, timeout=1)
            
            if response.status_code == 200:
                job_ids = response.json().get("job_ids", [])
                # We don't wait for the result here to keep sqlmap fast.
                # The workers in the background will handle the analysis.
                pass
        except Exception:
            # Silent fail to ensure sqlmap scan isn't interrupted
            pass

    return page, headers, code
