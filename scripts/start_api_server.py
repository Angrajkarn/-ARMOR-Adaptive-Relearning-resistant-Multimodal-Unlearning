"""
scripts/start_api_server.py
===========================
Boots the ARMOR Online Unlearning API Server using Uvicorn.
"""
import sys
import os
import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    print("Starting ARMOR API Server on port 8080...")
    uvicorn.run("armor.api.server:app", host="0.0.0.0", port=8080, reload=False)
