import requests
import json

# Base URL and API Key
BASE_URL = "http://localhost:5100"
API_KEY = "Polynomialai"

# Headers for admin endpoints
admin_headers = {
    "X-API-Key": API_KEY
}

def add_document(file_path):
    files = {
        'documents': open(file_path, 'rb')
    }
    response = requests.post(
        f"{BASE_URL}/admin/upload",
        headers=admin_headers,
        files=files
    )
    return response.json()

print(add_document("acme_corp_data.pdf"))