import requests
import json
import random

# Generate random vector
dim = 128
vector = [random.random() for _ in range(dim)]

# Upsert
url = "http://localhost:3000"
upsert_payload = {
    "id": 1,
    "vector": vector,
    "metadata": {"source": "python_test"}
}
print("Upserting...")
r = requests.post(f"{url}/upsert", json=upsert_payload)
print(r.status_code, r.text)

# Search
search_payload = {
    "vector": vector,
    "k": 5,
    "filter": {}
}
print("Searching...")
r = requests.post(f"{url}/search", json=search_payload)
print(r.status_code, r.text)
