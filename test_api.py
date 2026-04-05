import requests
import json

url = "http://127.0.0.1:8000/prediction"

with open("tests/mes_transactions.json") as f:
    data = json.load(f)

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())
