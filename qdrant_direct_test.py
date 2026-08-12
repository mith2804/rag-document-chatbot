import os
import requests
from dotenv import load_dotenv

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print("URL loaded:", bool(url))
print("API key loaded:", bool(api_key))

headers = {
    "api-key": api_key
}

response = requests.get(
    url + "/collections",
    headers=headers,
    timeout=30,
    proxies={"http": None, "https": None}
)

print("Status:", response.status_code)
print("Response:", response.text)