import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print("URL:", url)
print("API key loaded:", bool(api_key))

client = QdrantClient(
    url=url,
    api_key=api_key
)

print(client.get_collections())