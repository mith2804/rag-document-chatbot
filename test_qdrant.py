import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

print("URL loaded:", bool(url))
print("API key loaded:", bool(api_key))

client = QdrantClient(
    url=url,
    api_key=api_key,
    prefer_grpc=False,
    timeout=60
)

print("Connected to Qdrant!")
print(client.get_collections())