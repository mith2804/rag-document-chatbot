import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Load environment variables
load_dotenv()

url = os.getenv("QDRANT_URL")
api_key = os.getenv("QDRANT_API_KEY")

# Connect to Qdrant


client = QdrantClient(
    url=url,
    api_key=api_key,
    timeout=60,
    check_compatibility=False
)

print("Connected to Qdrant!")

# Read PDF
reader = PdfReader("rag_llm_mcp.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks)

print("Embeddings created!")

# Collection name
collection_name = "rag_documents"

# Create collection
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

print("Collection ready!")

# Create points
points = []

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    points.append(
        PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload={
                "text": chunk,
                "source": "rag_llm_mcp.pdf"
            }
        )
    )

# Upload to Qdrant
client.upsert(
    collection_name=collection_name,
    points=points
)

print("✅ All embeddings stored in Qdrant!")

# Check collection
print(client.get_collections())