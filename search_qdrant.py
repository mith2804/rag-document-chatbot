import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

# -----------------------------
# 1. Qdrant connection
# -----------------------------

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# -----------------------------
# 2. Embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

collection_name = "rag_documents"

# -----------------------------
# 3. Load Qwen
# -----------------------------

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# -----------------------------
# 4. User question
# -----------------------------

query = input("Enter your question: ")

# -----------------------------
# 5. Convert question to vector
# -----------------------------

query_vector = embedding_model.encode(query).tolist()

# -----------------------------
# 6. Search Qdrant
# -----------------------------

results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=5
)

# -----------------------------
# 7. Get retrieved text
# -----------------------------

context = ""
print("\n===== RETRIEVED CONTEXT =====\n")
print(context)
print("\n=============================\n")

for result in results.points:
    context += result.payload["text"] + "\n\n"

# -----------------------------
# 8. Create RAG prompt
# -----------------------------

prompt = f"""
You are a RAG assistant.

Answer the question using ONLY the information provided in the context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not present in the context, say:
  "I don't have enough information in the provided document."
- Give a clear and concise answer.
- Do not explain the RAG process unless the question asks about it.

Context:
{context}

Question:
{query}

Answer:
"""

# -----------------------------
# 9. Generate answer using Qwen
# -----------------------------

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = llm.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

answer = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

# -----------------------------
# 10. Display answer
# -----------------------------

print("\n===== RAG ANSWER =====\n")
print(answer)