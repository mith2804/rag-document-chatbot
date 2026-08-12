import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚"
)

st.title("📚 RAG Document Chatbot")

# Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection_name = "rag_documents"

# Embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()

# Qwen
@st.cache_resource
def load_llm():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    return tokenizer, model

tokenizer, llm = load_llm()

# Question
query = st.text_input("Ask your question:")

if query:

    with st.spinner("Searching the document..."):

        # Question → vector
        query_vector = embedding_model.encode(query).tolist()

        # Search Qdrant
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=5
        )

        # Get context
        context = ""

        for result in results.points:
            context += result.payload["text"] + "\n\n"

    with st.spinner("Generating answer..."):

        prompt = f"""
You are a RAG assistant.

Answer the question using ONLY the information provided in the context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer is not present in the context, say:
"I don't have enough information in the provided document."
- Give a clear and concise answer.

Context:
{context}

Question:
{query}

Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

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

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Context"):
        st.write(context)