from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Read PDF
pdf_path = "rag_llm_mcp.pdf"
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Convert chunks into vectors
embeddings = model.encode(chunks)

print("Embedding created!")
print("Number of vectors:", len(embeddings))
print("Vector dimension:", len(embeddings[0]))

# 5. Show first vector
print("\nFirst vector:")
print(embeddings[0])