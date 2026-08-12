from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Read PDF
pdf_path = "rag_llm_mcp.pdf"
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

# 2. Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 3. Split text into chunks
chunks = text_splitter.split_text(text)

# 4. Display results
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)