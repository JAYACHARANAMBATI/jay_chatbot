import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load Pinecone credentials
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # for all-MiniLM-L6-v2 (outputs 384-dim vectors)
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# Load your documents folder
DATA_DIR = "docs"

docs = []
for file in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file)
    if file.endswith(".txt") or file.endswith(".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            docs.append(f.read())
    elif file.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            docs.append(text)

# Split docs into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_text("\n".join(docs))
print("Total Chunks:", len(chunks))

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectors = []

for i, chunk in enumerate(chunks):
    vec = embeddings.embed_query(chunk)
    vectors.append({
        "id": f"chunk-{i}",
        "values": vec,
        "metadata": {"text": chunk}
    })

index.upsert(vectors)
print("Pinecone DB build completed!")
