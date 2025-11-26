import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")


pc = Pinecone(api_key=PINECONE_API_KEY)


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, 
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)


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


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_text("\n".join(docs))
print("Total Chunks:", len(chunks))


HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

hf_client = InferenceClient(token=HF_API_KEY)

def embed_text(text: str):
    """
    Get embedding from HuggingFace Inference API
    Returns 1D Python float list for Pinecone
    """
    embedding = hf_client.feature_extraction(text, model=HF_MODEL)

   
    if isinstance(embedding[0], list):
        embedding = embedding[0]

  
    embedding = [float(x) for x in embedding]
    return embedding


vectors = []

for i, chunk in enumerate(chunks):
    vec = embed_text(chunk)  
    vectors.append({
        "id": f"chunk-{i}",
        "values": vec,
        "metadata": {"text": chunk}
    })

# Upload to Pinecone
index.upsert(vectors)
print("Pinecone DB build completed successfully!")
