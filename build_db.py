import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PDF_PATH = "/Users/STAFF1/Desktop/charan_agent/AMBATIJAYACHARAN_RESUME.pdf"
CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "pdf_rag_db"

def build_vector_db():
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found: {PDF_PATH}")
        return

    print("📄 Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    print("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"🔍 Total chunks created: {len(chunks)}")

    print("🧠 Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("🗂 Creating Chroma vectorstore...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    print("➕ Adding chunks to vectorstore...")
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    print("✅ Vector DB successfully created and saved!")

if __name__ == "__main__":
    build_vector_db()
