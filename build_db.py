import os
from dotenv import load_dotenv  
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

pdf_path = "/Users/STAFF1/Desktop/charan_agent/AMBATIJAYACHARAN_RESUME.pdf"

loader = PyPDFLoader(pdf_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(docs)

print(f"Total chunks = {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="pdf_rag_db",
    embedding_function=embeddings,
    persist_directory="chroma_store"
)

vectorstore.add_documents(chunks)
vectorstore.persist()

print("Database build completed")
