import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from pinecone import Pinecone
from huggingface_hub import InferenceClient


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

INDEX_NAME = "rag-index"
HF_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"


app = FastAPI(title="RAG Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


hf_client = InferenceClient(token=HF_API_KEY)

def embed_text(text: str):
    embedding = hf_client.feature_extraction(text, model=HF_MODEL)
    
    
    if isinstance(embedding[0], list):
        embedding = embedding[0]

    
    embedding = [float(x) for x in embedding]
    
    return embedding


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


class QueryRequest(BaseModel):
    query: str


def retrieve_docs(query: str):
    query_vector = embed_text(query)


    results = index.query(
        vector=query_vector,
        top_k=15,
        include_metadata=True
    )


    docs = []
    for match in results["matches"]:
        text = match["metadata"]["text"]
        docs.append(Document(page_content=text))

    return docs


def rag_chat(query: str):
    
    docs = retrieve_docs(query)
    context = "\n\n".join([d.page_content for d in docs])

    
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        prompt_template = f.read()

    
    prompt = prompt_template.format(context=context, query=query)

    
    response = llm.invoke(prompt)
    return response.content


@app.post("/chat")
async def chat(request: QueryRequest):
    answer = rag_chat(request.query)
    return {"response": answer}

@app.get("/")
async def home():
    return {"message": "RAG chatbot running successfully!"}
