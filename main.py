import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.embeddings import VoyageEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document  
from pinecone import Pinecone


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

INDEX_NAME = "rag-index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

app = FastAPI(title="RAG Chatbot API")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)



class QueryRequest(BaseModel):
    query: str



def retrieve_docs(query):
    query_vector = embedder.embed_query(query)

    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    docs = []
    for match in results["matches"]:
        text = match["metadata"]["text"]
        docs.append(Document(page_content=text))

    return docs



def rag_chat(query):
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
