import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="PDF RAG Chatbot API")

@lru_cache(maxsize=1)
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name="pdf_rag_db",
        embedding_function=embeddings,
        persist_directory="chroma_store",
    )

@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def load_prompt():
    file_path = "system_prompt.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if "{context}" not in text:
        text += "\n\n{context}"
    return text

@lru_cache(maxsize=1)
def get_chain():
    vectorstore = get_vectorstore()
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    system_prompt = load_prompt()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
    )

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    chain = get_chain()
    answer = chain.invoke({"question": query.question})
    return {"response": answer["answer"]}

@app.post("/reset")
async def reset_chat():
    chain = get_chain()
    chain.memory.clear()
    get_chain.cache_clear()  
    return {"status": "Chat memory reset"}

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API Running"}




