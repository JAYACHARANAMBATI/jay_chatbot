import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache
from operator import itemgetter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

app = FastAPI(title="PDF RAG Chatbot API")

# --------------------------------------------------------
# VECTORSTORE
# --------------------------------------------------------
@lru_cache(maxsize=1)
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name="pdf_rag_db",
        embedding_function=embeddings,
        persist_directory="chroma_store",
    )


# --------------------------------------------------------
# LLM
# --------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )


# --------------------------------------------------------
# SYSTEM PROMPT LOADER
# --------------------------------------------------------
def load_prompt():
    file_path = "system_prompt.txt"

    if not os.path.exists(file_path):
        return "You are a helpful assistant.\n\n{context}"

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if "{context}" not in text:
        text += "\n\n{context}"
    return text


# --------------------------------------------------------
# RAG CHAIN
# --------------------------------------------------------
@lru_cache(maxsize=1)
def get_chain():
    vectorstore = get_vectorstore()
    llm = get_llm()

    system_prompt = load_prompt()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # FIXED: Only send the question string into retriever
    rag_chain = (
        RunnableParallel(
            question=itemgetter("question"),
            chat_history=itemgetter("chat_history"),
            context=itemgetter("question") | retriever
        )
        | prompt
        | llm
    )

    return rag_chain


# --------------------------------------------------------
# FASTAPI MODELS
# --------------------------------------------------------
class Query(BaseModel):
    question: str


# Chat memory
chat_history = []


# --------------------------------------------------------
# ASK ENDPOINT
# --------------------------------------------------------
@app.post("/ask")
async def ask(query: Query):
    global chat_history

    chain = get_chain()

    response = chain.invoke({
        "question": query.question,
        "chat_history": chat_history
    })

    answer = response.content

    # Update chat history
    chat_history.append(("human", query.question))
    chat_history.append(("assistant", answer))

    return {"response": answer}


# --------------------------------------------------------
# RESET CHAT
# --------------------------------------------------------
@app.post("/reset")
async def reset_chat():
    global chat_history
    chat_history = []
    return {"status": "Chat memory reset"}


# --------------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API Running Successfully!"}
