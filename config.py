import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model= "openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# ── Embeddings ────────────────────────────────────────────────────────────────
embeddings = CohereEmbeddings(
    model="embed-v4.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
)

# ── Vector Store (Chroma Cloud) ───────────────────────────────────────────────
import chromadb

chroma_client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)

vectorstore = Chroma(
    client=chroma_client,
    collection_name="rag_docs",
    embedding_function=embeddings,
)


retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)