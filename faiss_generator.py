from langchain_core.documents import Document
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import os
import json
import torch


EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
JSON_FILE_PATH = "klnce_chunks.json"
FAISS_INDEX_PATH = "faiss_index_bge_qwen2"

CACHE_DIR = "D:\CampusAssistant\CampusAssistant\cache" 

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = CACHE_DIR


def load_documents_from_json(file_path: str) -> List[Document]:
    """Load documents from JSON or return mock data if file not found."""
    if not os.path.exists(file_path):
        print(f"Error: JSON file not found at {file_path}. Using mock data.")
        return [
            Document(
                page_content="KLN College of Engineering offers a Bachelor of Engineering in Electrical and Electronics Engineering (EEE) started in 1994 with an intake of 40 students, increased to 60 in 1996 and 120 in 2011.",
                metadata={"source": "Electrical and Electronics Engineering(EEE)", "department": "Electrical and Electronics Engineering(EEE)"}
            ),
            Document(
                page_content="The Master of Business Administration department has consistently produced rank holders, with 3 rank holders reported in the 2021-2023 batch.",
                metadata={"source": "Master of Business Administration(MBA)", "department": "Master of Business Administration(MBA)"}
            ),
        ]

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for chunk in data:
        metadata = {
            "source": f"{chunk.get('department', 'Unknown Department')} (Chunk {chunk.get('id', 'N/A')})",
            "department": chunk.get('department', 'Unknown Department'),
            "keywords": chunk.get('keywords', []),
            "aliases": chunk.get('aliases', [])
        }
        documents.append(Document(page_content=chunk['text'], metadata=metadata))

    print(f"✅ Loaded {len(documents)} document chunks from JSON file.")
    return documents


def setup_hybrid_retriever(documents: List[Document], faiss_path: str) -> EnsembleRetriever:
    """Initializes embeddings, creates/loads FAISS index, sets up BM25, and returns the EnsembleRetriever."""
    print(f"Initializing BGE embedding model: {EMBEDDING_MODEL_NAME}...")

    embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}  # ✅ prevent meta tensor issue
    )

    # 1. Dense Retriever (FAISS) Setup
    rebuild_index = True
    # if os.path.exists(faiss_path):
    #     print(f"Attempting to load existing FAISS index from {faiss_path}...")
    #     try:
    #         vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    #         if vectorstore.index.d == len(embeddings.embed_query("test query")):
    #             rebuild_index = False
    #         else:
    #             print("⚠ Embedding dimension mismatch. Rebuilding FAISS index...")
    #     except:
    #         print("⚠ Failed to load FAISS index. Rebuilding...")

    # if rebuild_index:
    #     print(f"Creating new FAISS index with {len(documents)} documents and saving to {faiss_path}...")
    #     vectorstore = FAISS.from_documents(documents, embeddings)
    #     vectorstore.save_local(faiss_path)
    vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Lexical Retriever (BM25) Setup
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    # 3. Ensemble Retriever (Hybrid Search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.2, 0.8]  # Prioritize semantic search (BGE) results
    )

    print("✅ Hybrid (Ensemble) Retriever configured.")
    return ensemble_retriever

