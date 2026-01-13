from faiss_generator import setup_hybrid_retriever, load_documents_from_json    
from langchain_core.prompts import PromptTemplate
from typing import List, Any
import re
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.fake import FakeListLLM
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI


EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL_NAME = "qwen/qwen3-32b"
JSON_FILE_PATH = "klnce_chunks.json"
FAISS_INDEX_PATH = "faiss_index_bge_qwen2"


def initialize_qwen3_groq_llm():
    """
    Initializes Qwen3-32B on Groq using OpenAI-compatible API.
    No local GPU load — runs fully on Groq's hardware.
    """
    try:
        GROQ_API_KEY = "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"  # ⚠ add your key

        llm = ChatOpenAI(
            model=LLM_MODEL_NAME,   # Example name; check Groq’s exact model ID via their docs/dashboard
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=GROQ_API_KEY,
            temperature=0.3,
            max_tokens=512
        )

        print("✅ LLM initialized: Qwen3-32B (Groq-hosted)")
        return llm

    except Exception as e:
        print(f"❌ Failed to initialize Groq LLM: {e}")
        return FakeListLLM(responses=[
            "[MOCK] Groq LLM failed to load. Please check your API key or model ID."
        ])

# -------------------------------
# ⚙ RAG Chain Setup (✅ UPDATED)
# -------------------------------
def initialize_rag_chain_with_qwen2(json_path: str, faiss_path: str) -> Any: # Return type changed
    """
    Load documents, setup BGE hybrid retriever, and create RAG chain
    using the modern 'create_retrieval_chain'.
    """
    # 1. Load documents
    documents = load_documents_from_json(json_path)

    # 2. Setup hybrid retriever (BGE + BM25)
    retriever = setup_hybrid_retriever(documents, faiss_path)

    # 3. Initialize Qwen2 LLM
    llm = initialize_qwen3_groq_llm()

    # 4. Define Prompt (Your template is already compatible)
    PROMPT_TEMPLATE = """

    You are a concise, factual assistant.
    Do NOT include any reasoning traces or internal thoughts.
    Only provide the final answer to the question clearly and directly.
    Never include <think> or similar reasoning tags in your output.

    You are an assistant answering questions from retrieved documents.
    Use ONLY the information present in the following documents to answer the question.
    an
    Answer the question in the same language as the question itself (English means English, Tamil means Tamil).
    Also answer college-related questions accurately without hallucinating.

    Context:
    {context}

    Question: {input}

    Answer:"""

    PROMPT = PromptTemplate(
        input_variables=["context", "input"],  # ✅ Changed here
        template=PROMPT_TEMPLATE
    )

    # 5. Create RAG chain (✅ Modern LCEL Method)
    
    # 5a. Create a chain to stuff docs into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
    
    # 5b. Create the main retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("✅ BGE + Qwen2 (create_retrieval_chain) RAG chain is ready.")
    return rag_chain


# -------------------------------
# 🧹 Cleaned Output Helper Function (✅ UPDATED)
# -------------------------------
def get_clean_answer(qa_chain: Any, query: str) -> dict:
    """
    Extracts a clean answer from the RAG model output,
    efficiently removes <think>...</think> blocks, and collects sources.
    Optimized for Groq-hosted LLMs.
    """
    try:
        output = qa_chain.invoke({"input": query})
        raw_answer = output.get("answer", output.get("result", "")).strip()

        # ✅ Early trim: cut out heavy reasoning before regex
        if "<think>" in raw_answer.lower():
            start = raw_answer.lower().find("<think>")
            end = raw_answer.lower().find("</think>")
            if end != -1:
                raw_answer = raw_answer[:start] + raw_answer[end + 8:]  # remove entire reasoning block

        # ✅ Clean minimal regex for leftover tags
        clean_answer = re.sub(r"(?is)<think>.*?</think>", "", raw_answer)
        clean_answer = re.sub(r"\s+", " ", clean_answer).strip()

        # ✅ Avoid flooding terminal
        print(f"Clean Answer: {clean_answer[:500]}...")  # preview only first 500 chars

        # ✅ Extract sources safely
        source_docs = output.get("context", [])
        sources = [doc.metadata.get("source", "Unknown Source") for doc in source_docs[:2]]

        return {
            "query": query,
            "answer": clean_answer,
            "source": sources
        }

    except Exception as e:
        print(f"[Error in get_clean_answer] {e}")
        return {
            "query": query,
            "answer": f"[Error] Failed to process answer: {e}",
            "source": []
        }