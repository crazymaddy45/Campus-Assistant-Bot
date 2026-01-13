import os
import json
from typing import List, Any
import re
import torch
import streamlit as st

CACHE_DIR = "D:\StudySphere\cache" 

# Set the HF_HOME environment variable
os.environ["HF_HOME"] = CACHE_DIR

# LangChain and RAG components
# --- Core Document and Prompt ---
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# --- Embeddings and VectorStore ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Retrievers ---
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  # ✅ CORRECTED Import

# --- LLMs ---
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms.fake import FakeListLLM

# --- Chains (Updated for modern RAG) ---
from langchain_classic.chains import create_retrieval_chain  # ✅ NEW
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # ✅ NEW

# --- Text Splitter ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Transformers ---
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


data_folder = "klnce_data"
output_file = "klnce_chunks.json"

# ACRONYM_MAP = {
#     # --- Core Departments ---
#     "AI & DS": "Artificial Intelligence and Data Science",
#     "AIDS": "Artificial Intelligence and Data Science",
#     "AI-DS": "Artificial Intelligence and Data Science",
#     "CSE": "Computer Science and Engineering",
#     "CS": "Computer Science",
#     "IT": "Information Technology",
#     "ECE": "Electronics and Communication Engineering",
#     "EEE": "Electrical and Electronics Engineering",
#     "EIE": "Electronics and Instrumentation Engineering",
#     "MECH": "Mechanical Engineering",
#     "CIVIL": "Civil Engineering",
#     "BIO": "Biomedical Engineering",
#     "BME": "Biomedical Engineering",
#     "CHEM": "Chemical Engineering",
#     "CHE": "Chemical Engineering",
#     "AUTO": "Automobile Engineering",
#     "MBA": "Master of Business Administration",
#     "MCA": "Master of Computer Applications",
#     "IOT": "Internet of Things",
#     "CYBER SECURITY": "Cyber Security",
#     "CSBS": "Computer Science and Business Systems",
#     "AIML": "Artificial Intelligence and Machine Learning",
#     "DS": "Data Science",
#     "ML": "Machine Learning",
#     "AI": "Artificial Intelligence",

#     # --- Academic and Administrative Acronyms ---
#     "HOD": "Head of the Department",
#     "COE": "Controller of Examinations",
#     "IQAC": "Internal Quality Assurance Cell",
#     "NBA": "National Board of Accreditation",
#     "NAAC": "National Assessment and Accreditation Council",
#     "AICTE": "All India Council for Technical Education",
#     "UGC": "University Grants Commission",
#     "TNEA": "Tamil Nadu Engineering Admissions",
#     "AU": "Anna University",
#     "R&D": "Research and Development",
#     "PG": "Postgraduate",
#     "UG": "Undergraduate",
#     "B.E": "Bachelor of Engineering",
#     "B.TECH": "Bachelor of Technology",
#     "M.E": "Master of Engineering",
#     "M.TECH": "Master of Technology",
#     "PhD": "Doctor of Philosophy",
#     "RAC": "Research Advisory Committee",

#     # --- Campus and Institutional ---
#     "CDC": "Career Development Cell",
#     "PLACEMENT CELL": "Placement and Training Cell",
#     "PTA": "Parent Teachers Association",
#     "NSS": "National Service Scheme",
#     "NCC": "National Cadet Corps",
#     "YRC": "Youth Red Cross",
#     "WDC": "Women Development Cell",
#     "EDC": "Entrepreneurship Development Cell",
#     "IIC": "Institution Innovation Council",
#     "ALUMNI": "Alumni Association",
#     "CSI": "Computer Society of India",
#     "ISTE": "Indian Society for Technical Education",
#     "IEEE": "Institute of Electrical and Electronics Engineers",
#     "IE": "Institution of Engineers",
#     "IETE": "Institution of Electronics and Telecommunication Engineers",
#     "SAE": "Society of Automotive Engineers",
#     "ASME": "American Society of Mechanical Engineers",
#     "ROBOTICS CLUB": "Robotics and Automation Club",
#     "AI CLUB": "Artificial Intelligence Club",
#     "CSE CLUB": "Computer Science Club",
#     "INNOVATION CELL": "Innovation and Startup Cell",
# }



# def normalize_text(text):
#     # Replace acronyms and clean unwanted characters
#     for key, full_form in ACRONYM_MAP.items():
#         text = re.sub(rf"\b{re.escape(key)}\b", full_form, text, flags=re.IGNORECASE)
#     text = re.sub(r"\s+", " ", text)  # normalize spaces
#     text = re.sub(r"[^a-zA-Z0-9.,:;!?()\-\n ]", "", text)  # remove junk symbols
#     return text.strip()

# def extract_keywords(text):
#     # Very lightweight keyword extraction
#     words = re.findall(r'\b[A-Za-z]{4,}\b', text)
#     common = ["department", "engineering", "college", "faculty", "student", "lab", "course"]
#     keywords = [w.lower() for w in words if w.lower() not in common]
#     return list(set(keywords[:10]))

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", ".", " ", ""]
# )


# docs = []

# for file in os.listdir(data_folder):
#     if file.endswith(".txt"):
#         dept_name = file.replace(".txt", "")
#         with open(os.path.join(data_folder, file), "r", encoding="utf-8") as f:
#             text = f.read()

#         text = normalize_text(text)

#         # Merge small related headers like Vision, Mission, About into one chunk
#         sections = re.split(r"(?<=:)\s*(?=[A-Z][a-z])", text)
#         merged_text = " ".join(sections)

#         chunks = text_splitter.split_text(merged_text)
#         for i, chunk in enumerate(chunks):
#             keywords = extract_keywords(chunk)
#             docs.append({
#                 "id": f"{dept_name}_{i}",
#                 "department": dept_name,
#                 "text": chunk,
#                 "keywords": keywords,
#                 "aliases": [k for k, v in ACRONYM_MAP.items() if v.lower() in chunk.lower() or k.lower() in chunk.lower()]
#             })

# # Save enhanced chunks
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(docs, f, indent=2, ensure_ascii=False)

# print(f"✅ Created {len(docs)} enhanced chunks from {len(os.listdir(data_folder))} department files.")





# -------------------------------
# 1. Setup: Installation and Imports
# -------------------------------
# !pip install langchain langchain-community faiss-cpu sentence-transformers transformers pypdf lxml InstructorEmbedding rank_bm25 accelerate bitsandbytes


# --- Configuration for BGE + Qwen2 Combo ---
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
LLM_MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
JSON_FILE_PATH = "klnce_chunks.json"
FAISS_INDEX_PATH = "faiss_index_bge_qwen2"
# -------------------------------------------

# --- Helper Functions (No changes needed) ---

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

# -------------------------------
# 🔍 Retriever Setup (No changes needed)
# -------------------------------
def setup_hybrid_retriever(documents: List[Document], faiss_path: str) -> EnsembleRetriever:
    """Initializes embeddings, creates/loads FAISS index, sets up BM25, and returns the EnsembleRetriever."""
    print(f"Initializing BGE embedding model: {EMBEDDING_MODEL_NAME}...")

    embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}  # ✅ prevent meta tensor issue
    )

    # 1. Dense Retriever (FAISS) Setup
    # rebuild_index = True
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

# -------------------------------
# 🔑 LLM Initialization (No changes needed)
# -------------------------------
def initialize_qwen2_llm():
    """
    Initializes Qwen2-7B-Instruct using HuggingFacePipeline.
    Optimized for Colab T4 GPU via 8-bit loading.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        load_in_8bit = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 12 * 1024**3

        if load_in_8bit:
            print(f"Loading {LLM_MODEL_NAME} with 8-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch_dtype,
                device_map="auto",
                load_in_8bit=load_in_8bit,
                trust_remote_code=True
            )
        else:
            print(f"Loading {LLM_MODEL_NAME} with standard settings (bfloat16)...")
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                offload_folder="offload_dir"
            )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"✅ LLM initialized: {LLM_MODEL_NAME}")
        return llm

    except Exception as e:
        print(f"❌ Critical Error initializing LLM Pipeline: {e}")
        return FakeListLLM(responses=["[MOCK] The Qwen2 LLM failed to load. This response is from a placeholder model. Please check your GPU/memory configuration."])


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
    llm = initialize_qwen2_llm()

    # 4. Define Prompt (Your template is already compatible)
    PROMPT_TEMPLATE = """You are an assistant answering questions from retrieved documents.
    Use ONLY the information present in the following documents to answer the question.
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
def get_clean_answer(qa_chain: Any, query: str) -> dict: # Type hint changed
    """
    Invokes the modern RAG chain, extracts the clean answer from the LLM output,
    and removes repeated prompt text or unnecessary content.
    """
    try:
        # ✅ Invoke with 'input' key
        output = output = qa_chain.invoke({"input": query})

        # ✅ Extract 'answer' key
        clean_answer = output.get("answer", output.get("result", "")).strip()

        # --- Remove the repeating prompt text if the model echoes it ---
        prompt_start_marker = "You are an assistant answering questions from retrieved documents."
        if clean_answer.startswith(prompt_start_marker):
            answer_start = clean_answer.find("Answer:")
            if answer_start != -1:
                clean_answer = clean_answer[answer_start + len("Answer:"):]
            else:
                clean_answer = clean_answer.replace(prompt_start_marker, "")

        # --- Final cleanup ---
        clean_answer = re.sub(r"\s+", " ", clean_answer).strip()

        # ✅ Extract sources from 'context' key
        source_docs = output.get("context", [])
        sources = [
            doc.metadata.get("source", "Unknown Source")
            for doc in source_docs[:2]
        ]

        return {
            "query": query,
            "answer": clean_answer,
            "source": sources
        }

    except Exception as e:
        return {
            "query": query,
            "answer": f"[Error] Failed to process answer: {e}",
            "source": []
        }
    

# -------------------------------
# Example Usage (No changes needed)
# -------------------------------
# if _name_ == "_main_":
#     print("\n🚀 Starting BGE + Qwen2 RAG setup...\n")
#     qa_chain = initialize_rag_chain_with_qwen2(JSON_FILE_PATH, FAISS_INDEX_PATH)

#     queries = [
#         "How many seats are available in EEE?",
#         "How many rank holders did the MBA department produce in the 2021-2023 batch?",
#         "Who is the Head of the Department for Electrical and Electronics Engineering?"
#     ]

#     for q in queries:
#         print(f"\n❓ Query: {q}")
#         result = get_clean_answer(qa_chain, q)
#         print("💡 Answer:", result["answer"])
#         print("📄 Source:", result["source"])
@st.cache_resource(show_spinner=False)
def load_qa_chain():
    return initialize_rag_chain_with_qwen2(JSON_FILE_PATH, FAISS_INDEX_PATH)

def main():
    st.set_page_config(page_title="StudySphere | KLNCE Campus Assistant", page_icon="🎓", layout="centered")

    st.title("🎓 StudySphere – KLNCE Campus Assistant")
    st.caption("Ask any question about KLN College of Engineering (departments, staff, or events).")

    qa_chain = load_qa_chain()
    query = st.text_input("💬 Ask your question here:")

    if st.button("Ask") or query:
        if query.strip() == "":
            st.warning("⚠ Please enter a valid question.")
            return

        progress = st.progress(0)
        status = st.empty()

        with st.spinner("🚀 Thinking..."):
            progress.progress(30)
            status.text("🔍 Retrieving relevant info...")
            answer_data = get_clean_answer(qa_chain, query)
            progress.progress(100)
            status.empty()

        st.success("✅ Answer:")
        st.markdown(f"{answer_data['answer']}")

        if answer_data.get("source"):
            with st.expander("📄 View Sources"):
                for src in answer_data["source"]:
                    st.write(f"- {src}")

    st.markdown("---")
    st.caption("Built with ❤ using LangChain + Qwen2 + Streamlit")

if _name_ == "_main_":
    main()