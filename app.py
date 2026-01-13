import streamlit as st
from llm_initializer import initialize_rag_chain_with_qwen2, get_clean_answer
from faiss_generator import JSON_FILE_PATH, FAISS_INDEX_PATH



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

if __name__ == "__main__":
    main()