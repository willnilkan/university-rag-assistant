import streamlit as st
from ingest import ingest_pdf, list_documents
from rag_pipeline import ask_university

st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="wide")
st.title("🎓 University AI Assistant")
st.caption("Document-based University Knowledge System")

with st.sidebar:
    st.header("📚 Knowledge Base")
    uploaded_files = st.file_uploader("Upload university PDF documents", type=["pdf"], accept_multiple_files=True)
    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        message = ingest_pdf(uploaded_file)
                    st.success(message)
                except Exception as e:
                    st.error(f"Failed: {uploaded_file.name} - {e}")
                progress.progress((i + 1) / len(uploaded_files))
        else:
            st.warning("Please select PDF files first.")
    st.divider()
    st.subheader("Stored Documents")
    documents = list_documents()
    if documents:
        st.write(f"PDF files: {len(documents)}")
        for doc in documents:
            st.write(f"📄 {doc.name}")
    else:
        st.info("No PDFs stored yet.")

st.header("💬 Ask the University Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Retrieved Sources"):
                for i, source in enumerate(message["sources"], start=1):
                    doc_name = source.get("document", "Unknown")
                    page_num = source.get("page", 1)
                    st.write(f"[Source {i}] {doc_name} — Page {page_num}")

question = st.chat_input("Ask a question about the university...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching university documents..."):
            try:
                result = ask_university(question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("View Retrieved Sources"):
                        for i, source in enumerate(result["sources"], start=1):
                            doc_name = source.get("document", "Unknown")
                            page_num = source.get("page", 1)
                            st.write(f"[Source {i}] {doc_name} — Page {page_num}")
                st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
            except Exception as e:
                st.error(f"Error: {e}")
