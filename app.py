import streamlit as st
import time
from ingest import ingest_pdf_to_chroma
from chain import build_chain

st.set_page_config(page_title="RAG Tutor", page_icon="📚", layout="wide")
st.markdown(
    """
    <style>
    .main { background-color: #f5f6fa; }
    .stButton>button { background-color: #4F8BF9; color: white; }
    .st-expanderHeader { font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 RAG Tutor")
st.markdown(
    "Ask questions and get answers only from your uploaded book. Powered by Retrieval-Augmented Generation (RAG)."
)

with st.sidebar:
    st.header("Upload your book")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    vectordb, docs = None, None
    if uploaded:
        save_dir = "data/book_collection"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded.name)
        if "qa" not in st.session_state or st.session_state.get("last_uploaded") != uploaded.name:
            with st.spinner("Saving and ingesting your book..."):
                with open(save_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                # ingest now returns vectordb, docs
                vectordb, docs = ingest_pdf_to_chroma(save_path, collection_name=uploaded.name)
        else:
            # If already ingested, try to get from session_state if available
            vectordb = st.session_state.get("vectordb")
            docs = st.session_state.get("docs")
        if vectordb:
            st.success("Book ingested and converted to vector store!")
            st.session_state.qa = build_chain(vectordb, docs, k=6)
            st.session_state.last_uploaded = uploaded.name
            st.session_state.vectordb = vectordb
            st.session_state.docs = docs
        qa = st.session_state.qa    
    else:
        qa = None

st.markdown("---")
print("Debugging QA:", qa)
if qa:
    st.subheader("Ask a question from your book")
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Type your question here:", key="question_input")
    with col2:
        ask_clicked = st.button("Ask", key="ask_button", use_container_width=True)
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] { margin-top: 1.5rem !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    if ask_clicked and question:
        with st.spinner("Searching for the answer..."):
            start = time.process_time()
            result = {}
            try:
                result = qa({"input": question})
                answer_text = result["answer"].content
                response_time = time.process_time() - start
                if answer_text:
                    st.success("**Answer:**")
                    st.markdown(f"> {answer_text}")
                    st.caption(f"⏱️ Response time: {response_time:.2f} seconds")
                else:
                    st.warning("No answer found in your book.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

            # show sources
            sources = result.get("source_documents", [])
            if sources:
                with st.expander("#### Sources", expanded=False):
                    for i, doc in enumerate(sources):
                        meta = doc.metadata
                        citation = f"[Chapter: {meta.get('chapter','?')} | Section: {meta.get('section','?')} | Page: {meta.get('page_number','?')}]"
                        
                        st.markdown(f"**Source {i+1}:** {citation}")
                        st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                        # Instead of dumping full dict, show structured metadata
                        st.markdown(f"**Book:** {meta.get('book_title','?')}")
                        st.markdown(f"**Chapter:** {meta.get('chapter','—')}")
                        st.markdown(f"**Section:** {meta.get('section','—')}")
                        st.markdown(f"**Page:** {meta.get('page_number','?')}")
                        st.markdown("---")
            else:
                st.info("No sources found for this answer.")
else:
    st.info("Please upload a PDF book from the sidebar to get started.")
