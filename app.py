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
    if uploaded:
        save_path = f"data/book_collection/{uploaded.name}"
        with st.spinner("Saving and ingesting your book..."):
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            vectordb = ingest_pdf_to_chroma(save_path, collection_name=uploaded.name)
        if vectordb:
            st.success("Book ingested and converted to vector store!")
            qa = build_chain(vectordb, k=4)
        else:
            qa = None
    else:
        qa = None

st.markdown("---")

if qa:
    st.subheader("Ask a question from your book")
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input("Type your question here:", key="question_input")
    with col2:
        ask_clicked = st.button("Ask", key="ask_button", use_container_width=True)

    if ask_clicked and question:
        with st.spinner("Searching for the answer..."):
            start = time.process_time()
            try:
                result = qa.invoke({'input': question})
                answer_text = result.get("answer")
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
                st.markdown("#### Sources")
                for i, doc in enumerate(sources):
                    meta = doc.metadata
                    with st.expander(
                        f"Source {i+1}: page {meta.get('page','?')} — {meta.get('chapter','') or meta.get('source','') }"
                    ):
                        st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                        st.code(meta)
            else:
                st.info("No sources found for this answer.")
else:
    st.info("Please upload a PDF book from the sidebar to get started.")
