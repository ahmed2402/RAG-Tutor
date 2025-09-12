import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# -------------------- Prompt with Metadata Injection --------------------
prompt = ChatPromptTemplate.from_template(
    """
    You are a tutor limited to the given book excerpts.
    Answer ONLY from the book. If not enough info → say: ❌ Insufficient evidence.

    Always include citations in this format: [Chapter | Section | Page].

    Context excerpts (with metadata injected):
    {context}

    Question: {input}

    Answer:
    """
)

def format_docs_with_metadata(docs):
    """Prepend metadata tags like [Chapter: X | Section: Y | Page: Z] before content."""
    formatted = []
    for d in docs:
        meta = d.metadata
        tag = f"[Chapter: {meta.get('chapter','?')} | Section: {meta.get('section','?')} | Page: {meta.get('page_number','?')}]"
        formatted.append(f"{tag} {d.page_content}")
    return "\n\n".join(formatted)

# -------------------- Hybrid Retrieval with RRF --------------------
def build_chain(vectordb, docs, k=4):
    """
    Build a hybrid retrieval chain:
    - BM25 (keyword)
    - Vector (semantic)
    Combined using Reciprocal Rank Fusion (RRF).
    """

    # Keyword retriever (BM25)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k

    # Semantic retriever (vector)
    semantic = vectordb.as_retriever(search_kwargs={"k": k})

    # Hybrid retriever with Reciprocal Rank Fusion
    retriever = EnsembleRetriever(
        retrievers=[bm25, semantic],
        weights=None,       # None = use RRF automatically
        search_type="rrf"   # enables Reciprocal Rank Fusion
    )

    # Chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Wrap retrieval chain with metadata formatting
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    def chain_with_metadata(inputs):
        results = retriever.invoke(inputs["input"])
        formatted_context = format_docs_with_metadata(results)
        return llm.invoke(prompt.format(context=formatted_context, input=inputs["input"]))

    return chain_with_metadata
