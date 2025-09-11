# qa_chain.py
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate

groq_api_key = os.environ["GROQ_API_KEY"]
llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    You are a tutor limited to the given book excerpts.
    Answer only from the book. If not enough info → say: ❌ Insufficient evidence.
    Always provide [Chapter | Section | Page].
    -Rules:
    - Always cite the book.
    - No outside knowledge.
    - Refuse if confidence is low.
    Context:{context}
    Question: {input}
    Answer:
    """
)

def build_chain(vectordb, k=4):
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    document_chain = create_stuff_documents_chain(llm,prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
   
    return retrieval_chain
