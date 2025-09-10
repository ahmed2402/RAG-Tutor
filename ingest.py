import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"]

def extract_chapter_section(text):
    # very simple heuristic: look for "Chapter" or "CHAPTER" headings on the page
    for line in text.splitlines()[:6]:
        if 'chapter' in line.lower():
            return line.strip()
    return None

def ingest_pdf_to_chroma(pdf_path, collection_name="book_collection"):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()                  
    for d in docs:
        chapter = extract_chapter_section(d.page_content)
        if chapter:
            d.metadata['chapter'] = chapter


    # 3) chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # 4) embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")    
    # 5) persist to Chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="vector_stores"
    )
    vectordb.persist()
    return vectordb

def retrieve_from_chroma(query, collection_name="book_collection", persist_directory="vector_stores"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
    )
    retrieved_results = db.similarity_search(query)
    return retrieved_results

def main():
    pdf_path = "data/book_collection/11.-The-Time-Machine-H.G.-Wells.pdf"
    collection_name = "book_collection"
    vectordb = ingest_pdf_to_chroma(pdf_path, collection_name)
    print(f"PDF '{pdf_path}' has been ingested into Chroma collection '{collection_name}'.")
        
if __name__ == "__main__":
    query = "You can show black is white by argument,” said Filby, “but you will never convince me."
    results = retrieve_from_chroma(query)
    print(results[0].page_content)