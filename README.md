# 📚 RAG Tutor

RAG Tutor is a Streamlit-based application that leverages Retrieval-Augmented Generation (RAG) to provide answers to questions based *only* on the content of an uploaded PDF book. This ensures that responses are factual and directly supported by the provided text, making it an excellent tool for educational purposes, research, or in-depth document analysis.

## ✨ Features

- **PDF Ingestion:** Upload any PDF book, and the application will process its content.
- **Intelligent Chunking:** PDFs are intelligently split by headings (chapters, sections) and further chunked to optimize retrieval.
- **Hybrid Retrieval:** Combines BM25 (keyword-based) and Vector (semantic) retrieval methods using Reciprocal Rank Fusion (RRF) for highly relevant context retrieval.
- **Context-Aware LLM:** Utilizes a Large Language Model (LLM) (Groq's Llama-3.1-8b-instant) to generate answers *only* from the provided context.
- **Source Citation:** Every key point in the answer includes inline citations (Chapter, Section, Page Number) to verify information directly from the source.
- **Streamlit Interface:** An intuitive web interface for uploading PDFs and asking questions.

## 🚀 Setup and Installation

Follow these steps to set up and run RAG Tutor locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RAG-Tutor.git
cd RAG-Tutor
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys

The application requires API keys for HuggingFace and Groq. Create a `.env` file in the root directory of the project with the following content:

```
HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

- **HuggingFace API Token:** Obtain from [HuggingFace Settings](https://huggingface.co/settings/tokens).
- **Groq API Key:** Obtain from [Groq Console](https://console.groq.com/keys).

## 🏃 How to Run

After completing the setup, run the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your web browser.

## 🌐 Deployment

This application is deployed and can be accessed at: [https://rag-tutor-by-ahmed.streamlit.app/](https://rag-tutor-by-ahmed.streamlit.app/)


## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── chain.py                # Defines the RAG chain and hybrid retrieval logic
├── data/                   # Directory for storing uploaded PDF books
│   └── book_collection/    # Contains uploaded PDFs
│       └── your_book.pdf
├── ingest.py               # Handles PDF loading, text splitting, and vector store creation
├── requirements.txt        # Project dependencies
├── vector_stores/          # Stores ChromaDB vector databases
│   └── (auto-generated)
└── .env                    # Environment variables (API keys)
```

## 🛠️ How it Works

1.  **PDF Upload & Ingestion (`app.py` -> `ingest.py`):**
    *   A user uploads a PDF file via the Streamlit interface.
    *   `ingest.py` uses `PyPDFLoader` to load the PDF pages.
    *   Pages are semantically split based on detected headings (chapters, sections) to maintain context.
    *   Larger fragments are further split using `RecursiveCharacterTextSplitter`.
    *   Each chunk is converted into an embedding using `HuggingFaceEmbeddings` and stored in a Chroma vector database. Metadata (book title, chapter, section, page, chunk ID) is enriched for better retrieval and citation.

2.  **RAG Chain Construction (`app.py` -> `chain.py`):**
    *   `chain.py` builds the core RAG chain.
    *   **Hybrid Retrieval:** It employs an `EnsembleRetriever` combining `BM25Retriever` (keyword search) and a vector-based retriever (semantic search) from ChromaDB. These are merged using Reciprocal Rank Fusion (RRF) to get the most relevant documents.
    *   **LLM Integration:** A `ChatGroq` model (e.g., Llama-3.1-8b-instant) is used as the language model.
    *   **Prompt Engineering:** A `ChatPromptTemplate` is used to instruct the LLM to answer *only* from the provided context and to include inline citations.

3.  **Question Answering (`app.py` -> `chain.py`):**
    *   When a user asks a question, `app.py` invokes the RAG chain.
    *   The `retriever` fetches the most relevant document chunks from the vector store based on the question.
    *   These chunks, along with their metadata, are formatted and passed to the LLM via the prompt.
    *   The LLM generates an answer, adhering to the prompt's instructions for context-only responses and inline citations.
    *   The answer and source documents are displayed in the Streamlit interface.

## 👨‍💻 Developer

This project was developed by **Ahmed Raza**.

- **LinkedIn**: [https://www.linkedin.com/in/ahmvd/](https://www.linkedin.com/in/ahmvd/)
- **Email**: [ahmedraza312682@gmail.com](mailto:ahmedraza312682@gmail.com)
- **GitHub**: [ahmed2402](https://github.com/ahmed2402)
