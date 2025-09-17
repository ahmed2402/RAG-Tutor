import os
import re
import time
from uuid import uuid4
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"]

#Global Variables
DEFAULT_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# heading detection patterns (order matters: Chapter -> Section -> numbered headings)
HEADING_PATTERNS = [
    re.compile(r'^(Chapter|CHAPTER)\b.*', flags=re.IGNORECASE | re.MULTILINE),
    re.compile(r'^(Section|SECTION)\b.*', flags=re.IGNORECASE | re.MULTILINE),
    # numeric section like "2.1 Introduction" or "3 Summary"
    re.compile(r'^\d+(?:\.\d+)*\s+.+', flags=re.MULTILINE),
    re.compile(r'^\d+(?:\.\d+)*\s+[A-Za-z].*', flags=re.MULTILINE),
    # Pattern to exclude "chapter. Here's the link:" text
    re.compile(r'^chapter\.\s+Here\'s\s+the\s+link:', flags=re.IGNORECASE | re.MULTILINE),
]


def _find_headings(text):
    """
    Return list of (pos, heading_text) sorted by pos.
    We deduplicate matches by start index to avoid duplicates from multiple patterns.
    """
    matches = {}
    for pat in HEADING_PATTERNS:
        for m in pat.finditer(text):
            matches[m.start()] = m.group().strip()
    items = sorted([(pos, matches[pos]) for pos in matches])
    return items


def _split_page_by_headings(page_text):
    """
    Split a single page by detected headings.
    Returns list of (heading_text, fragment_text).
    If no headings found, returns [("", page_text)].
    """
    headings = _find_headings(page_text)
    if not headings:
        return [("", page_text.strip())]

    fragments = []
    # Add prefix before first heading if it exists
    first_pos = headings[0][0]
    if first_pos > 0:
        prefix = page_text[:first_pos].strip()
        if prefix:
            fragments.append(("", prefix))

    # create fragments from each heading start to next heading start (or end)
    starts = [pos for pos, _ in headings] + [len(page_text)]
    for i, (pos, heading_text) in enumerate(headings):
        start = pos
        end = starts[i + 1]
        segment = page_text[start:end].strip()
        fragments.append((heading_text, segment))

    return fragments


def ingest_pdf_to_faiss(
    pdf_path,
    collection_name: str | None = None,
    hf_model: str = DEFAULT_HF_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    """
    Ingest PDF into Chroma with enriched metadata per chunk:
      - book_title, chapter, section, page_number, chunk_id, source

    Splitting strategy:
      - load PDF pages (PyPDFLoader gives page-level docs)
      - for each page: split by headings (Chapter/Section/numbered headings)
      - for each fragment: if fragment is long, further split using RecursiveCharacterTextSplitter
    """

    if collection_name is None:
        collection_name = os.path.splitext(os.path.basename(pdf_path))[0]

    start = time.process_time()
    # 1) Load pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # each doc typically has .page_content and metadata['page']
    print(f"[ingest] loaded {len(pages)} pages from {pdf_path}")

    # 2) Prepare splitter for long fragments
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # 3) Walk pages and create chunk Documents with metadata
    chunk_docs = []
    current_chapter = None  # carry-forward chapter across pages if found
    for page_doc in pages:
        page_text = page_doc.page_content or ""
        page_num = page_doc.metadata.get("page", None)
        # split page into semantic fragments by headings
        fragments = _split_page_by_headings(page_text)

        for frag_idx, (heading, frag_text) in enumerate(fragments):
            # determine chapter and section from heading
            chapter = None
            section = None
            if heading:
                low = heading.lower()
                if "chapter" in low:
                    chapter = heading
                    current_chapter = chapter
                elif "section" in low:
                    section = heading
                else:
                    # numeric heading likely a section "2.1 Intro"
                    if re.match(r'^\d+(?:\.\d+)*\s+', heading):
                        section = heading

            # if no explicit chapter on this fragment, inherit the last seen chapter
            if chapter is None:
                chapter = current_chapter

            # further split the fragment if it's large (using the text_splitter)
            # note: split_text returns a list of strings
            small_chunks = (
                text_splitter.split_text(frag_text)
                if len(frag_text) > chunk_size
                else [frag_text]
            )

            for sidx, chunk_text in enumerate(small_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunk_meta = {
                    "book_title": collection_name,
                    "chapter": chapter or "",
                    "section": section or "",
                    "page_number": page_num,
                    "source": os.path.basename(pdf_path),
                    "chunk_id": f"{os.path.basename(pdf_path)}_p{page_num}_f{frag_idx}_s{sidx}_{uuid4().hex[:8]}",
                }
                doc = Document(page_content=chunk_text, metadata=chunk_meta)
                chunk_docs.append(doc)

    print(f"[ingest] created {len(chunk_docs)} chunk documents (ready for embeddings)")

    # 4) Create embeddings and persist to Chroma
    embeddings = HuggingFaceEmbeddings(model_name=hf_model)
    vectordb = FAISS.from_documents(
        documents=chunk_docs,
        embedding=embeddings
    )
    faiss_index_path = f"vector_stores/{collection_name}_faiss"
    vectordb.save_local(faiss_index_path)

    print(f"[ingest] persisted collection '{collection_name}'")
    print(f"Response Time : ", time.process_time() - start)
    return vectordb, chunk_docs