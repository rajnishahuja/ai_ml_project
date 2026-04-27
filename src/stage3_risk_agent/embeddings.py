import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"


def get_embeddings_model():
    """Returns the globally configured Ollama embeddings client."""
    return OllamaEmbeddings(
        base_url=OLLAMA_HOST,
        model=OLLAMA_EMBED_MODEL,
    )


def embed_and_store(contract_text: str, document_id: str) -> str:
    """
    Takes raw contract text, splits it into digestible chunks,
    embeds them synchronously through local Ollama, and merges them into the local FAISS DB.
    """
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = str(FAISS_INDEX_DIR / "legal_contracts_index")

    logger.info(f"Chunking document {document_id}")

    # 1. Chunking logic specifically targeting legal syntax density
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(contract_text)

    # Bundle raw strings into proper Document DTOs representing state metadata
    docs = [
        Document(page_content=c, metadata={"document_id": document_id, "chunk_id": i})
        for i, c in enumerate(chunks)
    ]

    # 2. Embedding creation payload
    logger.info(
        f"Generating FAISS embeddings for {len(docs)} chunks via Ollama ({OLLAMA_EMBED_MODEL})"
    )
    embeddings = get_embeddings_model()

    # 3. Securely merging to FAISS Index Storage
    if os.path.exists(index_path):
        logger.info("FAISS index found. Appending vectors...")
        # allow_dangerous_deserialization=True is strictly required in newer LangChain versions to load local pickles
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(docs)
    else:
        logger.info("Initializing Genesis FAISS index...")
        vectorstore = FAISS.from_documents(docs, embeddings)

    # Save mutated state back sequentially to prevent corruption
    vectorstore.save_local(index_path)
    logger.info(f"Persisted {len(docs)} encoded chunks to vector storage.")

    return index_path


# ==========================================
# RETRIEVAL / EXPLORER UTILS
# ==========================================


def _get_faiss_index():
    """Helper to safely load the global index for querying"""
    index_path = str(FAISS_INDEX_DIR / "legal_contracts_index")
    if not os.path.exists(index_path):
        return None
    return FAISS.load_local(
        index_path, get_embeddings_model(), allow_dangerous_deserialization=True
    )


def get_all_document_ids() -> list:
    """Scans the FAISS dict and returns all unique document UUIDs"""
    vectorstore = _get_faiss_index()
    if not vectorstore:
        return []

    unique_ids = set()
    # FAISS docstore maps an internal hash to a langchain Document
    for doc in vectorstore.docstore._dict.values():
        doc_id = doc.metadata.get("document_id")
        if doc_id:
            unique_ids.add(doc_id)

    return list(unique_ids)


def get_document_chunks(document_id: str) -> list:
    """Retrieves all physical text chunks for a given document UUID"""
    vectorstore = _get_faiss_index()
    if not vectorstore:
        return []

    chunks = []
    for doc in vectorstore.docstore._dict.values():
        if doc.metadata.get("document_id") == document_id:
            chunks.append(
                {"chunk_id": doc.metadata.get("chunk_id", 0), "text": doc.page_content}
            )

    # Reassemble in exact sequential order
    chunks.sort(key=lambda x: x["chunk_id"])
    return chunks
