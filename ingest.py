import os
import logging
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.settings import (
    DATA_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    HF_MODEL_NAME,
    DEVICE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logging.basicConfig(
    filename=os.path.join("logs", "ingest.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def load_documents():
    loader = PyMuPDFLoader(DATA_PATH)
    docs = loader.load()
    logging.info("Loaded %d documents from %s", len(docs), DATA_PATH)
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logging.info(
        "Split into %d chunks (size=%d, overlap=%d)",
        len(chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return chunks

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

def persist_to_chroma(chunks):
    embeddings = build_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    for chunk in tqdm(chunks, desc="Adding chunks"):
        vector_store.add_documents([chunk])
    vector_store.persist()
    logging.info("Persisted %d chunks to %s", len(chunks), CHROMA_PATH)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs(os.path.dirname(CHROMA_PATH), exist_ok=True)
    documents = load_documents()
    chunks = split_documents(documents)
    persist_to_chroma(chunks)