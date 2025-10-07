from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.settings import (
    CHROMA_PATH,
    COLLECTION_NAME,
    HF_MODEL_NAME,
    DEVICE,
)

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name=HF_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )
    query = "黑土的有机质含量有怎样的特点？"
    docs = store.similarity_search(query, k=3)
    for idx, doc in enumerate(docs, 1):
        print(f"--- Result {idx} ---")
        print(doc.metadata)
        print(doc.page_content[:200], "...\n")

if __name__ == "__main__":
    main()