import os

DATA_PATH = os.path.join("data", "raw", "土壤学 (黄巧云) (Z-Library).pdf")
CHROMA_PATH = os.path.join("data", "chroma")
COLLECTION_NAME = "soil_science"
HF_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-your-api-key")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")