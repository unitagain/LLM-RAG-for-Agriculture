# Agricultural RAG Playground

A lightweight Retrieval-Augmented Generation (RAG) project for agricultural knowledge. The repository demonstrates how to ingest the textbook `土壤学 (黄巧云) (Z-Library).pdf`, build a Chroma vector store with Qwen3 embeddings, and run a LangChain retrieval + generation pipeline powered by DashScope’s Qwen3 chat models.

## Why This Matters

- **Domain focus**: targets Chinese agricultural soil science content with structure and citations.
- **Lightweight stack**: pure Python, LangChain, Chroma (local persistence), HuggingFace embeddings, DashScope chat API.
- **Reproducible workflow**: ingestion → embedding → vector persistence → retrieval → answer generation.

## Repository Layout

```
├─ data/
│  ├─ raw/                   # Source PDFs (not tracked)
│  └─ chroma/                # Persisted Chroma index (generated)
├─ logs/                     # Ingestion & experiment logs
├─ src/
│  ├─ ingest.py              # PDF -> chunks -> embeddings -> Chroma
│  ├─ query_demo.py          # Quick similarity search sanity check
│  ├─ rag_pipeline.py        # Full RAG chain (retriever + Qwen3 LLM)
│  └─ settings.py            # Common configuration constants
└─ README.md
```

## Prerequisites

- Python 3.10 (tested with Anaconda environment `rag-agri`)
- (Optional) CUDA-enabled GPU for faster embeddings; otherwise CPU works.
- HuggingFace account to download `Qwen/Qwen3-Embedding-0.6B` (accept TOS if required).
- DashScope API key for Qwen3 chat completion.

## Setup

1. **Clone & enter**
   ```bash
   git clone <your-repo-url> learn_RAG
   cd learn_RAG
   ```

2. **Create environment (example with Conda)**
   ```bash
   conda create -n rag-agri python=3.10 -y
   conda activate rag-agri
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install torch --index-url https://download.pytorch.org/whl/cu121  # or CPU wheel
   pip install transformers accelerate safetensors
   pip install langchain langchain-community langchain-openai chromadb tqdm python-dotenv pymupdf
   pip install sentence-transformers
   # (Optional) silence deprecation warnings
   # pip install -U langchain-huggingface langchain-chroma
   ```

4. **Configure credentials**

   Create an `.env` file (not committed) or export environment variables before running scripts:
   ```bash
   echo "DASHSCOPE_API_KEY=sk-xxxxxxxx" >> .env
   echo "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1" >> .env
   echo "USE_CUDA=1" >> .env  # optional
   ```
   Then load them in your shell:
   ```bash
   # PowerShell
   $env:DASHSCOPE_API_KEY="sk-xxxxxxxx"
   $env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
   $env:USE_CUDA="1"
   ```

5. **Place PDF**
   - Drop `土壤学 (黄巧云) (Z-Library).pdf` into `data/raw/`.
   - Update `src/settings.py` if your filename differs.

## Workflow

### 1. Ingest & Build Vector Store
```bash
python -m src.ingest
```
- Uses `PyMuPDFLoader` to read the PDF (307 pages).
- Splits into ~641 chunks (`chunk_size=800`, `overlap=120`).
- Embeds with `Qwen/Qwen3-Embedding-0.6B` via HuggingFace pipeline.
- Persists vectors into `data/chroma/`.
- Progress & diagnostics in `logs/ingest.log`.

### 2. Validate Retrieval Only
```bash
python -m src.query_demo
```
- Loads the persisted Chroma collection.
- Runs a similarity search (`k=3` by default).
- Prints metadata (page numbers) and excerpt for manual inspection.

### 3. Run the RAG Pipeline
```bash
python -m src.rag_pipeline
```
- Wraps the Chroma retriever and DashScope Qwen3 chat model in a LangChain Runnable.
- Prompt template enforces Chinese structured answers with provenance notes.
- Default question: “东北黑土区的有机质管理要点是什么？”
- Modify `rag_pipeline.py` or invoke `rag_chain.invoke(<your question>)` from another module to customize.

## Configuration

Adjust key parameters in `src/settings.py`:
- `HF_MODEL_NAME`: swap to a larger embedding model (e.g., `Qwen/Qwen3-Embedding-4B`).
- `DEVICE`: toggle CUDA with `USE_CUDA` env var.
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: tune chunk granularity.
- `DASHSCOPE_*`: API credentials for generation model.

## Logs & Troubleshooting

- `logs/ingest.log` captures ingestion stats, HuggingFace download retries, and Chroma persistence status.
- First model download (≈1.2 GB) may trigger timeout retries or symlink warnings on Windows—these are expected; enable developer mode or run as admin to allow symlinks.
- If you hit `ModuleNotFoundError` for `langchain_openai`, ensure `pip install langchain-openai`. For missing `sentence_transformers`, install it explicitly.
- DashScope `openai.OpenAIError: api_key` means the environment variable is not set; double-check `.env` loading.

## Roadmap Ideas

- Replace deprecated classes with `langchain-huggingface` and `langchain-chroma` wrappers.
- Add reranking (`Qwen/Qwen3-Reranker-*`) and response citation formatting.
- Build evaluation scripts (batch Q/A, accuracy & latency metrics).
- Wrap the pipeline in a CLI or web UI (e.g., Gradio) for non-technical users.
- Extend corpus: integrate additional agricultural PDFs or structured datasets.

## License

Specify your preferred license (MIT, Apache-2.0, etc.) or update this section accordingly.
