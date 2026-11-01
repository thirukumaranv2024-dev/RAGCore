# RAGCore Local Cross-Encoder Reranker

This optional service reranks hybrid-retrieval candidates using the local
`cross-encoder/ms-marco-MiniLM-L-6-v2` model. Query and document text are sent
only to this local process; the model is downloaded from Hugging Face on first
startup unless already cached. For a strict offline deployment, download/cache
the model before disconnecting the machine from the network.

## Windows setup (PowerShell)

```powershell
cd services/reranker
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8001
```

Health check: `http://127.0.0.1:8001/health`.
uvicorn app:app --host 127.0.0.1 --port 8001
The React app calls `http://127.0.0.1:8001/rerank` by default. Set
`VITE_RERANKER_URL` in the frontend `.env` to override it. If the service is
not available, the app safely uses the original hybrid-search ranking.

## Important

The model download requires internet access the first time unless the model is
already cached. Benchmark retrieval quality and latency with the service on and
off before making claims in the project report.
