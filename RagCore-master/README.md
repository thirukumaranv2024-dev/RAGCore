# RAGCore

RAGCore is a retrieval-augmented generation (RAG) learning application built with React, TypeScript, Vite, and Supabase. It includes document-based retrieval and learning tools such as quizzes, notes, flashcards, and mind maps. An optional local cross-encoder service can rerank retrieved results.

## Prerequisites

- Node.js and npm
- A Supabase project and its public frontend configuration
- Python 3 (only if running the optional local reranker)
- Git (optional, for cloning)

## Installation

1. Clone the repository and enter its directory:

   ```bash
   git clone https://github.com/thirukumaranv2024-dev/RAGCore.git
   cd RAGCore
   ```

   Alternatively, extract the project ZIP and open its root directory.

2. Install JavaScript dependencies:

   ```bash
   npm install
   ```

## Environment variables

Copy `.env.example` to `.env` in the project root. In Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` with values for your own setup:

```dotenv
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your-supabase-publishable-key
VITE_SUPABASE_PROJECT_ID=your-project-id

# Optional settings (uncomment only when needed)
# VITE_OLLAMA_URL=http://localhost:11434
# VITE_RERANKER_URL=http://127.0.0.1:8001
```

Get the Supabase values from your Supabase project dashboard. **Never commit `.env`, private API keys, service-role keys, or passwords.** Variables prefixed with `VITE_` are exposed to browser code; do not place secrets in them. The optional API-key placeholders in `.env.example` must not be used for confidential keys in a deployed frontend.

## Run locally

From the project root:

```bash
npm run dev
```

Open **http://localhost:8080** in your browser (the port configured in `vite.config.ts`).

## Optional: local reranker

The local cross-encoder reranker improves ordering of retrieved passages. To start it on Windows PowerShell, open a second terminal at the project root:

```powershell
cd services/reranker
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8001
```

Check **http://127.0.0.1:8001/health**. The reranker model may download on its first launch. If the service is unavailable, the frontend falls back to its original retrieval ranking. See `services/reranker/README.md` for details.

## Build and preview

```bash
npm run build
npm run preview
```

## Project structure

- `src/` — React UI, learning tools, and RAG logic
- `src/lib/rag/` — chunking, embeddings, retrieval, and reranking modules
- `src/integrations/supabase/` — Supabase integration
- `services/reranker/` — optional Python reranking service
- `.env.example` — environment-variable template without real credentials

## Troubleshooting

- If `npm` is not recognized, install Node.js and reopen the terminal.
- If Supabase initialization fails, check that `.env` contains valid project values and restart the Vite server.
- If the configured port is busy, stop the other process or change `server.port` in `vite.config.ts`.
- If reranking is unavailable, check that the optional Python service is running on port 8001.
