Project Summary — RAGCore-main

Overview:
- This is a frontend application built with React + TypeScript using Vite as the build tool.
- UI is implemented with shadcn-style components (Radix + Tailwind CSS) and many reusable UI primitives in `src/components/ui`.
- The app implements Retrieval-Augmented Generation (RAG) features: document ingestion (PDF, text, office, images), chunking, embeddings, local vector store, document search, and streaming LLM responses.
- Supports both local mode (Ollama) and remote provider mode (remote LLM and remote embeddings).
- Integrates with Supabase for edge functions (OCR) and optional remote storage/remote embeddings.

Main technologies:
- Frontend:
  - React 18 + TypeScript
  - Vite (dev server & bundler)
  - Tailwind CSS + shadcn component set (Radix + Tailwind)
  - React Router, React Query
- AI / RAG stack (client-side):
  - Local Ollama integration for LLM and embeddings (`/api/chat` calls)
  - Remote provider interface for embeddings and streaming LLM (`src/lib/rag/remoteProvider.ts`)
  - Chunker, PDF parser, URL parser, embedding generation in `src/lib/rag`
  - Vector store (in-memory client-side) for retrieval
- Integrations:
  - Supabase client in `src/integrations/supabase/client.ts`
  - Edge functions under `supabase/functions/` (examples: OCR, remote embedding, remote-llm)

Key files & folders (where to look):
- `src/main.tsx` — App entry
- `src/App.tsx` — Routes and top-level layout
- `src/pages/Index.tsx` — Main chat / upload UI and Ollama configuration
- `src/api.ts` — High-level API helpers used by UI (wraps ragService)
- `src/lib/rag/` — Core RAG implementation
  - `ragService.ts` — Main orchestrator (parsing, chunking, embeddings, query, generateResponse)
  - `chunker.ts` — Break documents into chunks
  - `embeddings.ts` — Calls to generate embeddings (local Ollama or remote)
  - `pdfParser.ts` — PDF parsing using `pdfjs-dist`
  - `vectorStore.ts` — In-memory vector store + search
  - `remoteProvider.ts` — Remote LLM/embeddings streaming helpers
- `src/components/chat/` — Chat UI: upload zone, messages, sources, streaming
- `src/components/settings/` — Settings dialog for choosing mode (local vs remote)
- `supabase/` — Supabase config and serverless function examples

How the app works (high level):
1. User uploads a PDF / text / image / URL.
2. App parses the document (`pdfParser`, `parseURL`), runs OCR for images (via Supabase edge function), and chunks content (`chunker`).
3. App generates embeddings for each chunk. Depending on settings, it either:
   - Calls local Ollama embedding model (if `mode` is `local`), or
   - Calls a remote embedding API through `remoteProvider`.
4. Embeddings are stored in the in-memory `vectorStore`.
5. On user query, the app embeds the query, searches the vector store for top-k chunks, and builds a context.
6. The app streams a response from the LLM (Ollama local `/api/chat` or remote provider streaming). Streaming tokens are rendered in the chat UI.

Configuration & environment:
- `import.meta.env.VITE_OLLAMA_URL` used as fallback base URL for local Ollama (default `http://localhost:11434`).
- `.env.example` exists in the repo root; fill in `VITE_SUPABASE_URL` and other VITE_ env vars as needed for Supabase integration.
- Settings accessible in the UI allow switching between `local` (Ollama) and `remote` modes and choosing models/embedding models.

Running the project (dev):
- Install dependencies: `npm install`
- Start dev server: `npm run dev` (Vite will open at a local port, e.g., `http://localhost:8081/`)
- The app will attempt to reach a local Ollama instance at `http://localhost:11434` when using Local Mode. If Ollama is not running, configure Remote Mode or start Ollama.

Troubleshooting common issues:
- "Ollama runner process has terminated" or 5xx errors: choose a smaller model, lower `contextLength` in settings, or disable images/vision features. The repo includes defensive request logic to clamp `num_ctx` before sending to Ollama.
- Port conflicts: Vite will try another port; check terminal output for the actual URL.
- OCR failures: OCR is performed via Supabase edge function at `supabase/functions/ocr-image`; ensure Supabase URL is set and function deployed.

Development notes & extension points:
- Vector store is client-side. For production, consider a server-side or persistent vector DB (Pinecone, Milvus, Supabase vectors).
- Large models and vision models require significant local resources; local Ollama usage is intended for powerful devices.
- Remote mode supports streaming remote LLMs and remote embedding providers — useful for low-spec machines.
- Add persistent storage for sources and vectors if users need long-lived indices.

License & README:
- See `README.md` in the repo root for project overview and screenshots.
- `LICENSE` is included at the repo root.

Contact / Next steps I can do for you:
- Produce a PDF of this summary and save to the repo (`docs/PROJECT_SUMMARY.pdf`) (I can attempt to convert it here if you allow running a conversion step).
- Create a short developer quickstart with setup examples and recommended Ollama model choices.

-- End of summary --
