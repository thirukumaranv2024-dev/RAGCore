import { VectorSearchResult } from './types';

/**
 * Optional local cross-encoder adapter.
 * The Python service runs sentence-transformers/ms-marco-MiniLM-L-6-v2 on the
 * same machine. If it is not configured or unavailable, retrieval remains
 * functional and returns the original hybrid ranking.
 */
export interface RerankResult {
  results: VectorSearchResult[];
  applied: boolean;
  reason?: string;
}

const DEFAULT_RERANKER_URL = 'http://127.0.0.1:8001/rerank';

export async function rerankResults(
  query: string,
  candidates: VectorSearchResult[],
  topK: number
): Promise<RerankResult> {
  if (candidates.length < 2) {
    return { results: candidates.slice(0, topK), applied: false, reason: 'Not enough candidates' };
  }

  const endpoint = (import.meta.env.VITE_RERANKER_URL as string | undefined) || DEFAULT_RERANKER_URL;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 8000);

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        candidates: candidates.map((item, index) => ({
          id: item.chunk.id,
          text: item.chunk.content,
          original_score: item.score,
          original_rank: index + 1,
        })),
        top_k: topK,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      return { results: candidates.slice(0, topK), applied: false, reason: `Reranker HTTP ${response.status}` };
    }

    const data = await response.json();
    if (!Array.isArray(data.results)) {
      return { results: candidates.slice(0, topK), applied: false, reason: 'Invalid reranker response' };
    }

    const byId = new Map(candidates.map(item => [item.chunk.id, item]));
    const reranked = data.results
      .map((item: { id: string; score: number }) => {
        const original = byId.get(item.id);
        return original ? { ...original, score: Number.isFinite(item.score) ? item.score : original.score } : undefined;
      })
      .filter((item: VectorSearchResult | undefined): item is VectorSearchResult => !!item);

    // Do not drop candidates if the service returns a partial list.
    const included = new Set(reranked.map(item => item.chunk.id));
    const remaining = candidates.filter(item => !included.has(item.chunk.id));
    return { results: [...reranked, ...remaining].slice(0, topK), applied: reranked.length > 0 };
  } catch (error) {
    const reason = error instanceof Error ? error.message : 'Reranker unavailable';
    return { results: candidates.slice(0, topK), applied: false, reason };
  } finally {
    window.clearTimeout(timeout);
  }
}
