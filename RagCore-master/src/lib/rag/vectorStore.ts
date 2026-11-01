import { TextChunk, VectorSearchResult } from './types';

// IndexedDB configuration for persistent vector storage
const DB_NAME = 'ragcore_vectors_db';
const STORE_NAME = 'chunks';
const DB_VERSION = 1;

function openIndexedDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      return reject(new Error('IndexedDB not supported'));
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Client-side BM25 Index for keyword search
class BM25Index {
  private k1 = 1.2;
  private b = 0.75;
  private docLengths: Map<string, number> = new Map();
  private avgDocLength = 0;
  private docFreqs: Map<string, number> = new Map();
  private termFreqs: Map<string, Map<string, number>> = new Map();
  private totalDocs = 0;

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length > 1);
  }

  build(chunks: TextChunk[]): void {
    this.docLengths.clear();
    this.docFreqs.clear();
    this.termFreqs.clear();
    this.totalDocs = chunks.length;

    let totalLength = 0;
    for (const chunk of chunks) {
      const tokens = this.tokenize(chunk.content);
      const length = tokens.length;
      this.docLengths.set(chunk.id, length);
      totalLength += length;

      const tf = new Map<string, number>();
      const seenTerms = new Set<string>();

      for (const token of tokens) {
        tf.set(token, (tf.get(token) || 0) + 1);
        seenTerms.add(token);
      }

      this.termFreqs.set(chunk.id, tf);
      for (const term of seenTerms) {
        this.docFreqs.set(term, (this.docFreqs.get(term) || 0) + 1);
      }
    }

    this.avgDocLength = this.totalDocs > 0 ? totalLength / this.totalDocs : 0;
  }

  search(query: string): Map<string, number> {
    const scores = new Map<string, number>();
    const queryTokens = this.tokenize(query);
    if (queryTokens.length === 0 || this.totalDocs === 0) return scores;

    for (const token of queryTokens) {
      const df = this.docFreqs.get(token) || 0;
      if (df === 0) continue;

      const idf = Math.log((this.totalDocs - df + 0.5) / (df + 0.5) + 1);

      for (const [docId, tfMap] of this.termFreqs.entries()) {
        const tf = tfMap.get(token) || 0;
        if (tf === 0) continue;

        const docLen = this.docLengths.get(docId) || this.avgDocLength;
        const num = tf * (this.k1 + 1);
        const denom = tf + this.k1 * (1 - this.b + this.b * (docLen / (this.avgDocLength || 1)));
        const score = idf * (num / denom);

        scores.set(docId, (scores.get(docId) || 0) + score);
      }
    }

    return scores;
  }
}

class VectorStore {
  private chunks: Map<string, TextChunk> = new Map();
  private bm25: BM25Index = new BM25Index();
  private dbPromise: Promise<IDBDatabase> | null = null;
  
  constructor() {
    this.initPersistence();
  }

  private async initPersistence(): Promise<void> {
    try {
      this.dbPromise = openIndexedDB();
      const db = await this.dbPromise;
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();

      req.onsuccess = () => {
        const storedChunks: TextChunk[] = req.result || [];
        if (storedChunks.length > 0) {
          storedChunks.forEach(chunk => this.chunks.set(chunk.id, chunk));
          this.bm25.build(Array.from(this.chunks.values()));
          console.info(`[VectorStore] Loaded ${storedChunks.length} persistent chunks from IndexedDB`);
        }
      };
    } catch (err) {
      console.warn('[VectorStore] IndexedDB persistence not available:', err);
    }
  }

  private async persistChunk(chunk: TextChunk): Promise<void> {
    try {
      if (!this.dbPromise) return;
      const db = await this.dbPromise;
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(chunk);
    } catch (e) {
      console.warn('[VectorStore] Failed to persist chunk:', e);
    }
  }

  private async persistManyChunks(chunks: TextChunk[]): Promise<void> {
    try {
      if (!this.dbPromise || chunks.length === 0) return;
      const db = await this.dbPromise;
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      chunks.forEach(c => store.put(c));
    } catch (e) {
      console.warn('[VectorStore] Failed to persist chunks:', e);
    }
  }

  add(chunk: TextChunk): void {
    if (!chunk.embedding) {
      console.warn('Adding chunk without embedding:', chunk.id);
    }
    this.chunks.set(chunk.id, chunk);
    this.bm25.build(Array.from(this.chunks.values()));
    this.persistChunk(chunk);
  }
  
  addMany(chunks: TextChunk[]): void {
    chunks.forEach(chunk => this.chunks.set(chunk.id, chunk));
    this.bm25.build(Array.from(this.chunks.values()));
    this.persistManyChunks(chunks);
  }
  
  get(id: string): TextChunk | undefined {
    return this.chunks.get(id);
  }
  
  getAll(): TextChunk[] {
    return Array.from(this.chunks.values());
  }
  
  clear(): void {
    this.chunks.clear();
    this.bm25.build([]);
    if (this.dbPromise) {
      this.dbPromise.then(db => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).clear();
      }).catch(err => console.warn('[VectorStore] Clear DB failed:', err));
    }
  }
  
  size(): number {
    return this.chunks.size;
  }
  
  // Cosine similarity between two vectors
  private cosineSimilarity(a: number[], b: number[]): number {
    if (!a || !b || a.length !== b.length || a.length === 0) {
      return 0;
    }
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
  }
  
  // Search for similar chunks using cosine similarity
  search(queryEmbedding: number[], topK: number = 5): VectorSearchResult[] {
    const results: VectorSearchResult[] = [];
    
    for (const chunk of this.chunks.values()) {
      if (!chunk.embedding || chunk.embedding.length !== queryEmbedding.length) continue;
      
      const score = this.cosineSimilarity(queryEmbedding, chunk.embedding);
      if (score > 0) {
        results.push({ chunk, score });
      }
    }
    
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
  
  // Hybrid search combining BM25 Keyword Search + Vector Cosine Search using Reciprocal Rank Fusion
  searchHybrid(
    queryText: string,
    queryEmbedding: number[],
    topK: number = 5,
    sourcePriorities: Map<string, number> = new Map()
  ): VectorSearchResult[] {
    const allChunks = Array.from(this.chunks.values()).filter(
      c => c.embedding && c.embedding.length === queryEmbedding.length
    );

    if (allChunks.length === 0) {
      return [];
    }

    // 1. Vector Cosine Ranking
    const vectorScored: { chunk: TextChunk; score: number }[] = [];
    for (const chunk of allChunks) {
      const score = this.cosineSimilarity(queryEmbedding, chunk.embedding!);
      if (score > 0) {
        vectorScored.push({ chunk, score });
      }
    }
    vectorScored.sort((a, b) => b.score - a.score);

    // 2. BM25 Keyword Ranking
    const bm25Scores = this.bm25.search(queryText);
    const bm25Scored = allChunks
      .map(chunk => ({ chunk, score: bm25Scores.get(chunk.id) || 0 }))
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score);

    // 3. Reciprocal Rank Fusion (RRF)
    const rrfMap = new Map<string, { chunk: TextChunk; rrfScore: number; vecScore: number; bm25Score: number }>();
    const RRF_K = 60;

    vectorScored.forEach((item, rank) => {
      const rrf = 1 / (RRF_K + rank + 1);
      rrfMap.set(item.chunk.id, {
        chunk: item.chunk,
        rrfScore: rrf * 0.7, // 70% vector weight
        vecScore: item.score,
        bm25Score: 0,
      });
    });

    bm25Scored.forEach((item, rank) => {
      const rrf = 1 / (RRF_K + rank + 1);
      const existing = rrfMap.get(item.chunk.id);
      if (existing) {
        existing.rrfScore += rrf * 0.3; // 30% keyword weight
        existing.bm25Score = item.score;
      } else {
        rrfMap.set(item.chunk.id, {
          chunk: item.chunk,
          rrfScore: rrf * 0.3,
          vecScore: 0,
          bm25Score: item.score,
        });
      }
    });

    // 4. Apply Source Priority Weighting
    const results: VectorSearchResult[] = Array.from(rrfMap.values()).map(item => {
      const priority = item.chunk.metadata.priority || sourcePriorities.get(item.chunk.metadata.source) || 3;
      const priorityMultiplier = 0.5 + (priority * 0.2);
      const weightedScore = item.rrfScore * priorityMultiplier;

      return {
        chunk: item.chunk,
        score: item.vecScore || item.rrfScore,
        weightedScore,
      };
    });

    return results
      .sort((a, b) => (b.weightedScore || b.score) - (a.weightedScore || a.score))
      .slice(0, topK);
  }

  // Weighted search (backward compatible fallback)
  searchWeighted(
    queryEmbedding: number[],
    topK: number = 5,
    sourcePriorities: Map<string, number> = new Map()
  ): VectorSearchResult[] {
    const results: VectorSearchResult[] = [];
    
    for (const chunk of this.chunks.values()) {
      if (!chunk.embedding || chunk.embedding.length !== queryEmbedding.length) continue;
      
      const baseScore = this.cosineSimilarity(queryEmbedding, chunk.embedding);
      if (baseScore <= 0) continue;
      
      const priority = chunk.metadata.priority || sourcePriorities.get(chunk.metadata.source) || 3;
      const priorityMultiplier = 0.5 + (priority * 0.2);
      const weightedScore = baseScore * priorityMultiplier;
      
      results.push({ 
        chunk, 
        score: baseScore,
        weightedScore 
      });
    }
    
    return results
      .sort((a, b) => (b.weightedScore || b.score) - (a.weightedScore || a.score))
      .slice(0, topK);
  }
  
  // Get chunks by source
  getBySource(source: string): TextChunk[] {
    return Array.from(this.chunks.values()).filter(
      chunk => chunk.metadata.source === source
    );
  }
  
  // Remove chunks by source
  removeBySource(source: string): number {
    let removed = 0;
    const toRemove: string[] = [];
    for (const [id, chunk] of this.chunks.entries()) {
      if (chunk.metadata.source === source) {
        this.chunks.delete(id);
        toRemove.push(id);
        removed++;
      }
    }
    this.bm25.build(Array.from(this.chunks.values()));

    if (this.dbPromise && toRemove.length > 0) {
      this.dbPromise.then(db => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        toRemove.forEach(id => store.delete(id));
      }).catch(err => console.warn('[VectorStore] Delete from DB failed:', err));
    }
    return removed;
  }
}

// Singleton instance
export const vectorStore = new VectorStore();

