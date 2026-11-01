import { TextChunk } from './types';

export async function generateEmbedding(
  text: string,
  baseUrl: string,
  model: string
): Promise<number[]> {
  const response = await fetch(`${baseUrl}/api/embeddings`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      prompt: text,
    }),
  });
  
  if (!response.ok) {
    const errorText = await response.text().catch(() => '<failed to read error body>');

    // Detect Ollama memory error and attempt fallback to smaller nomic model if applicable
    if (response.status >= 500 && /requires more system memory/i.test(errorText) && /nomic-embed-text/i.test(model)) {
      const baseModel = model.split(':')[0];
      const fallbackModel = `${baseModel}:v1.5`;
      try {
        console.warn(`Embedding model ${model} failed due to memory. Retrying with fallback model ${fallbackModel}.`);
        const retryResp = await fetch(`${baseUrl}/api/embeddings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: fallbackModel, prompt: text }),
        });

        if (retryResp.ok) {
          const data = await retryResp.json();
          // Optionally update telemetry or inform caller via console
          console.info(`Embedding succeeded with fallback model ${fallbackModel}`);
          return data.embedding;
        }

        const retryBody = await retryResp.text().catch(() => '<failed to read retry body>');
        throw new Error(`Fallback embedding failed: ${retryResp.status} ${retryResp.statusText} - ${retryBody}`);
      } catch (retryErr) {
        throw new Error(`Failed to generate embedding (original): ${errorText}. Fallback attempt failed: ${retryErr instanceof Error ? retryErr.message : String(retryErr)}`);
      }
    }

    throw new Error(`Failed to generate embedding: ${errorText}`);
  }
  
  const data = await response.json();
  return data.embedding;
}

export async function generateEmbeddingsBatch(
  texts: string[],
  baseUrl: string,
  model: string
): Promise<number[][]> {
  // Try modern Ollama /api/embed batch endpoint first (orders of magnitude faster)
  try {
    const response = await fetch(`${baseUrl}/api/embed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        input: texts,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      if (Array.isArray(data.embeddings) && data.embeddings.length === texts.length) {
        return data.embeddings;
      }
    }
  } catch (err) {
    console.warn('Batch /api/embed request failed, falling back to concurrent embedding', err);
  }

  // Fallback: Concurrent /api/embeddings requests with concurrency limit
  const CONCURRENCY = 6;
  const results: number[][] = [];
  for (let i = 0; i < texts.length; i += CONCURRENCY) {
    const batch = texts.slice(i, i + CONCURRENCY);
    const batchResults = await Promise.all(
      batch.map(text => generateEmbedding(text, baseUrl, model))
    );
    results.push(...batchResults);
  }
  return results;
}

export async function generateEmbeddings(
  chunks: TextChunk[],
  baseUrl: string,
  model: string,
  onProgress?: (current: number, total: number) => void
): Promise<TextChunk[]> {
  const embeddedChunks: TextChunk[] = new Array(chunks.length);
  const BATCH_SIZE = 50; // Batch 50 chunks per request for ultra-fast throughput
  const CONCURRENCY = 2; // 2 parallel batch workers

  const batchIndices: number[] = [];
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    batchIndices.push(i);
  }

  let completedChunks = 0;

  // Process batch chunks with parallel workers
  for (let b = 0; b < batchIndices.length; b += CONCURRENCY) {
    const currentBatches = batchIndices.slice(b, b + CONCURRENCY);

    await Promise.all(
      currentBatches.map(async (startIdx) => {
        const chunkBatch = chunks.slice(startIdx, startIdx + BATCH_SIZE);
        const textBatch = chunkBatch.map(c => c.content);

        try {
          const embeddings = await generateEmbeddingsBatch(textBatch, baseUrl, model);
          for (let j = 0; j < chunkBatch.length; j++) {
            embeddedChunks[startIdx + j] = {
              ...chunkBatch[j],
              embedding: embeddings[j],
            };
          }
        } catch (error) {
          console.error(`Failed to embed batch starting at index ${startIdx}:`, error);
          for (let j = 0; j < chunkBatch.length; j++) {
            const chunk = chunkBatch[j];
            try {
              const embedding = await generateEmbedding(chunk.content, baseUrl, model);
              embeddedChunks[startIdx + j] = { ...chunk, embedding };
            } catch {
              embeddedChunks[startIdx + j] = chunk;
            }
          }
        }

        completedChunks += chunkBatch.length;
        if (onProgress) {
          onProgress(Math.min(completedChunks, chunks.length), chunks.length);
        }
      })
    );
  }

  return embeddedChunks;
}

export async function embedQuery(
  query: string,
  baseUrl: string,
  model: string
): Promise<number[]> {
  return generateEmbedding(query, baseUrl, model);
}

