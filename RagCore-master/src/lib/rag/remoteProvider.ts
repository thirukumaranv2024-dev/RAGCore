import { RemoteProviderConfig } from '@/types/settings';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;

// Retry configuration
const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 10000;

interface RetryConfig {
  maxRetries?: number;
  initialBackoff?: number;
  maxBackoff?: number;
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function calculateBackoff(attempt: number, initialBackoff: number, maxBackoff: number): number {
  const backoff = initialBackoff * Math.pow(2, attempt);
  return Math.min(backoff, maxBackoff);
}

async function fetchWithRetry(
  url: string,
  options: RequestInit,
  config: RetryConfig = {}
): Promise<Response> {
  const {
    maxRetries = MAX_RETRIES,
    initialBackoff = INITIAL_BACKOFF_MS,
    maxBackoff = MAX_BACKOFF_MS
  } = config;

  let lastError: Error | null = null;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);

      // Success
      if (response.ok) {
        return response;
      }

      // Don't retry on client errors (except 429 rate limit)
      if (response.status >= 400 && response.status < 500 && response.status !== 429) {
        return response;
      }

      // Retry on server errors (5xx) and rate limits (429)
      if (response.status === 429 || response.status >= 500) {
        const errorText = await response.text();
        lastError = new Error(`HTTP ${response.status}: ${errorText}`);

        // Don't retry on last attempt
        if (attempt < maxRetries) {
          const backoffMs = calculateBackoff(attempt, initialBackoff, maxBackoff);
          console.warn(`Request failed (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${backoffMs}ms...`);
          await sleep(backoffMs);
          continue;
        }
      }

      return response;
    } catch (error) {
      lastError = error instanceof Error ? error : new Error('Unknown error');

      // Network errors - retry
      if (attempt < maxRetries) {
        const backoffMs = calculateBackoff(attempt, initialBackoff, maxBackoff);
        console.warn(`Network error (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${backoffMs}ms...`);
        await sleep(backoffMs);
        continue;
      }
    }
  }

  throw lastError || new Error('Request failed after retries');
}

async function directRemoteEmbedding(
  text: string,
  config: RemoteProviderConfig
): Promise<number[]> {
  if (config.embeddingProvider === 'gemini') {
    const model = config.embeddingModel || 'text-embedding-004';
    const apiKey = config.embeddingApiKey;
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:embedContent?key=${apiKey}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: { parts: [{ text }] },
      }),
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Gemini embedding error (${res.status}): ${err}`);
    }
    const data = await res.json();
    return data.embedding?.values || [];
  }

  // OpenAI / Custom OpenAI-compatible
  const baseUrl = config.embeddingBaseUrl || 'https://api.openai.com/v1';
  const url = `${baseUrl.replace(/\/+$/, '')}/embeddings`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.embeddingApiKey}`,
      ...(config.embeddingCustomHeaders || {}),
    },
    body: JSON.stringify({
      model: config.embeddingModel || 'text-embedding-3-small',
      input: text,
    }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Embedding error (${res.status}): ${err}`);
  }
  const data = await res.json();
  return data.data?.[0]?.embedding || [];
}

export async function generateRemoteEmbedding(
  text: string,
  config: RemoteProviderConfig,
  retryCount: number = 0
): Promise<number[]> {
  const MAX_RETRIES = 3;
  const RATE_LIMIT_RETRY_DELAY = 5000;

  // Try direct provider call if API key is provided directly
  if (config.embeddingApiKey) {
    try {
      return await directRemoteEmbedding(text, config);
    } catch (directErr) {
      console.warn('Direct remote embedding attempt error:', directErr);
    }
  }

  // Fallback to Supabase Edge Function if configured
  if (SUPABASE_URL && !SUPABASE_URL.includes('supabase.com/dashboard')) {
    try {
      const response = await fetchWithRetry(`${SUPABASE_URL}/functions/v1/remote-embedding`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider: config.embeddingProvider,
          model: config.embeddingModel,
          text,
          baseUrl: config.embeddingBaseUrl,
          apiKey: config.embeddingApiKey,
          customHeaders: config.embeddingCustomHeaders,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        return data.embedding;
      }
    } catch (supabaseErr) {
      console.warn('Supabase remote embedding edge function failed:', supabaseErr);
    }
  }

  throw new Error(`Failed to generate embedding with ${config.embeddingProvider}. Please verify your API key in Settings.`);
}

export async function generateRemoteEmbeddings(
  texts: string[],
  config: RemoteProviderConfig,
  onProgress?: (current: number, total: number) => void
): Promise<number[][]> {
  const embeddings: number[][] = [];
  const CONCURRENCY = 4;

  for (let i = 0; i < texts.length; i += CONCURRENCY) {
    const batch = texts.slice(i, i + CONCURRENCY);
    const batchResults = await Promise.all(
      batch.map(text => generateRemoteEmbedding(text, config))
    );
    embeddings.push(...batchResults);

    if (onProgress) {
      onProgress(Math.min(i + CONCURRENCY, texts.length), texts.length);
    }
  }

  return embeddings;
}

async function directStreamRemoteLLM(
  messages: Array<{ role: string; content: string }>,
  config: RemoteProviderConfig,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (error: Error) => void
): Promise<void> {
  if (config.llmProvider === 'gemini') {
    const model = config.llmModel.replace(/^models\//, '') || 'gemini-1.5-flash';
    const apiKey = config.llmApiKey;
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?key=${apiKey}&alt=sse`;

    const contents = messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
      }));
    const systemMsg = messages.find(m => m.role === 'system');

    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents,
        systemInstruction: systemMsg ? { parts: [{ text: systemMsg.content }] } : undefined,
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      onError(new Error(`Gemini API error (${res.status}): ${err}`));
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      onError(new Error('No response body'));
      return;
    }
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const parsed = JSON.parse(line.slice(6));
            const token = parsed.candidates?.[0]?.content?.parts?.[0]?.text;
            if (token) onToken(token);
          } catch (error) {
            console.warn('Failed to parse Gemini streaming response:', error);
          }
        }
      }
    }
    onDone();
    return;
  }

  // OpenAI / Custom OpenAI-compatible
  const baseUrl = config.llmBaseUrl || 'https://api.openai.com/v1';
  const url = `${baseUrl.replace(/\/+$/, '')}/chat/completions`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${config.llmApiKey}`,
      ...(config.llmCustomHeaders || {}),
    },
    body: JSON.stringify({
      model: config.llmModel,
      messages: messages.map(m => ({ role: m.role, content: m.content })),
      stream: true,
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    onError(new Error(`LLM error (${res.status}): ${err}`));
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    onError(new Error('No response body'));
    return;
  }
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const jsonStr = line.slice(6).trim();
        if (jsonStr === '[DONE]') {
          onDone();
          return;
        }
        try {
          const parsed = JSON.parse(jsonStr);
          const token = parsed.choices?.[0]?.delta?.content;
          if (token) onToken(token);
        } catch (error) {
          console.warn('Failed to parse OpenAI-compatible streaming response:', error);
        }
      }
    }
  }
  onDone();
}

export async function streamRemoteLLM(
  messages: Array<{ role: string; content: string }>,
  config: RemoteProviderConfig,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (error: Error) => void,
  images?: string[]
): Promise<void> {
  // If API key is provided, try direct call first
  if (config.llmApiKey || config.llmProvider === 'custom') {
    try {
      await directStreamRemoteLLM(messages, config, onToken, onDone, onError);
      return;
    } catch (directErr) {
      console.warn('Direct stream attempt failed, checking fallback:', directErr);
    }
  }

  // Fallback to Supabase Edge Function
  if (SUPABASE_URL && !SUPABASE_URL.includes('supabase.com/dashboard')) {
    try {
      const response = await fetch(`${SUPABASE_URL}/functions/v1/remote-llm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider: config.llmProvider,
          model: config.llmModel,
          messages,
          images,
          stream: true,
          baseUrl: config.llmBaseUrl,
          apiKey: config.llmApiKey,
          customHeaders: config.llmCustomHeaders,
        }),
      });

      if (response.ok && response.body) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const jsonStr = line.slice(6).trim();
              if (jsonStr === '[DONE]') {
                onDone();
                return;
              }
              try {
                const parsed = JSON.parse(jsonStr);
                const token = parsed.choices?.[0]?.delta?.content;
                if (token) onToken(token);
              } catch (error) {
                console.warn('Failed to parse Supabase streaming response:', error);
              }
            }
          }
        }
        onDone();
        return;
      }
    } catch (supabaseErr) {
      console.warn('Supabase remote LLM stream failed:', supabaseErr);
    }
  }

  onError(new Error(`Failed to stream from ${config.llmProvider}. Please verify your API key in Settings.`));
}
