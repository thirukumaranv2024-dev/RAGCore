export interface ChunkMetadata {
  source: string;
  sourceType: 'pdf' | 'url' | 'text' | 'office' | 'image';
  pageNumber?: number;
  chunkIndex: number;
  totalChunks: number;
  priority?: number; // 1-5, where 5 is highest priority
}

export interface TextChunk {
  id: string;
  content: string;
  metadata: ChunkMetadata;
  embedding?: number[];
}

export interface VectorSearchResult {
  chunk: TextChunk;
  score: number;
  weightedScore?: number; // Score adjusted by source priority
}

export type SourcePriority = 1 | 2 | 3 | 4 | 5;

export interface DocumentSource {
  id: string;
  name: string;
  type: 'pdf' | 'url' | 'text' | 'office' | 'image';
  chunks: number;
  priority: SourcePriority;
  addedAt: number;
  usageStats: SourceUsageStats;
  data?: string; // Optional raw data (e.g., base64 for images)
}

export interface SourceUsageStats {
  timesUsed: number; // How many times this source appeared in answers
  positiveRatings: number; // Number of positive ratings on answers using this source
  negativeRatings: number; // Number of negative ratings on answers using this source
  lastUsed?: number; // Timestamp of last use
  learningScore: number; // Calculated score for automatic priority adjustment (0-1)
  confidenceInterval: [number, number]; // 95% confidence interval [lower, upper]
  feedbackHistory: FeedbackEvent[]; // History of all feedback events
}

export interface FeedbackEvent {
  timestamp: number;
  isPositive: boolean;
  weight: number; // Time-decayed weight (1.0 = recent, < 1.0 = older)
}

export interface ABTest {
  id: string;
  name: string;
  description: string;
  startedAt: number;
  endedAt?: number;
  configA: SourceConfiguration; // Control configuration
  configB: SourceConfiguration; // Experimental configuration
  activeConfig: 'A' | 'B'; // Currently active configuration
  results: ABTestResults;
}

export interface SourceConfiguration {
  name: string;
  sourcePriorities: Map<string, SourcePriority>; // sourceId -> priority
}

export interface ABTestResults {
  configAStats: {
    questionsAnswered: number;
    positiveRatings: number;
    negativeRatings: number;
    avgConfidence: number;
  };
  configBStats: {
    questionsAnswered: number;
    positiveRatings: number;
    negativeRatings: number;
    avgConfidence: number;
  };
  winner?: 'A' | 'B' | 'inconclusive';
  pValue?: number; // Statistical significance
}

export interface SourceRecommendation {
  sourceId: string;
  sourceName: string;
  type: 'increase_priority' | 'decrease_priority' | 'remove' | 'keep_current';
  currentPriority: SourcePriority;
  suggestedPriority?: SourcePriority;
  reason: string;
  confidence: 'high' | 'medium' | 'low';
  impact: 'high' | 'medium' | 'low';
  metrics: {
    learningScore: number;
    confidenceInterval: [number, number];
    totalRatings: number;
    positiveRate: number;
    timesUsed: number;
    daysSinceLastUse?: number;
  };
}

export interface OllamaModelParams {
  temperature: number;
  contextLength: number;
  topP: number;
  topK: number;
  repeatPenalty: number;
}

export interface OllamaConfig {
  baseUrl: string;
  model: string;
  embeddingModel: string;
  params: OllamaModelParams;
}

export const DEFAULT_MODEL_PARAMS: OllamaModelParams = {
  temperature: 0.7,
  contextLength: 8192, // Increased for better "all size" parsing
  topP: 0.9,
  topK: 40,
  repeatPenalty: 1.1,
};

export const PREDEFINED_LLM_MODELS = [
  // Best for Low Spec (Fast & Reliable)
  { name: 'llama3.2:3b', description: 'BEST FOR LOW SPEC - Very fast, smart for its size' },
  { name: 'llama3.2:1b', description: 'FASTEST - Ultra lightweight, good for simple tasks' },
  { name: 'phi3.5:latest', description: 'BEST REASONING (3.8B) - Microsoft model, great for facts' },

  // Best Quality (Mid/High Spec)
  { name: 'llama3.1:8b', description: 'BEST ALL-ROUNDER - High quality, good performance' },
  { name: 'deepseek-r1:8b', description: 'BEST THINKING - Advanced reasoning and logic' },
  { name: 'mistral:7b', description: 'RELIABLE - Classic high-performance model' },

  // Best for Vision (Images)
  { name: 'moondream:latest', description: 'BEST VISION (LOW SPEC) - 1.6B tiny vision expert' },
  { name: 'llama3.2-vision:11b', description: 'BEST VISION (QUALITY) - Meta Llama 3.2 Vision' },
  { name: 'llava:7b', description: 'STABLE VISION - LLaVA 1.5 7B' },

  // Large Scale (Requires High RAM)
  { name: 'llama3.1:70b', description: 'ULTIMATE QUALITY - Very slow on most PCs' },
  { name: 'deepseek-r1:32b', description: 'HEAVY THINKING - DeepSeek R1 32B' },
];

export const PREDEFINED_EMBEDDING_MODELS = [
  // Nomic Models
  { name: 'nomic-embed-text:latest', description: 'Nomic Embed - Best quality', dimensions: 768 },
  { name: 'nomic-embed-text:v1.5', description: 'Nomic Embed v1.5 - Stable version', dimensions: 768 },

  // MixedBread Models
  { name: 'mxbai-embed-large:latest', description: 'MixedBread Large - High quality', dimensions: 1024 },
  { name: 'mxbai-embed-large:335m', description: 'MixedBread Large 335M - High quality', dimensions: 1024 },

  // BGE Models
  { name: 'bge-large:335m', description: 'BGE Large - High performance', dimensions: 1024 },
  { name: 'bge-m3:567m', description: 'BGE M3 - Multilingual', dimensions: 1024 },
  { name: 'bge-base:109m', description: 'BGE Base - Balanced', dimensions: 768 },
  { name: 'bge-small:33m', description: 'BGE Small - Fast', dimensions: 384 },

  // All-MiniLM Models
  { name: 'all-minilm:latest', description: 'MiniLM - Very fast', dimensions: 384 },
  { name: 'all-minilm:33m', description: 'MiniLM 33M - Efficient', dimensions: 384 },
  { name: 'all-minilm:22m', description: 'MiniLM 22M - Ultra fast', dimensions: 384 },

  // Snowflake Models
  { name: 'snowflake-arctic-embed:335m', description: 'Arctic Embed 335M - Enterprise', dimensions: 1024 },
  { name: 'snowflake-arctic-embed:137m', description: 'Arctic Embed 137M - Balanced', dimensions: 768 },
  { name: 'snowflake-arctic-embed:33m', description: 'Arctic Embed 33M - Fast', dimensions: 384 },

  // Other Embedding Models
  { name: 'llama3-embeddings:8b', description: 'Llama 3 Embeddings - Llama based', dimensions: 4096 },
  { name: 'e5-large:335m', description: 'E5 Large - General purpose', dimensions: 1024 },
  { name: 'e5-base:109m', description: 'E5 Base - Efficient', dimensions: 768 },
  { name: 'gte-large:335m', description: 'GTE Large - High quality', dimensions: 1024 },
  { name: 'paraphrase-multilingual:278m', description: 'Paraphrase Multilingual - 50+ languages', dimensions: 768 },
];
