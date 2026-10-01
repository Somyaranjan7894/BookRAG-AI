/**
 * Semantic vector search and cross-encoder reranking types matching FastAPI schemas.
 */

export interface SearchResult {
  rank: number;
  original_rank?: number | null;
  chunk_id: string;
  document_id: string;
  page_number: number;
  text: string;
  similarity_score: number;
  reranker_score?: number | null;
  chunk_index?: number;
  metadata?: Record<string, unknown>;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  document_id?: string | null;
  candidate_k?: number;
  enable_reranking?: boolean;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
  document_id?: string | null;
  reranking_applied: boolean;
  candidate_count?: number | null;
}
