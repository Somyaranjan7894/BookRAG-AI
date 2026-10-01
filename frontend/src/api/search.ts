import { apiClient } from './client';
import { SearchRequest, SearchResponse } from '@/types/search';

export const searchApi = {
  /**
   * Execute two-stage semantic vector search and Cross-Encoder precision reranking (Phase 5 & Phase 6).
   * First stage retrieves candidate_k nearest chunks via dense vector search;
   * second stage scores joint (query, passage) pairs with a Cross-Encoder transformer.
   */
  async search(payload: SearchRequest): Promise<SearchResponse> {
    return apiClient<SearchResponse>('/api/v1/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  },
};
