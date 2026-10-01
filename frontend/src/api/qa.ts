import { apiClient } from './client';
import {
  GroundedAnswerRequest,
  GroundedAnswerResponse,
  QARequest,
  QAResponse,
} from '@/types/qa';

export const qaApi = {
  /**
   * Submit a question for full grounded RAG synthesis with sentence-level NLI verification
   * and deterministic book citations (Phase 9 + Phase 10).
   */
  async askGroundedQuestion(payload: GroundedAnswerRequest): Promise<GroundedAnswerResponse> {
    return apiClient<GroundedAnswerResponse>('/api/v1/grounded-answer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  },

  /**
   * Submit a question for exact extractive span extraction with RoBERTa SQuAD2 (Phase 7).
   */
  async askExtractiveQuestion(payload: QARequest): Promise<QAResponse> {
    return apiClient<QAResponse>('/api/v1/qa', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });
  },
};
