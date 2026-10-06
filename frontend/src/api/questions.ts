import { apiClient } from './client';
import {
  QuestionGenerationRequest,
  QuestionGenerationResponse,
} from '@/types/questionGeneration';

export const questionsApi = {
  /**
   * Request controlled, evidence-grounded question generation for a document.
   */
  async generateQuestions(
    payload: QuestionGenerationRequest
  ): Promise<QuestionGenerationResponse> {
    return apiClient<QuestionGenerationResponse>('/api/v1/questions/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      timeoutMs: 300000,
    });
  },
};
