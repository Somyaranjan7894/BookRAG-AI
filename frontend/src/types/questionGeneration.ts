/**
 * Phase 22 Controlled Question Generation TypeScript Definitions
 */

export type QuestionType =
  | 'direct_fact'
  | 'factual'
  | 'definition'
  | 'explanation'
  | 'comparison'
  | 'numerical'
  | 'numerical_fact'
  | 'reasoning'
  | 'multi_page_synthesis'
  | 'who'
  | 'what'
  | 'when'
  | 'where'
  | 'why'
  | 'how'
  | 'how_many';

export type QuestionDifficulty = 'easy' | 'medium' | 'hard';

export interface GeneratedQuestion {
  question_id?: string;
  question: string;
  answer: string;
  question_type: QuestionType;
  difficulty: QuestionDifficulty;
  document_id: string;
  chunk_id: string;
  chunk_ids: string[];
  page_number: number;
  page_numbers: number[];
  source_text: string;
  start_offset?: number | null;
  end_offset?: number | null;
  qa_predicted_answer?: string | null;
  qa_confidence_score?: number | null;
  validation_status: string;
  rejection_reason?: string | null;
  metadata?: Record<string, any>;
}

export interface RejectedCandidateInfo {
  candidate_id: string;
  question_text: string;
  answer_text: string;
  rejection_reason: string;
  rejection_category: string;
  chunk_id: string;
  chunk_ids: string[];
  page_number: number;
  page_numbers: number[];
  question_type: QuestionType;
}

export interface QuestionGenerationRequest {
  document_id: string;
  count?: number;
  difficulty?: QuestionDifficulty;
  question_type?: QuestionType;
  chapter?: number;
  page_number?: number;
  include_rejected?: boolean;
  ensure_diversity?: boolean;
}

export interface QuestionGenerationResponse {
  document_id: string;
  requested_count: number;
  generated_candidates: number;
  validated_count: number;
  returned_count: number;
  questions: GeneratedQuestion[];
  rejected_candidates?: RejectedCandidateInfo[];
  rejection_summary?: Record<string, number>;
  latency_ms?: number;
}
