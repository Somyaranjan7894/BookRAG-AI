/**
 * Question Answering, Grounded Synthesis, and Citation types matching FastAPI schemas.
 */

export interface Citation {
  citation_id: string;
  document_id: string;
  chunk_id: string;
  page_number: number;
  chunk_index: number;
  source_text: string;
  similarity_score?: number | null;
  reranker_score?: number | null;
  evidence_rank?: number | null;
}

export interface ClaimCitationRef {
  citation_id: string;
  document_id?: string | null;
  chunk_id?: string | null;
  page_number?: number | null;
  chunk_index?: number | null;
  source_text?: string | null;
  relation: 'supports' | 'contradicts';
}

export interface ClaimEvidenceProvenance {
  chunk_id: string;
  document_id: string;
  page_number: number;
  chunk_index: number;
  source_text: string;
  rank?: number | null;
  similarity_score?: number | null;
  reranker_score?: number | null;
  nli_score?: number | null;
}

export type ClaimStatus = 'entailed' | 'contradicted' | 'unsupported' | 'conflicted';

export interface ClaimResult {
  claim_index: number;
  claim_text: string;
  status: ClaimStatus;
  grounding_status?: string | null;
  entailment_score: number;
  contradiction_score: number;
  neutral_score: number;
  supporting_evidence?: ClaimEvidenceProvenance | null;
  supporting_evidences: ClaimEvidenceProvenance[];
  contradicting_evidence: ClaimEvidenceProvenance[];
  citations: ClaimCitationRef[];
  contradicting_citations: ClaimCitationRef[];
}

export interface GenerationEvidenceItem {
  rank: number;
  chunk_id: string;
  document_id: string;
  page_number: number;
  chunk_index: number;
  source_text: string;
  similarity_score?: number | null;
  reranker_score?: number | null;
}

export interface GroundedAnswerRequest {
  query: string;
  document_id?: string | null;
  top_k?: number;
  candidate_k?: number;
  enable_reranking?: boolean;
}

export interface GroundedAnswerResponse {
  query: string;
  answer: string | null;
  answerable: boolean;
  grounded: boolean;
  groundedness_score: number;
  grounding_status: string;
  claims: ClaimResult[];
  evidence: GenerationEvidenceItem[];
  citations: Citation[];
  reason?: string | null;
  model_name?: string | null;
  grounding_model_name?: string | null;
}

export interface QARequest {
  query: string;
  document_id?: string | null;
  top_k?: number;
  candidate_k?: number;
  enable_reranking?: boolean;
}

export interface QAResponse {
  query: string;
  answer: string | null;
  answerable: boolean;
  qa_score?: number | null;
  no_answer_score?: number | null;
  document_id?: string | null;
  chunk_id?: string | null;
  page_number?: number | null;
  chunk_index?: number | null;
  answer_start?: number | null;
  answer_end?: number | null;
  source_text?: string | null;
  evidence_rank?: number | null;
  similarity_score?: number | null;
  reranker_score?: number | null;
  total_evidence_evaluated: number;
}
