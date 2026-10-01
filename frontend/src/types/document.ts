/**
 * Document, page, and chunk domain models matching backend FastAPI contracts.
 */

export type DocumentStatus = 'queued' | 'processing' | 'processed' | 'failed';

export type ProcessingStage =
  | 'queued'
  | 'validation'
  | 'ingestion'
  | 'parsing'
  | 'chunking'
  | 'persistence'
  | 'embedding'
  | 'indexing'
  | 'completed'
  | 'database'
  | 'queue_failed'
  | 'failed'
  | string;

export interface PersistedDocument {
  document_id: str;
  filename: str;
  title: string | null;
  author: string | null;
  page_count: number;
  status: DocumentStatus | string;
  processing_stage: ProcessingStage | null;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
}

// Helper alias
type str = string;

export interface DocumentUploadResponse {
  document_id: string;
  task_id: string | null;
  status: DocumentStatus | string;
  message: string;
}

export interface DocumentListResponse {
  total: number;
  documents: PersistedDocument[];
}

export interface PersistedPage {
  page_id: string;
  document_id: string;
  page_number: number;
  text: string;
  char_count: number;
  word_count: number;
}

export interface PersistedChunk {
  chunk_id: string;
  document_id: string;
  page_id: string;
  page_number: number;
  chunk_index: number;
  text: string;
  char_count: number;
  word_count: number;
}
