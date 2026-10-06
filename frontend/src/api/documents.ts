import { apiClient } from './client';
import {
  DocumentListResponse,
  DocumentUploadResponse,
  PersistedChunk,
  PersistedDocument,
  PersistedPage,
} from '@/types/document';

export const documentsApi = {
  /**
   * Upload a PDF file for asynchronous processing (HTTP 202 Accepted).
   */
  async upload(file: File, documentId?: string): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (documentId && documentId.trim()) {
      formData.append('document_id', documentId.trim());
    }

    return apiClient<DocumentUploadResponse>('/api/v1/documents', {
      method: 'POST',
      body: formData,
      timeoutMs: 300000,
    });
  },

  /**
   * List paginated documents persisted in PostgreSQL.
   */
  async list(skip = 0, limit = 100): Promise<DocumentListResponse> {
    return apiClient<DocumentListResponse>(`/api/v1/documents?skip=${skip}&limit=${limit}`, {
      method: 'GET',
    });
  },

  /**
   * Fetch a single document's metadata and processing status.
   */
  async get(documentId: string): Promise<PersistedDocument> {
    return apiClient<PersistedDocument>(`/api/v1/documents/${encodeURIComponent(documentId)}`, {
      method: 'GET',
    });
  },

  /**
   * Fetch all persisted pages for a document.
   */
  async getPages(documentId: string): Promise<PersistedPage[]> {
    return apiClient<PersistedPage[]>(`/api/v1/documents/${encodeURIComponent(documentId)}/pages`, {
      method: 'GET',
    });
  },

  /**
   * Fetch all persisted chunks for a document.
   */
  async getChunks(documentId: string): Promise<PersistedChunk[]> {
    return apiClient<PersistedChunk[]>(`/api/v1/documents/${encodeURIComponent(documentId)}/chunks`, {
      method: 'GET',
    });
  },

  /**
   * Delete a document and cascade deletion to associated pages and chunks.
   */
  async delete(documentId: string): Promise<{ status: string; document_id: string }> {
    return apiClient<{ status: string; document_id: string }>(
      `/api/v1/documents/${encodeURIComponent(documentId)}`,
      {
        method: 'DELETE',
      }
    );
  },
};

export const documentApi = documentsApi;
