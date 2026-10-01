import { useState, useEffect, useCallback } from 'react';
import { documentApi } from '@/api/documents';
import { PersistedDocument, PersistedPage, PersistedChunk } from '@/types/document';
import { ApiError } from '@/types/api';

export interface UseDocumentReturn {
  document: PersistedDocument | null;
  pages: PersistedPage[];
  chunks: PersistedChunk[];
  isLoading: boolean;
  error: ApiError | null;
  refetch: () => Promise<void>;
  deleteDocument: () => Promise<boolean>;
}

export function useDocument(documentId: string | null | undefined): UseDocumentReturn {
  const [document, setDocument] = useState<PersistedDocument | null>(null);
  const [pages, setPages] = useState<PersistedPage[]>([]);
  const [chunks, setChunks] = useState<PersistedChunk[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<ApiError | null>(null);

  const fetchDocumentData = useCallback(async () => {
    if (!documentId) {
      setDocument(null);
      setPages([]);
      setChunks([]);
      setError(null);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const doc = await documentApi.get(documentId);
      setDocument(doc);

      // Attempt to load pages and chunks if document is processed
      if (doc.status === 'processed') {
        try {
          const [pagesList, chunksList] = await Promise.all([
            documentApi.getPages(documentId),
            documentApi.getChunks(documentId),
          ]);
          setPages(pagesList || []);
          setChunks(chunksList || []);
        } catch {
          // Non-critical: failure to fetch chunks or pages shouldn't block document overview
        }
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err);
      } else {
        setError(
          new ApiError({
            code: 500,
            message: err instanceof Error ? err.message : 'Failed to fetch document',
            error_type: 'FETCH_DOCUMENT_FAILED',
          })
        );
      }
    } finally {
      setIsLoading(false);
    }
  }, [documentId]);

  useEffect(() => {
    fetchDocumentData();
  }, [fetchDocumentData]);

  const deleteDocument = useCallback(async (): Promise<boolean> => {
    if (!documentId) return false;
    try {
      await documentApi.delete(documentId);
      setDocument(null);
      return true;
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err);
      }
      return false;
    }
  }, [documentId]);

  return {
    document,
    pages,
    chunks,
    isLoading,
    error,
    refetch: fetchDocumentData,
    deleteDocument,
  };
}
