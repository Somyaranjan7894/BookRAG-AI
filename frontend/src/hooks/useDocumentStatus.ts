import { useEffect, useRef, useState, useCallback } from 'react';
import { documentsApi } from '@/api/documents';
import { ApiError } from '@/types/api';
import { PersistedDocument, DocumentStatus, ProcessingStage } from '@/types/document';

interface UseDocumentStatusOptions {
  pollingIntervalMs?: number;
  initialDocument?: PersistedDocument | null;
  onCompleted?: (doc: PersistedDocument) => void;
  onComplete?: (doc: PersistedDocument) => void;
  onFailed?: (doc: PersistedDocument) => void;
}

export function useDocumentStatus(
  documentId: string | null,
  options: UseDocumentStatusOptions = {}
) {
  const {
    pollingIntervalMs = 2500,
    initialDocument = null,
    onCompleted,
    onComplete,
    onFailed,
  } = options;

  const [document, setDocument] = useState<PersistedDocument | null>(initialDocument);
  const [isPolling, setIsPolling] = useState<boolean>(false);
  const [error, setError] = useState<ApiError | null>(null);

  // Keep callback refs stable
  const onCompletedRef = useRef(onComplete || onCompleted);
  onCompletedRef.current = onComplete || onCompleted;
  const onFailedRef = useRef(onFailed);
  onFailedRef.current = onFailed;

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isMountedRef = useRef<boolean>(true);

  const fetchStatus = useCallback(async (id: string): Promise<PersistedDocument | null> => {
    try {
      const doc = await documentsApi.get(id);
      if (!isMountedRef.current) return null;

      setDocument(doc);
      setError(null);

      if (doc.status === 'processed') {
        setIsPolling(false);
        onCompletedRef.current?.(doc);
      } else if (doc.status === 'failed') {
        setIsPolling(false);
        onFailedRef.current?.(doc);
      }

      return doc;
    } catch (err) {
      if (!isMountedRef.current) return null;
      if (err instanceof ApiError) {
        setError(err);
      } else {
        setError(
          new ApiError({
            code: 0,
            message: err instanceof Error ? err.message : 'Failed to fetch document status',
            error_type: 'NETWORK_ERROR',
          })
        );
      }
      return null;
    }
  }, []);

  const refresh = useCallback(() => {
    if (documentId) {
      void fetchStatus(documentId);
    }
  }, [documentId, fetchStatus]);

  useEffect(() => {
    isMountedRef.current = true;

    if (!documentId) {
      setDocument(null);
      setIsPolling(false);
      setError(null);
      return;
    }

    let active = true;

    const poll = async () => {
      if (!active) return;
      setIsPolling(true);

      const doc = await fetchStatus(documentId);
      if (!active || !isMountedRef.current) return;

      // Continue polling only if state is active/transient
      if (doc && (doc.status === 'queued' || doc.status === 'processing')) {
        timerRef.current = setTimeout(poll, pollingIntervalMs);
      } else {
        setIsPolling(false);
      }
    };

    void poll();

    return () => {
      active = false;
      isMountedRef.current = false;
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [documentId, pollingIntervalMs, fetchStatus]);

  const status: DocumentStatus = (document?.status as DocumentStatus) || 'queued';
  const stage: ProcessingStage = document?.processing_stage || 'queued';
  const isComplete = status === 'processed';
  const isFailed = status === 'failed';

  return {
    document,
    status,
    stage,
    isComplete,
    isSuccess: isComplete,
    isFailed,
    isTerminal: isComplete || isFailed,
    isPolling,
    error,
    refresh,
  };
}
