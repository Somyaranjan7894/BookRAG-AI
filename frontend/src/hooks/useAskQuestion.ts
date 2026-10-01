import { useState, useCallback } from 'react';
import { qaApi } from '@/api/qa';
import { searchApi } from '@/api/search';
import { GroundedAnswerResponse, QAResponse } from '@/types/qa';
import { SearchResponse } from '@/types/search';
import { ApiError } from '@/types/api';

export type QAMode = 'grounded' | 'extractive' | 'matching';

export interface UseAskQuestionOptions {
  documentId?: string | null;
  defaultMode?: QAMode;
  topK?: number;
  candidateK?: number;
  enableReranking?: boolean;
}

export interface UseAskQuestionReturn {
  query: string;
  mode: QAMode;
  isLoading: boolean;
  error: ApiError | null;
  groundedResult: GroundedAnswerResponse | null;
  extractiveResult: QAResponse | null;
  searchResult: SearchResponse | null;
  setMode: (mode: QAMode) => void;
  ask: (questionText: string, overrideMode?: QAMode) => Promise<boolean>;
  clear: () => void;
}

export function useAskQuestion(options: UseAskQuestionOptions = {}): UseAskQuestionReturn {
  const {
    documentId = null,
    defaultMode = 'grounded',
    topK = 5,
    candidateK = 20,
    enableReranking = true,
  } = options;

  const [query, setQuery] = useState<string>('');
  const [mode, setMode] = useState<QAMode>(defaultMode);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [groundedResult, setGroundedResult] = useState<GroundedAnswerResponse | null>(null);
  const [extractiveResult, setExtractiveResult] = useState<QAResponse | null>(null);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);

  const clear = useCallback(() => {
    setQuery('');
    setError(null);
    setGroundedResult(null);
    setExtractiveResult(null);
    setSearchResult(null);
  }, []);

  const ask = useCallback(
    async (questionText: string, overrideMode?: QAMode): Promise<boolean> => {
      const trimmed = questionText.trim();
      if (!trimmed) {
        setError(
          new ApiError({
            code: 400,
            message: 'Please enter a valid question before asking.',
            error_type: 'EMPTY_QUESTION',
          })
        );
        return false;
      }

      if (isLoading) {
        return false; // prevent duplicate submission
      }

      const activeMode = overrideMode || mode;
      setQuery(trimmed);
      setIsLoading(true);
      setError(null);

      try {
        if (activeMode === 'grounded') {
          const res = await qaApi.askGroundedQuestion({
            query: trimmed,
            document_id: documentId || undefined,
            top_k: topK,
            candidate_k: candidateK,
            enable_reranking: enableReranking,
          });
          setGroundedResult(res);
          setExtractiveResult(null);
          setSearchResult(null);
        } else if (activeMode === 'extractive') {
          const res = await qaApi.askExtractiveQuestion({
            query: trimmed,
            document_id: documentId || undefined,
            top_k: topK,
            candidate_k: candidateK,
            enable_reranking: enableReranking,
          });
          setExtractiveResult(res);
          setGroundedResult(null);
          setSearchResult(null);
        } else {
          // Matching Board transparency search mode
          const res = await searchApi.search({
            query: trimmed,
            document_id: documentId || undefined,
            top_k: topK,
            candidate_k: candidateK,
            enable_reranking: enableReranking,
          });
          setSearchResult(res);
          setGroundedResult(null);
          setExtractiveResult(null);
        }
        return true;
      } catch (err) {
        if (err instanceof ApiError) {
          setError(err);
        } else {
          setError(
            new ApiError({
              code: 500,
              message: err instanceof Error ? err.message : 'Failed to retrieve answer.',
              error_type: 'QUESTION_FAILED',
            })
          );
        }
        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [documentId, isLoading, mode, topK, candidateK, enableReranking]
  );

  return {
    query,
    mode,
    isLoading,
    error,
    groundedResult,
    extractiveResult,
    searchResult,
    setMode,
    ask,
    clear,
  };
}
