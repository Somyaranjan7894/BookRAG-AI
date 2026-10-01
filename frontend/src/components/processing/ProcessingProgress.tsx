import React from 'react';
import { CheckCircle2, AlertCircle, ArrowRight, BookOpen } from 'lucide-react';
import { useDocumentStatus } from '@/hooks/useDocumentStatus';
import { Link } from '@/router';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';

export interface ProcessingProgressProps {
  documentId: string;
  filename?: string;
  pollingIntervalMs?: number;
  onFinished?: () => void;
  className?: string;
}

export const ProcessingProgress: React.FC<ProcessingProgressProps> = ({
  documentId,
  filename,
  pollingIntervalMs = 1500,
  onFinished,
  className = '',
}) => {
  const { document, error, isSuccess, isFailed } = useDocumentStatus(
    documentId,
    {
      pollingIntervalMs,
      onComplete: () => {
        if (onFinished) onFinished();
      },
    }
  );

  const stage = (document?.processing_stage || '').toLowerCase();
  const status = (document?.status || 'queued').toLowerCase();

  // Map backend stages to linear progression percentage
  const getStagePercentage = (): number => {
    if (isSuccess || status === 'processed' || stage === 'completed') return 100;
    if (isFailed || status === 'failed') return 100;
    if (stage === 'indexing' || stage === 'embedding') return 80;
    if (stage === 'persistence' || stage === 'chunking') return 55;
    if (stage === 'parsing' || stage === 'ingestion' || stage === 'validation') return 30;
    return 10;
  };

  const getStageDescription = (): string => {
    if (isSuccess) return 'Document fully indexed and ready for grounded retrieval!';
    if (isFailed) return document?.error_message || 'Document processing encountered an error.';
    if (stage === 'indexing' || stage === 'embedding') {
      return 'Generating dense embeddings & updating vector indices (FAISS/pgvector)...';
    }
    if (stage === 'persistence' || stage === 'chunking') {
      return 'Creating semantic chunks and persisting page hierarchies to PostgreSQL...';
    }
    if (stage === 'parsing' || stage === 'ingestion' || stage === 'validation') {
      return 'Extracting text and structure from PDF pages...';
    }
    return 'Task queued in Celery broker. Waiting for worker assignment...';
  };

  const percentage = getStagePercentage();

  return (
    <div
      role="region"
      aria-label="Document Processing Status"
      className={`rounded-2xl border p-5 transition-all shadow-sm ${
        isSuccess
          ? 'border-emerald-200 bg-emerald-50/50'
          : isFailed
          ? 'border-rose-200 bg-rose-50/50'
          : 'border-indigo-100 bg-gradient-to-br from-white via-indigo-50/30 to-purple-50/20'
      } ${className}`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center ${
              isSuccess
                ? 'bg-emerald-100 text-emerald-700'
                : isFailed
                ? 'bg-rose-100 text-rose-700'
                : 'bg-indigo-100 text-indigo-700'
            }`}
          >
            {isSuccess ? (
              <CheckCircle2 className="w-5 h-5" />
            ) : isFailed ? (
              <AlertCircle className="w-5 h-5" />
            ) : (
              <LoadingSpinner size="sm" />
            )}
          </div>
          <div>
            <h4 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
              <span>{filename || document?.filename || documentId}</span>
              <span className="text-[11px] font-mono text-slate-500 font-normal">
                ({documentId.slice(0, 8)}...)
              </span>
            </h4>
            <p className="text-xs text-slate-600 mt-0.5">{getStageDescription()}</p>
          </div>
        </div>

        {/* Action button if ready */}
        {isSuccess && (
          <Link
            to={`/books/${documentId}`}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-600 text-white hover:bg-emerald-700 transition shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
          >
            <BookOpen className="w-3.5 h-3.5" />
            Open Book
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        )}
      </div>

      {/* Progress bar */}
      <div className="mt-4">
        <div className="w-full bg-slate-200/80 rounded-full h-2 overflow-hidden">
          <div
            className={`h-2 rounded-full transition-all duration-500 ${
              isSuccess
                ? 'bg-emerald-500'
                : isFailed
                ? 'bg-rose-500'
                : 'bg-gradient-to-r from-indigo-500 to-purple-600 animate-pulse'
            }`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>

      {/* Stages indicator pills */}
      <div className="mt-3 flex items-center justify-between text-[11px] font-medium text-slate-500">
        <span
          className={
            percentage >= 10
              ? 'text-indigo-600 font-semibold'
              : 'text-slate-400'
          }
        >
          1. Queued
        </span>
        <span
          className={
            percentage >= 30
              ? 'text-indigo-600 font-semibold'
              : 'text-slate-400'
          }
        >
          2. Parse Pages
        </span>
        <span
          className={
            percentage >= 55
              ? 'text-indigo-600 font-semibold'
              : 'text-slate-400'
          }
        >
          3. Chunk & Persist
        </span>
        <span
          className={
            percentage >= 80
              ? 'text-indigo-600 font-semibold'
              : 'text-slate-400'
          }
        >
          4. Vector Indexing
        </span>
        <span
          className={
            isSuccess
              ? 'text-emerald-600 font-semibold'
              : 'text-slate-400'
          }
        >
          5. Ready
        </span>
      </div>

      {/* Error display if failed */}
      {isFailed && (
        <div className="mt-3 text-xs text-rose-700 bg-rose-100/60 p-2.5 rounded-lg border border-rose-200">
          <span className="font-semibold">Processing Failed: </span>
          {document?.error_message || error?.message || 'Check backend Celery logs for details.'}
        </div>
      )}
    </div>
  );
};
