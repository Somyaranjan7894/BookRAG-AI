import React, { useState } from 'react';
import { ArrowLeft, BookOpen, FileText, Sparkles, HelpCircle, Layers, Info } from 'lucide-react';
import { useRouter, Link } from '@/router';
import { useDocument } from '@/hooks/useDocument';
import { useAskQuestion } from '@/hooks/useAskQuestion';
import { QuestionInput } from '@/components/qa/QuestionInput';
import { AnswerDisplay } from '@/components/qa/AnswerDisplay';
import { MatchingBoard } from '@/components/matching/MatchingBoard';
import { ProcessingStatusBadge } from '@/components/processing/ProcessingStatusBadge';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorAlert } from '@/components/common/ErrorAlert';

import { QuestionGenerator } from '@/components/questions/QuestionGenerator';

export const BookDetail: React.FC = () => {
  const { params } = useRouter();
  const documentId = params.id;

  const { document, chunks, isLoading: isDocLoading, error: docError, refetch } = useDocument(documentId);
  const {
    query,
    mode,
    isLoading: isQuestionLoading,
    error: qaError,
    groundedResult,
    extractiveResult,
    searchResult,
    setMode,
    ask,
  } = useAskQuestion({ documentId });

  const [activeTab, setActiveTab] = useState<'grounded' | 'extractive' | 'matching' | 'qgen'>('grounded');
  const [hasAsked, setHasAsked] = useState<boolean>(false);

  const handleAsk = async (questionText: string) => {
    setHasAsked(true);
    await ask(questionText, mode);
  };

  if (isDocLoading && !document) {
    return (
      <div className="py-24 flex justify-center items-center">
        <LoadingSpinner size="lg" label="Loading book details..." />
      </div>
    );
  }

  if (docError) {
    return (
      <div className="space-y-6">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Library
        </Link>
        <ErrorAlert error={docError} onRetry={refetch} />
      </div>
    );
  }

  if (!document) {
    return (
      <div className="space-y-6">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-slate-900"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Library
        </Link>
        <div className="p-8 text-center bg-white rounded-2xl border border-slate-200">
          <p className="text-slate-600 text-sm">Document not found.</p>
        </div>
      </div>
    );
  }

  const isProcessed = document.status === 'processed';

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Navigation Header */}
      <div className="flex items-center justify-between">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-xs font-semibold text-slate-600 hover:text-indigo-600 transition p-1 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Library</span>
        </Link>

        <ProcessingStatusBadge status={document.status} />
      </div>

      {/* Book Metadata Header Card */}
      <div className="rounded-3xl border border-slate-200/90 bg-white p-6 sm:p-8 shadow-sm">
        <div className="flex flex-col sm:flex-row items-start gap-5">
          <div className="w-16 h-16 rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center shrink-0 shadow-2xs">
            <BookOpen className="w-8 h-8" />
          </div>

          <div className="flex-1 min-w-0">
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              {document.title || document.filename}
            </h1>

            {document.author && (
              <p className="text-sm font-medium text-slate-500 mt-1">Author: {document.author}</p>
            )}

            <div className="mt-4 flex flex-wrap items-center gap-4 text-xs text-slate-500">
              <span className="flex items-center gap-1.5 font-medium">
                <FileText className="w-3.5 h-3.5 text-slate-400" />
                {document.page_count} Pages
              </span>
              <span className="text-slate-300">•</span>
              <span className="flex items-center gap-1.5 font-medium">
                <Layers className="w-3.5 h-3.5 text-slate-400" />
                {chunks.length > 0 ? `${chunks.length}+ Chunks Indexed` : 'Indexed in pgvector'}
              </span>
              <span className="text-slate-300">•</span>
              <span className="font-mono text-[11px] text-slate-400 truncate max-w-xs">
                ID: {document.document_id}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Q&A / Exploration Section */}
      {isProcessed ? (
        <div className="space-y-6">
          {/* Controls & Mode Selection */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-indigo-600" />
                {activeTab === 'qgen' ? 'Question Generator' : 'Ask This Book'}
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                {activeTab === 'qgen'
                  ? 'Generate controlled, evidence-grounded reading comprehension questions from this book'
                  : 'Query full document contents with verifiable citation provenance'}
              </p>
            </div>

            {/* QA & Tool Mode Switcher */}
            <div className="flex flex-wrap items-center p-1 rounded-xl bg-slate-100 border border-slate-200 text-xs max-w-full">
              <button
                type="button"
                onClick={() => {
                  setActiveTab('grounded');
                  setMode('grounded');
                }}
                className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                  activeTab === 'grounded'
                    ? 'bg-white text-indigo-700 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Grounded Synthesis (NLI)
              </button>
              <button
                type="button"
                onClick={() => {
                  setActiveTab('extractive');
                  setMode('extractive');
                }}
                className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                  activeTab === 'extractive'
                    ? 'bg-white text-indigo-700 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Extractive QA (RoBERTa)
              </button>
              <button
                type="button"
                onClick={() => {
                  setActiveTab('matching');
                  setMode('matching');
                }}
                className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                  activeTab === 'matching'
                    ? 'bg-white text-indigo-700 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Matching Board
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('qgen')}
                className={`px-3 py-1.5 rounded-lg font-semibold transition ${
                  activeTab === 'qgen'
                    ? 'bg-white text-indigo-700 shadow-2xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Generate Questions
              </button>
            </div>
          </div>

          {/* Tab Content */}
          {activeTab === 'qgen' ? (
            <QuestionGenerator
              documentId={document.document_id}
              bookTitle={document.title || document.filename}
              onAskQuestion={(qText) => {
                setActiveTab('grounded');
                setMode('grounded');
                handleAsk(qText);
              }}
            />
          ) : (
            <>
              {/* Question Input */}
              <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm">
                <QuestionInput
                  onSubmit={handleAsk}
                  isLoading={isQuestionLoading}
                  placeholder={
                    mode === 'matching'
                      ? `Search semantic candidates & inspect reranking for "${document.title || document.filename}"...`
                      : `Ask a question about "${document.title || document.filename}"...`
                  }
                />
              </div>

              {/* Error Banner */}
              {qaError && (
                <ErrorAlert
                  error={qaError}
                  title={mode === 'matching' ? 'Failed to Search Book' : 'Failed to Answer Question'}
                  onRetry={() => query && handleAsk(query)}
                />
              )}

              {/* Active Question / Search Loading State */}
              {isQuestionLoading && (
                <div className="p-8 rounded-2xl border border-indigo-100 bg-white shadow-2xs text-center">
                  <LoadingSpinner
                    size="lg"
                    label={
                      mode === 'matching'
                        ? 'Retrieving candidate pool & running Cross-Encoder reranking...'
                        : 'Searching semantic chunks & verifying claims...'
                    }
                  />
                  <p className="mt-3 text-xs text-slate-500">
                    {mode === 'matching'
                      ? 'Evaluating dense vector similarity (FAISS/pgvector) and computing full cross-attention transformer scores...'
                      : 'Evaluating vector similarity, applying cross-encoder reranking, and running DeBERTa NLI grounding...'}
                  </p>
                </div>
              )}

              {/* Standalone Matching Board Display */}
              {mode === 'matching' && searchResult && !isQuestionLoading && (
                <MatchingBoard
                  query={query}
                  results={searchResult.results}
                  candidateCount={searchResult.candidate_count}
                  rerankingApplied={searchResult.reranking_applied}
                />
              )}

              {/* Answer Display (Grounded / Extractive) */}
              {mode !== 'matching' && (groundedResult || extractiveResult) && !isQuestionLoading && (
                <AnswerDisplay
                  query={query}
                  groundedResult={groundedResult}
                  extractiveResult={extractiveResult}
                />
              )}

              {/* Empty State before any question or search */}
              {!hasAsked && !groundedResult && !extractiveResult && !searchResult && !isQuestionLoading && (
                <div className="rounded-2xl border border-dashed border-slate-300 bg-white/60 p-10 text-center">
                  <div className="w-12 h-12 mx-auto rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3">
                    <HelpCircle className="w-6 h-6" />
                  </div>
                  <h3 className="text-sm font-semibold text-slate-900">
                    {mode === 'matching' ? 'No search executed yet' : 'No questions asked yet'}
                  </h3>
                  <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
                    {mode === 'matching'
                      ? 'Submit a search query above to inspect candidate retrieval pool sizes, Cross-Encoder reranking movements, and page provenance.'
                      : 'Type a question above to retrieve context-backed answers, claim verification scores, and page citations.'}
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      ) : (
        <div className="p-8 rounded-2xl border border-amber-200 bg-amber-50/80 text-amber-900 flex items-start gap-4">
          <Info className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <h3 className="font-semibold text-base">Book is not ready for querying</h3>
            <p className="text-xs text-amber-800 mt-1 leading-relaxed">
              This document is currently in state: <strong>{document.status}</strong>.
              Questions can only be asked once parsing, chunking, and embedding indexing are complete.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
