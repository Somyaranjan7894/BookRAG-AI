import React, { useState } from 'react';
import {
  Sparkles,
  HelpCircle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  FileText,
  Clock,
  Layers,
  Copy,
  Check,
} from 'lucide-react';
import { questionsApi } from '@/api/questions';
import {
  QuestionDifficulty,
  QuestionGenerationResponse,
  QuestionType,
} from '@/types/questionGeneration';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorAlert } from '@/components/common/ErrorAlert';

interface QuestionGeneratorProps {
  documentId: string;
  bookTitle?: string;
  onAskQuestion?: (questionText: string) => void;
}

export const QuestionGenerator: React.FC<QuestionGeneratorProps> = ({
  documentId,
  bookTitle,
  onAskQuestion,
}) => {
  const [count, setCount] = useState<number>(5);
  const [targetType, setTargetType] = useState<string>('all');
  const [targetDifficulty, setTargetDifficulty] = useState<string>('all');
  const [includeRejected, setIncludeRejected] = useState<boolean>(false);
  const [ensureDiversity, setEnsureDiversity] = useState<boolean>(true);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<QuestionGenerationResponse | null>(null);

  const [expandedEvidence, setExpandedEvidence] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await questionsApi.generateQuestions({
        document_id: documentId,
        count,
        question_type: targetType !== 'all' ? (targetType as QuestionType) : undefined,
        difficulty: targetDifficulty !== 'all' ? (targetDifficulty as QuestionDifficulty) : undefined,
        include_rejected: includeRejected,
        ensure_diversity: ensureDiversity,
      });
      setResponse(res);
    } catch (err: any) {
      setError(err?.message || 'Failed to generate questions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleEvidence = (qid: string) => {
    setExpandedEvidence((prev) => ({ ...prev, [qid]: !prev[qid] }));
  };

  const handleCopyQuestion = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getTypeBadgeClass = (type: string) => {
    switch (type) {
      case 'definition':
        return 'bg-blue-50 text-blue-700 border-blue-200';
      case 'comparison':
        return 'bg-purple-50 text-purple-700 border-purple-200';
      case 'numerical':
      case 'numerical_fact':
        return 'bg-emerald-50 text-emerald-700 border-emerald-200';
      case 'explanation':
      case 'why':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'reasoning':
        return 'bg-rose-50 text-rose-700 border-rose-200';
      case 'multi_page_synthesis':
        return 'bg-cyan-50 text-cyan-700 border-cyan-200';
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  const getDifficultyBadgeClass = (diff: string) => {
    switch (diff) {
      case 'easy':
        return 'bg-emerald-50 text-emerald-700 border-emerald-200';
      case 'medium':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'hard':
        return 'bg-rose-50 text-rose-700 border-rose-200';
      default:
        return 'bg-slate-100 text-slate-700 border-slate-200';
    }
  };

  return (
    <div className="space-y-6">
      {/* Controls Card */}
      <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-5 border-b border-slate-100">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-indigo-600" />
              Generate Reading Comprehension Questions
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Extracts factual evidence spans from {bookTitle ? `"${bookTitle}"` : 'this book'}, conditions local T5 question generation, and strictly verifies answerability with extractive QA.
            </p>
          </div>

          <button
            type="button"
            onClick={handleGenerate}
            disabled={isLoading}
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-xs text-white bg-indigo-600 hover:bg-indigo-700 transition shadow-sm disabled:opacity-60 disabled:cursor-not-allowed shrink-0 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {isLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <Sparkles className="w-4 h-4" />
            )}
            <span>{isLoading ? 'Generating & Validating...' : 'Generate Questions'}</span>
          </button>
        </div>

        {/* Configuration Filters */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 pt-5">
          {/* Count */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Question Count
            </label>
            <select
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              disabled={isLoading}
              className="w-full text-xs rounded-xl border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-50"
            >
              <option value={3}>3 Questions</option>
              <option value={5}>5 Questions (Default)</option>
              <option value={8}>8 Questions</option>
              <option value={10}>10 Questions</option>
            </select>
          </div>

          {/* Question Type Filter */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Question Type
            </label>
            <select
              value={targetType}
              onChange={(e) => setTargetType(e.target.value)}
              disabled={isLoading}
              className="w-full text-xs rounded-xl border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-50"
            >
              <option value="all">All Types (Balanced)</option>
              <option value="direct_fact">Direct Fact</option>
              <option value="definition">Definition</option>
              <option value="explanation">Explanation</option>
              <option value="comparison">Comparison</option>
              <option value="numerical_fact">Numerical Fact</option>
              <option value="reasoning">Reasoning</option>
              <option value="multi_page_synthesis">Multi-Page Synthesis</option>
            </select>
          </div>

          {/* Difficulty Filter */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1.5">
              Difficulty
            </label>
            <select
              value={targetDifficulty}
              onChange={(e) => setTargetDifficulty(e.target.value)}
              disabled={isLoading}
              className="w-full text-xs rounded-xl border border-slate-300 bg-white px-3 py-2 text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-slate-50"
            >
              <option value="all">All Difficulties</option>
              <option value="easy">Easy (Single fact)</option>
              <option value="medium">Medium (Detailed fact)</option>
              <option value="hard">Hard (Reasoning / Synthesis)</option>
            </select>
          </div>

          {/* Diversity & Diagnostics */}
          <div className="flex flex-col justify-end space-y-2">
            <label className="inline-flex items-center gap-2 cursor-pointer text-xs text-slate-700 select-none">
              <input
                type="checkbox"
                checked={ensureDiversity}
                onChange={(e) => setEnsureDiversity(e.target.checked)}
                disabled={isLoading}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-3.5 h-3.5"
              />
              <span className="font-medium">Ensure Type Diversity</span>
            </label>
            <label className="inline-flex items-center gap-2 cursor-pointer text-xs text-slate-700 select-none">
              <input
                type="checkbox"
                checked={includeRejected}
                onChange={(e) => setIncludeRejected(e.target.checked)}
                disabled={isLoading}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 w-3.5 h-3.5"
              />
              <span className="font-medium">Show Diagnostics</span>
            </label>
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <ErrorAlert
          title="Question Generation Failed"
          error={error}
          onRetry={handleGenerate}
        />
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="p-8 rounded-2xl border border-indigo-100 bg-white shadow-2xs text-center space-y-3">
          <LoadingSpinner size="lg" label="Generating evidence-grounded questions..." />
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Extracting candidate answer spans from book chunks, generating conditioned questions with T5, deduplicating near-identical candidates, and verifying answerability with RoBERTa extractive QA.
          </p>
        </div>
      )}

      {/* Results Section */}
      {!isLoading && response && (
        <div className="space-y-6">
          {/* Metric KPI Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
              <span className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Candidates
              </span>
              <span className="text-lg font-bold text-slate-900 mt-1 block">
                {response.generated_candidates}
              </span>
              <span className="text-[10px] text-slate-500">Raw generated</span>
            </div>

            <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/50 shadow-2xs">
              <span className="block text-[11px] font-semibold text-emerald-600 uppercase tracking-wider">
                QA Validated
              </span>
              <span className="text-lg font-bold text-emerald-900 mt-1 block">
                {response.validated_count}
              </span>
              <span className="text-[10px] text-emerald-700">Passed answerability</span>
            </div>

            <div className="p-4 rounded-xl border border-indigo-200 bg-indigo-50/50 shadow-2xs">
              <span className="block text-[11px] font-semibold text-indigo-600 uppercase tracking-wider">
                Returned
              </span>
              <span className="text-lg font-bold text-indigo-900 mt-1 block">
                {response.returned_count}
              </span>
              <span className="text-[10px] text-indigo-700">Ready for review</span>
            </div>

            <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
              <span className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                Latency
              </span>
              <span className="text-lg font-bold text-slate-900 mt-1 block flex items-center gap-1">
                <Clock className="w-4 h-4 text-slate-400" />
                {response.latency_ms ? `${Math.round(response.latency_ms)}ms` : 'N/A'}
              </span>
              <span className="text-[10px] text-slate-500">Generation + Validation</span>
            </div>
          </div>

          {/* Validated Questions List */}
          {response.questions.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                  Validated & Grounded Questions ({response.questions.length})
                </h4>
                <span className="text-xs text-slate-500">
                  Strictly grounded in book evidence
                </span>
              </div>

              {response.questions.map((q, idx) => {
                const qid = q.question_id || `q_${idx}`;
                const isExpanded = !!expandedEvidence[qid];
                const isCopied = copiedId === qid;

                return (
                  <div
                    key={qid}
                    className="rounded-2xl border border-slate-200/90 bg-white p-5 shadow-2xs transition hover:border-slate-300"
                  >
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3">
                      {/* Badges */}
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-xs font-bold text-slate-400">
                          #{idx + 1}
                        </span>

                        <span
                          className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border capitalize ${getTypeBadgeClass(
                            q.question_type
                          )}`}
                        >
                          {q.question_type.replace('_', ' ')}
                        </span>

                        <span
                          className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border uppercase tracking-wider ${getDifficultyBadgeClass(
                            q.difficulty
                          )}`}
                        >
                          {q.difficulty}
                        </span>

                        {q.page_numbers && q.page_numbers.length > 1 ? (
                          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200 flex items-center gap-1">
                            <Layers className="w-3 h-3 text-slate-500" />
                            Pages {q.page_numbers.join(', ')}
                          </span>
                        ) : (
                          <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200 flex items-center gap-1">
                            <FileText className="w-3 h-3 text-slate-500" />
                            Page {q.page_number}
                          </span>
                        )}

                        <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                          Validated
                        </span>
                      </div>

                      {/* Action buttons */}
                      <div className="flex items-center gap-2 shrink-0">
                        <button
                          type="button"
                          onClick={() => handleCopyQuestion(q.question, qid)}
                          className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-600 hover:text-slate-900 bg-slate-50 hover:bg-slate-100 rounded-lg border border-slate-200 transition"
                          title="Copy question text"
                        >
                          {isCopied ? (
                            <>
                              <Check className="w-3.5 h-3.5 text-emerald-600" />
                              <span className="text-emerald-700">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5 text-slate-500" />
                              <span>Copy</span>
                            </>
                          )}
                        </button>

                        {onAskQuestion && (
                          <button
                            type="button"
                            onClick={() => onAskQuestion(q.question)}
                            className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-semibold text-indigo-700 hover:text-indigo-900 bg-indigo-50 hover:bg-indigo-100 rounded-lg border border-indigo-200 transition"
                          >
                            <span>Query Book</span>
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Question Text */}
                    <h5 className="text-base font-semibold text-slate-900 leading-snug">
                      {q.question}
                    </h5>

                    {/* Verified Answer Span */}
                    <div className="mt-3 p-3 rounded-xl bg-slate-50 border border-slate-200/80 flex items-start gap-2">
                      <span className="text-xs font-bold text-slate-500 uppercase tracking-wider shrink-0 mt-0.5">
                        Answer:
                      </span>
                      <p className="text-xs font-medium text-slate-800 leading-relaxed">
                        {q.answer}
                      </p>
                    </div>

                    {/* Toggle Evidence */}
                    <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
                      <button
                        type="button"
                        onClick={() => toggleEvidence(qid)}
                        className="inline-flex items-center gap-1.5 font-medium text-indigo-600 hover:text-indigo-800 transition"
                      >
                        {isExpanded ? (
                          <>
                            <ChevronUp className="w-3.5 h-3.5" />
                            <span>Hide Source Book Evidence</span>
                          </>
                        ) : (
                          <>
                            <ChevronDown className="w-3.5 h-3.5" />
                            <span>Inspect Source Book Evidence</span>
                          </>
                        )}
                      </button>

                      <span className="text-[11px] font-mono text-slate-400">
                        Chunk: {q.chunk_id}
                      </span>
                    </div>

                    {/* Expanded Source Evidence Passage */}
                    {isExpanded && (
                      <div className="mt-3 p-4 rounded-xl bg-slate-900 text-slate-100 text-xs font-mono leading-relaxed overflow-x-auto border border-slate-800">
                        <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-800 text-[11px] text-slate-400">
                          <span>
                            Source Evidence Passage (Page {q.page_numbers ? q.page_numbers.join(', ') : q.page_number})
                          </span>
                          {q.qa_confidence_score && (
                            <span>QA Score: {q.qa_confidence_score.toFixed(3)}</span>
                          )}
                        </div>
                        <p className="whitespace-pre-wrap font-sans text-xs text-slate-300">
                          {q.source_text}
                        </p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="p-8 rounded-2xl border border-dashed border-slate-300 bg-white/60 text-center">
              <AlertCircle className="w-8 h-8 text-amber-500 mx-auto mb-2" />
              <h4 className="text-sm font-semibold text-slate-800">
                No questions met strict validation criteria
              </h4>
              <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
                The generator extracted candidates, but none passed the extractive QA and quality thresholds. Try requesting a different question type or adjusting filters.
              </p>
            </div>
          )}

          {/* Diagnostic Rejection Details (if requested) */}
          {includeRejected && response.rejected_candidates && response.rejected_candidates.length > 0 && (
            <div className="mt-8 pt-6 border-t border-slate-200 space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-bold text-slate-700 flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-600" />
                  Diagnostic Rejections ({response.rejected_candidates.length})
                </h4>
                {response.rejection_summary && (
                  <div className="flex flex-wrap items-center gap-1.5 text-[11px]">
                    {Object.entries(response.rejection_summary).map(([cat, cnt]) => (
                      <span
                        key={cat}
                        className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 border border-slate-200"
                      >
                        {cat}: {cnt}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="space-y-2.5">
                {response.rejected_candidates.map((rej, rIdx) => (
                  <div
                    key={rej.candidate_id || `rej_${rIdx}`}
                    className="p-3.5 rounded-xl border border-amber-200/70 bg-amber-50/40 text-xs space-y-1"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-amber-900">
                        {rej.question_text}
                      </span>
                      <span className="px-2 py-0.5 rounded-md bg-amber-100 text-amber-800 font-mono text-[10px] uppercase">
                        {rej.rejection_category}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-slate-500 text-[11px]">
                      <span>Target: {rej.answer_text}</span>
                      <span className="text-amber-800 italic">
                        Reason: {rej.rejection_reason}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Initial Empty State before generation */}
      {!isLoading && !response && !error && (
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white/60 p-10 text-center">
          <div className="w-12 h-12 mx-auto rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3">
            <HelpCircle className="w-6 h-6" />
          </div>
          <h4 className="text-sm font-semibold text-slate-900">
            No questions generated yet
          </h4>
          <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
            Click <strong>"Generate Questions"</strong> above to extract evidence spans from {bookTitle ? `"${bookTitle}"` : 'this book'} and generate verified questions with full page citations.
          </p>
        </div>
      )}
    </div>
  );
};
