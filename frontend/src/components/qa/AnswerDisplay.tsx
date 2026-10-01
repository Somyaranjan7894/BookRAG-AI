import React from 'react';
import { Sparkles, HelpCircle, AlertOctagon, BookOpen } from 'lucide-react';
import { GroundedAnswerResponse, QAResponse } from '@/types/qa';
import { GroundingBadge } from './GroundingBadge';
import { CitationList } from './CitationList';
import { EvidenceCards } from './EvidenceCards';

export interface AnswerDisplayProps {
  query: string;
  groundedResult: GroundedAnswerResponse | null;
  extractiveResult: QAResponse | null;
  className?: string;
}

export const AnswerDisplay: React.FC<AnswerDisplayProps> = ({
  query,
  groundedResult,
  extractiveResult,
  className = '',
}) => {
  if (!groundedResult && !extractiveResult) {
    return null;
  }

  // Grounded Answer rendering
  if (groundedResult) {
    const {
      answer,
      answerable,
      grounded,
      groundedness_score,
      grounding_status,
      citations,
      evidence,
      claims,
      reason,
      model_name,
    } = groundedResult;

    return (
      <div className={`space-y-6 animate-fadeIn ${className}`}>
        {/* Step 1: Question */}
        <div className="rounded-2xl border border-indigo-100 bg-indigo-50/40 p-4 sm:p-5">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
              <HelpCircle className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-indigo-700">
                Question
              </span>
              <p className="text-base font-medium text-slate-900 mt-0.5">{query}</p>
            </div>
          </div>
        </div>

        {/* Step 2: Answer Card */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-600 to-purple-600 text-white flex items-center justify-center shadow-2xs">
                <Sparkles className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900">Synthesized Answer</h3>
                {model_name && (
                  <p className="text-[11px] font-mono text-slate-400">Model: {model_name}</p>
                )}
              </div>
            </div>

            {/* Step 3: Grounding Status */}
            <GroundingBadge
              grounded={grounded}
              score={groundedness_score}
              status={grounding_status}
            />
          </div>

          <div className="mt-5">
            {answerable && answer ? (
              <div className="prose prose-slate max-w-none text-slate-800 text-sm leading-relaxed whitespace-pre-wrap font-sans">
                {answer}
              </div>
            ) : (
              <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-amber-900 text-sm flex items-start gap-3">
                <AlertOctagon className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-semibold">Unanswerable Query</h4>
                  <p className="mt-1 text-xs text-amber-800">
                    {reason ||
                      'The retrieved context does not contain sufficient verified evidence to answer this question accurately.'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Step 4: Citations */}
        {citations && citations.length > 0 && (
          <CitationList citations={citations} />
        )}

        {/* Step 5: Evidence Passages & Claims */}
        <EvidenceCards evidence={evidence || []} claims={claims || []} />
      </div>
    );
  }

  // Extractive QA rendering
  if (extractiveResult) {
    const {
      answer,
      answerable,
      page_number,
      chunk_index,
      qa_score,
      source_text,
      total_evidence_evaluated,
    } = extractiveResult;

    return (
      <div className={`space-y-6 animate-fadeIn ${className}`}>
        {/* Question */}
        <div className="rounded-2xl border border-indigo-100 bg-indigo-50/40 p-4 sm:p-5">
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0 mt-0.5">
              <HelpCircle className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-semibold uppercase tracking-wider text-indigo-700">
                Question
              </span>
              <p className="text-base font-medium text-slate-900 mt-0.5">{query}</p>
            </div>
          </div>
        </div>

        {/* Extracted Answer */}
        <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between pb-4 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-emerald-600 text-white flex items-center justify-center">
                <BookOpen className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900">Extracted Answer Span</h3>
                <p className="text-[11px] text-slate-500">
                  RoBERTa Extractive QA • Evaluated {total_evidence_evaluated} candidate passages
                </p>
              </div>
            </div>

            {page_number !== undefined && page_number !== null && (
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                Found on Page {page_number}
              </span>
            )}
          </div>

          <div className="mt-5">
            {answerable && answer ? (
              <div className="p-4 rounded-xl bg-emerald-50/60 border border-emerald-200 text-emerald-950 font-semibold text-base">
                "{answer}"
                {qa_score !== undefined && qa_score !== null && (
                  <span className="ml-3 text-xs font-normal font-mono text-emerald-700">
                    (confidence: {(qa_score * 100).toFixed(1)}%)
                  </span>
                )}
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-sm">
                No extractive answer span found in candidate passages.
              </div>
            )}

            {source_text && (
              <div className="mt-4">
                <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
                  Context Passage (Chunk #{chunk_index})
                </h5>
                <blockquote className="p-3 rounded-lg bg-slate-50 border-l-2 border-indigo-400 text-xs text-slate-700 italic">
                  "{source_text}"
                </blockquote>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return null;
};
