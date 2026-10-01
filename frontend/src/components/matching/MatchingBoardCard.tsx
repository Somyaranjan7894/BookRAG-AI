import React, { useState } from 'react';
import { ChevronDown, ChevronUp, FileText, CheckCircle2 } from 'lucide-react';
import { RankingMovementBadge } from './RankingMovementBadge';
import { ScoreIndicator } from './ScoreIndicator';

export interface MatchingBoardCardProps {
  rank: number;
  originalRank?: number | null;
  chunkId: string;
  documentId: string;
  pageNumber: number;
  chunkIndex?: number;
  text: string;
  similarityScore: number;
  rerankerScore?: number | null;
  isUsedInAnswer?: boolean;
  citationId?: string | null;
  className?: string;
}

export const MatchingBoardCard: React.FC<MatchingBoardCardProps> = ({
  rank,
  originalRank,
  chunkId,
  pageNumber,
  chunkIndex = 0,
  text,
  similarityScore,
  rerankerScore,
  isUsedInAnswer = false,
  citationId,
  className = '',
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const passageSnippet = text.length > 240 ? `${text.slice(0, 240)}...` : text;
  const hasLongPassage = text.length > 240;

  return (
    <article
      className={`rounded-2xl border transition-all p-5 ${
        isUsedInAnswer
          ? 'border-indigo-300 bg-white shadow-sm ring-1 ring-indigo-200/50'
          : 'border-slate-200/90 bg-white shadow-2xs hover:border-slate-300'
      } ${className}`}
      aria-labelledby={`card-title-${chunkId}`}
    >
      {/* Top Header Line */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          {/* Final Rank Badge */}
          <span
            id={`card-title-${chunkId}`}
            className="w-7 h-7 rounded-lg bg-slate-900 text-white font-mono font-bold text-xs flex items-center justify-center shadow-2xs"
            title={`Final Rank #${rank}`}
          >
            #{rank}
          </span>

          {/* Original Rank Cue */}
          {originalRank !== undefined && originalRank !== null && (
            <span className="text-xs font-mono font-medium text-slate-500">
              Original #{originalRank}
            </span>
          )}

          {/* Ranking Movement */}
          <RankingMovementBadge originalRank={originalRank} finalRank={rank} />
        </div>

        {/* Right Badges: Page Provenance & Used In Answer */}
        <div className="flex items-center gap-2">
          {isUsedInAnswer && (
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
              <CheckCircle2 className="w-3.5 h-3.5 text-indigo-600" aria-hidden="true" />
              <span>Used in Answer</span>
              {citationId && <span className="font-mono text-[10px]">[{citationId}]</span>}
            </span>
          )}

          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200">
            <FileText className="w-3 h-3 text-slate-500" aria-hidden="true" />
            <span>Page {pageNumber}</span>
          </span>
        </div>
      </div>

      {/* Provenance Metadata Sub-bar */}
      <div className="mt-2.5 flex flex-wrap items-center gap-3 text-[11px] text-slate-400 font-mono">
        <span className="truncate max-w-[180px] sm:max-w-xs">ID: {chunkId}</span>
        <span>•</span>
        <span>Chunk #{chunkIndex}</span>
      </div>

      {/* Source Passage */}
      <div className="mt-3.5">
        <div className="p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/60 text-xs text-slate-800 leading-relaxed font-sans whitespace-pre-wrap">
          {isExpanded ? text : passageSnippet}
        </div>

        {hasLongPassage && (
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            aria-expanded={isExpanded}
            aria-controls={`passage-${chunkId}`}
            className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 hover:text-indigo-800 transition focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded p-1"
          >
            {isExpanded ? (
              <>
                <ChevronUp className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Show less</span>
              </>
            ) : (
              <>
                <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" />
                <span>Show full passage ({text.length} chars)</span>
              </>
            )}
          </button>
        )}
      </div>

      {/* Score Semantics Section */}
      <div className="mt-4 pt-3.5 border-t border-slate-100">
        <ScoreIndicator
          similarityScore={similarityScore}
          rerankerScore={rerankerScore}
        />
      </div>
    </article>
  );
};
