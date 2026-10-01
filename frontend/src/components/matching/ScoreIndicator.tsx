import React from 'react';
import { Compass, Sparkles } from 'lucide-react';

export interface ScoreIndicatorProps {
  similarityScore?: number | null;
  rerankerScore?: number | null;
  className?: string;
}

/**
 * Score semantics presentation:
 * - similarity_score: Vector-space semantic relevance (inner product / cosine similarity)
 * - reranker_score: Second-stage Cross-Encoder joint relevance score
 *
 * CRITICAL:
 * - Neither score represents factual truth, answer confidence, or hallucination probability.
 * - Scores are displayed as calibrated decimals, not misleading percentages.
 */
export const ScoreIndicator: React.FC<ScoreIndicatorProps> = ({
  similarityScore,
  rerankerScore,
  className = '',
}) => {
  return (
    <div className={`grid grid-cols-1 sm:grid-cols-2 gap-3 ${className}`}>
      {/* Semantic Relevance (Vector Distance) */}
      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="font-semibold text-slate-700 flex items-center gap-1.5">
            <Compass className="w-3.5 h-3.5 text-indigo-600" aria-hidden="true" />
            Semantic relevance
          </span>
          <span className="font-mono font-bold text-slate-900 text-xs">
            {similarityScore !== undefined && similarityScore !== null
              ? similarityScore.toFixed(4)
              : 'N/A'}
          </span>
        </div>

        {/* Visual Relative Bar (bounded 0.0 to 1.0 for cosine similarity) */}
        {similarityScore !== undefined && similarityScore !== null && (
          <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-indigo-600 h-1.5 rounded-full transition-all duration-300"
              style={{
                width: `${Math.min(100, Math.max(0, similarityScore * 100))}%`,
              }}
              role="progressbar"
              aria-valuenow={similarityScore}
              aria-valuemin={0}
              aria-valuemax={1}
              aria-label="Semantic vector relevance"
            />
          </div>
        )}
        <p className="text-[10px] text-slate-400 mt-1">Vector-space dense retrieval cosine score</p>
      </div>

      {/* Reranker Relevance (Cross-Encoder) */}
      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
        <div className="flex items-center justify-between text-xs mb-1.5">
          <span className="font-semibold text-slate-700 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-purple-600" aria-hidden="true" />
            Reranker relevance
          </span>
          <span className="font-mono font-bold text-slate-900 text-xs">
            {rerankerScore !== undefined && rerankerScore !== null
              ? rerankerScore.toFixed(4)
              : 'Not reranked'}
          </span>
        </div>

        {/* Cross-encoder visual indicator (sigmoid-style approximation for bar width if available) */}
        {rerankerScore !== undefined && rerankerScore !== null && (
          <div className="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
            <div
              className="bg-purple-600 h-1.5 rounded-full transition-all duration-300"
              style={{
                // Map common logit ranges (-5 to +10) gracefully to 5%-100% width for visual cue
                width: `${Math.min(100, Math.max(5, (1 / (1 + Math.exp(-rerankerScore))) * 100))}%`,
              }}
              role="progressbar"
              aria-valuenow={rerankerScore}
              aria-label="Cross-Encoder joint relevance score"
            />
          </div>
        )}
        <p className="text-[10px] text-slate-400 mt-1">
          {rerankerScore !== undefined && rerankerScore !== null
            ? 'Cross-Encoder joint query-passage transformer score'
            : 'First-stage candidate only (not scored by Cross-Encoder)'}
        </p>
      </div>
    </div>
  );
};
