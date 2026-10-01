import React from 'react';
import { ArrowUp, ArrowDown, Minus } from 'lucide-react';

export interface RankingMovementBadgeProps {
  originalRank?: number | null;
  finalRank: number;
  className?: string;
}

/**
 * Deterministic ranking movement badge:
 * rank_delta = original_rank - final_rank
 * - positive: moved upward (e.g. Original #7 -> Final #2 = ↑ 5 positions)
 * - negative: moved downward (e.g. Original #1 -> Final #4 = ↓ 3 positions)
 * - zero: unchanged (e.g. Original #2 -> Final #2 = — Unchanged)
 *
 * NOTE: Uses neutral "Ranking movement" terminology.
 * Does not imply accuracy or quality improvement.
 */
export const RankingMovementBadge: React.FC<RankingMovementBadgeProps> = ({
  originalRank,
  finalRank,
  className = '',
}) => {
  if (originalRank === undefined || originalRank === null) {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200 ${className}`}
        title="Original first-stage rank not available"
      >
        <Minus className="w-3 h-3 text-slate-400" />
        <span>No initial rank</span>
      </span>
    );
  }

  const delta = originalRank - finalRank;

  if (delta > 0) {
    return (
      <span
        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200/80 ${className}`}
        aria-label={`Ranking movement: moved up ${delta} positions`}
        title={`Promoted from original position #${originalRank} to #${finalRank} by Cross-Encoder reranking`}
      >
        <ArrowUp className="w-3.5 h-3.5 text-emerald-600 shrink-0" aria-hidden="true" />
        <span>↑ {delta} {delta === 1 ? 'position' : 'positions'}</span>
      </span>
    );
  }

  if (delta < 0) {
    const absDelta = Math.abs(delta);
    return (
      <span
        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-amber-50 text-amber-800 border border-amber-200/80 ${className}`}
        aria-label={`Ranking movement: moved down ${absDelta} positions`}
        title={`Demoted from original position #${originalRank} to #${finalRank} by Cross-Encoder reranking`}
      >
        <ArrowDown className="w-3.5 h-3.5 text-amber-600 shrink-0" aria-hidden="true" />
        <span>↓ {absDelta} {absDelta === 1 ? 'position' : 'positions'}</span>
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 border border-slate-200 ${className}`}
      aria-label="Ranking movement: unchanged position"
      title={`Maintained position #${finalRank} after Cross-Encoder reranking`}
    >
      <Minus className="w-3.5 h-3.5 text-slate-400 shrink-0" aria-hidden="true" />
      <span>— Unchanged</span>
    </span>
  );
};
