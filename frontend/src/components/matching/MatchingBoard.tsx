import React, { useState, useMemo } from 'react';
import { HelpCircle, ArrowRightLeft, Filter, Info } from 'lucide-react';
import { SearchResult } from '@/types/search';
import { GenerationEvidenceItem, Citation } from '@/types/qa';
import { MatchingBoardCard } from './MatchingBoardCard';

export interface MatchingBoardItem {
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
}

export interface MatchingBoardProps {
  query: string;
  results?: SearchResult[] | null;
  evidence?: GenerationEvidenceItem[] | null;
  citations?: Citation[] | null;
  candidateCount?: number | null;
  rerankingApplied?: boolean;
  className?: string;
  onClose?: () => void;
}

export const MatchingBoard: React.FC<MatchingBoardProps> = ({
  query,
  results,
  evidence,
  citations = [],
  candidateCount,
  rerankingApplied = true,
  className = '',
  onClose,
}) => {
  const [activeFilter, setActiveFilter] = useState<'all' | 'used' | 'promoted'>('all');

  // Cited chunk ID lookup
  const citedMap = useMemo(() => {
    const map = new Map<string, string>();
    if (citations) {
      for (const cit of citations) {
        if (cit.chunk_id) {
          map.set(cit.chunk_id, cit.citation_id);
        }
      }
    }
    return map;
  }, [citations]);

  // Normalize inputs into unified MatchingBoardItem list
  const unifiedItems: MatchingBoardItem[] = useMemo(() => {
    if (results && results.length > 0) {
      return results.map((r) => ({
        rank: r.rank,
        originalRank: r.original_rank,
        chunkId: r.chunk_id,
        documentId: r.document_id,
        pageNumber: r.page_number,
        chunkIndex: r.chunk_index ?? 0,
        text: r.text,
        similarityScore: r.similarity_score,
        rerankerScore: r.reranker_score,
        isUsedInAnswer: citedMap.has(r.chunk_id),
        citationId: citedMap.get(r.chunk_id) || null,
      }));
    }

    if (evidence && evidence.length > 0) {
      return evidence.map((e) => ({
        rank: e.rank,
        originalRank: e.original_rank,
        chunkId: e.chunk_id,
        documentId: e.document_id,
        pageNumber: e.page_number,
        chunkIndex: e.chunk_index ?? 0,
        text: e.source_text,
        similarityScore: e.similarity_score ?? 0,
        rerankerScore: e.reranker_score,
        isUsedInAnswer: citedMap.has(e.chunk_id),
        citationId: citedMap.get(e.chunk_id) || null,
      }));
    }

    return [];
  }, [results, evidence, citedMap]);

  // Filtered items
  const filteredItems = useMemo(() => {
    if (activeFilter === 'used') {
      return unifiedItems.filter((i) => i.isUsedInAnswer);
    }
    if (activeFilter === 'promoted') {
      return unifiedItems.filter(
        (i) => i.originalRank !== undefined && i.originalRank !== null && i.originalRank > i.rank
      );
    }
    return unifiedItems;
  }, [unifiedItems, activeFilter]);

  const effectiveCandidateCount = candidateCount || (unifiedItems.length > 0 ? unifiedItems.length : 0);
  const selectedCount = unifiedItems.length;

  return (
    <section
      className={`space-y-6 animate-fadeIn ${className}`}
      aria-labelledby="matching-board-title"
    >
      {/* Board Header Card */}
      <div className="rounded-3xl border border-slate-200/90 bg-white p-6 sm:p-8 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-100">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-purple-50 text-purple-700 border border-purple-200/80 mb-2">
              <ArrowRightLeft className="w-3.5 h-3.5 text-purple-600" aria-hidden="true" />
              <span>Phase 18 Retrieval Transparency</span>
            </div>
            <h2 id="matching-board-title" className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
              Relevant Matching Board
            </h2>
            <p className="text-xs text-slate-500 mt-1 max-w-xl">
              Inspect how candidate chunks were retrieved from dense vector space and reordered by the Cross-Encoder.
            </p>
          </div>

          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="self-start sm:self-center px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 transition focus:outline-none focus:ring-2 focus:ring-slate-400"
            >
              Close Board
            </button>
          )}
        </div>

        {/* User Query Display */}
        <div className="mt-5 p-4 rounded-2xl bg-indigo-50/50 border border-indigo-100">
          <div className="flex items-start gap-3">
            <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
              <HelpCircle className="w-3.5 h-3.5" aria-hidden="true" />
            </div>
            <div className="min-w-0 flex-1">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-indigo-700">
                Submitted Query
              </span>
              <p className="text-sm sm:text-base font-medium text-slate-900 mt-0.5 break-words">
                {query}
              </p>
            </div>
          </div>
        </div>

        {/* Retrieval Summary KPI Grid */}
        <div className="mt-5 grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <span className="text-xs font-semibold text-slate-500 block">Candidate Pool</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-2xl font-extrabold text-slate-900 font-mono">
                {effectiveCandidateCount}
              </span>
              <span className="text-xs text-slate-500">candidates retrieved</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">Stage 1 dense vector search</p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <span className="text-xs font-semibold text-slate-500 block">Selected Evidence</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span className="text-2xl font-extrabold text-indigo-600 font-mono">
                {selectedCount}
              </span>
              <span className="text-xs text-slate-500">passages selected</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">Top-K evaluated for answer</p>
          </div>

          <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200/80">
            <span className="text-xs font-semibold text-slate-500 block">Precision Reranking</span>
            <div className="mt-1 flex items-baseline gap-2">
              <span
                className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
                  rerankingApplied
                    ? 'bg-purple-50 text-purple-700 border-purple-200'
                    : 'bg-slate-100 text-slate-600 border-slate-200'
                }`}
              >
                {rerankingApplied ? 'Cross-Encoder Applied' : 'First-Stage Only'}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              {rerankingApplied ? 'ms-marco-MiniLM cross-attention' : 'Cosine similarity only'}
            </p>
          </div>
        </div>

        {/* Methodology Note on Score Semantics */}
        <div className="mt-5 p-3.5 rounded-xl bg-amber-50/70 border border-amber-200/60 text-xs text-amber-900 flex items-start gap-2.5">
          <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="leading-relaxed">
            <span className="font-semibold">Score Semantics: </span>
            <strong>Semantic relevance</strong> measures cosine vector proximity in dense embedding space.
            <strong> Reranker relevance</strong> reflects joint Cross-Encoder transformer attention over (query, passage).
            Neither score represents objective factual truth or answer confidence.
          </div>
        </div>
      </div>

      {/* Filter and Tab Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-100 border border-slate-200 text-xs">
          <button
            type="button"
            onClick={() => setActiveFilter('all')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              activeFilter === 'all'
                ? 'bg-white text-indigo-700 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            All Evidence ({unifiedItems.length})
          </button>

          <button
            type="button"
            onClick={() => setActiveFilter('used')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              activeFilter === 'used'
                ? 'bg-white text-indigo-700 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Used in Answer ({unifiedItems.filter((i) => i.isUsedInAnswer).length})
          </button>

          <button
            type="button"
            onClick={() => setActiveFilter('promoted')}
            className={`px-3 py-1.5 rounded-lg font-semibold transition ${
              activeFilter === 'promoted'
                ? 'bg-white text-indigo-700 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Promoted ↑ ({
              unifiedItems.filter(
                (i) => i.originalRank !== undefined && i.originalRank !== null && i.originalRank > i.rank
              ).length
            })
          </button>
        </div>

        <span className="text-xs text-slate-400 font-mono">
          Showing {filteredItems.length} of {unifiedItems.length} passages
        </span>
      </div>

      {/* Results List */}
      {filteredItems.length > 0 ? (
        <div className="space-y-4">
          {filteredItems.map((item) => (
            <MatchingBoardCard
              key={`${item.documentId}-${item.chunkId}-${item.rank}`}
              rank={item.rank}
              originalRank={item.originalRank}
              chunkId={item.chunkId}
              documentId={item.documentId}
              pageNumber={item.pageNumber}
              chunkIndex={item.chunkIndex}
              text={item.text}
              similarityScore={item.similarityScore}
              rerankerScore={item.rerankerScore}
              isUsedInAnswer={item.isUsedInAnswer}
              citationId={item.citationId}
            />
          ))}
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-slate-300 bg-white/60 p-10 text-center">
          <Filter className="w-8 h-8 mx-auto text-slate-400 mb-2" aria-hidden="true" />
          <h3 className="text-sm font-semibold text-slate-900">No passages match this filter</h3>
          <p className="text-xs text-slate-500 mt-1">
            Try switching back to &quot;All Evidence&quot; to inspect the complete set of retrieved passages.
          </p>
        </div>
      )}
    </section>
  );
};
