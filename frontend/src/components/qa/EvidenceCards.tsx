import React, { useState } from 'react';
import { Layers, ChevronDown, ChevronUp, CheckCircle2, XCircle, AlertCircle, FileText } from 'lucide-react';
import { GenerationEvidenceItem, ClaimResult } from '@/types/qa';
import { Badge } from '@/components/common/Badge';

export interface EvidenceCardsProps {
  evidence: GenerationEvidenceItem[];
  claims?: ClaimResult[];
  className?: string;
}

export const EvidenceCards: React.FC<EvidenceCardsProps> = ({
  evidence,
  claims = [],
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<'evidence' | 'claims'>('evidence');
  const [expandedIndices, setExpandedIndices] = useState<Record<number, boolean>>({});

  const toggleExpand = (index: number) => {
    setExpandedIndices((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  if ((!evidence || evidence.length === 0) && (!claims || claims.length === 0)) {
    return null;
  }

  return (
    <div className={`rounded-2xl border border-slate-200/90 bg-white shadow-2xs overflow-hidden ${className}`}>
      {/* Tab Header */}
      <div className="flex items-center justify-between border-b border-slate-200 px-6 py-3.5 bg-slate-50/70">
        <div className="flex items-center gap-2 text-xs font-semibold text-slate-800 tracking-wide">
          <Layers className="w-4 h-4 text-indigo-600" />
          <span>GROUNDING PROVENANCE & RETRIEVAL CONTEXT</span>
        </div>

        <div className="flex items-center gap-1.5 p-1 bg-slate-200/60 rounded-lg text-xs">
          <button
            type="button"
            onClick={() => setActiveTab('evidence')}
            className={`px-3 py-1 rounded-md font-medium transition ${
              activeTab === 'evidence'
                ? 'bg-white text-slate-900 shadow-2xs'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Evidence Chunks ({evidence.length})
          </button>
          {claims.length > 0 && (
            <button
              type="button"
              onClick={() => setActiveTab('claims')}
              className={`px-3 py-1 rounded-md font-medium transition ${
                activeTab === 'claims'
                  ? 'bg-white text-slate-900 shadow-2xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Claim Verification ({claims.length})
            </button>
          )}
        </div>
      </div>

      <div className="p-6">
        {/* Evidence Tab */}
        {activeTab === 'evidence' && (
          <div className="space-y-3">
            {evidence.map((item, idx) => {
              const isExpanded = !!expandedIndices[idx];
              const textPreview =
                item.source_text.length > 250 && !isExpanded
                  ? `${item.source_text.slice(0, 250)}...`
                  : item.source_text;

              return (
                <div
                  key={item.chunk_id || idx}
                  className="rounded-xl border border-slate-200/80 bg-slate-50/50 p-4 transition-all"
                >
                  <div className="flex items-center justify-between gap-3 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center justify-center w-5 h-5 rounded-md bg-slate-200 text-slate-700 font-bold text-[10px]">
                        #{item.rank || idx + 1}
                      </span>
                      <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5 text-slate-400" />
                        Page {item.page_number}
                        <span className="text-slate-400 font-normal">• Chunk #{item.chunk_index}</span>
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      {item.reranker_score !== undefined && item.reranker_score !== null && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-100">
                          rerank: {item.reranker_score.toFixed(3)}
                        </span>
                      )}
                      {item.similarity_score !== undefined && item.similarity_score !== null && (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600">
                          sim: {item.similarity_score.toFixed(3)}
                        </span>
                      )}
                    </div>
                  </div>

                  <p className="text-xs text-slate-700 leading-relaxed font-sans">{textPreview}</p>

                  {item.source_text.length > 250 && (
                    <button
                      type="button"
                      onClick={() => toggleExpand(idx)}
                      className="mt-2 text-[11px] font-semibold text-indigo-600 hover:text-indigo-800 flex items-center gap-1 focus:outline-none"
                    >
                      {isExpanded ? (
                        <>
                          <ChevronUp className="w-3 h-3" />
                          Show less
                        </>
                      ) : (
                        <>
                          <ChevronDown className="w-3 h-3" />
                          Read full passage
                        </>
                      )}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Claims Tab */}
        {activeTab === 'claims' && (
          <div className="space-y-3">
            {claims.map((claim) => {
              const status = claim.status.toLowerCase();
              let badgeVariant: 'success' | 'error' | 'warning' = 'warning';
              let StatusIcon = AlertCircle;

              if (status === 'entailed') {
                badgeVariant = 'success';
                StatusIcon = CheckCircle2;
              } else if (status === 'contradicted') {
                badgeVariant = 'error';
                StatusIcon = XCircle;
              }

              return (
                <div
                  key={claim.claim_index}
                  className="rounded-xl border border-slate-200/90 bg-white p-4 space-y-2 shadow-2xs"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-2">
                      <span className="text-xs font-mono font-bold text-slate-400 mt-0.5">
                        C{claim.claim_index + 1}.
                      </span>
                      <p className="text-xs font-medium text-slate-900 leading-snug">
                        "{claim.claim_text}"
                      </p>
                    </div>
                    <Badge variant={badgeVariant} className="shrink-0 gap-1 text-[11px] capitalize">
                      <StatusIcon className="w-3 h-3" />
                      {claim.status}
                    </Badge>
                  </div>

                  <div className="flex flex-wrap items-center gap-3 pt-1 text-[11px] text-slate-500 font-mono">
                    <span>entailment: {(claim.entailment_score * 100).toFixed(1)}%</span>
                    <span>contradiction: {(claim.contradiction_score * 100).toFixed(1)}%</span>
                    <span>neutral: {(claim.neutral_score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
