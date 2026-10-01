import React, { useState } from 'react';
import { Bookmark, FileText, Copy, Check } from 'lucide-react';
import { Citation } from '@/types/qa';

export interface CitationListProps {
  citations: Citation[];
  className?: string;
}

export const CitationList: React.FC<CitationListProps> = ({
  citations,
  className = '',
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  if (!citations || citations.length === 0) return null;

  const handleCopy = (citationId: string, text: string) => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      setCopiedId(citationId);
      setTimeout(() => setCopiedId(null), 2000);
    }
  };

  return (
    <div className={`space-y-3 ${className}`}>
      <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 tracking-wide uppercase">
        <Bookmark className="w-3.5 h-3.5 text-indigo-600" />
        <span>Citations & References ({citations.length})</span>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {citations.map((cite, index) => {
          const isCopied = copiedId === cite.citation_id;

          return (
            <div
              key={cite.citation_id || index}
              className="rounded-xl border border-slate-200 bg-white p-4 shadow-2xs hover:border-slate-300 transition-all text-xs"
            >
              <div className="flex items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-indigo-100 text-indigo-700 font-bold text-[10px]">
                    [{index + 1}]
                  </span>
                  <div className="flex items-center gap-1.5 font-semibold text-slate-800">
                    <FileText className="w-3.5 h-3.5 text-slate-400" />
                    <span>Page {cite.page_number}</span>
                    <span className="text-slate-400">•</span>
                    <span className="text-slate-500 font-normal">Chunk #{cite.chunk_index}</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {cite.reranker_score !== undefined && cite.reranker_score !== null && (
                    <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-mono text-[10px]">
                      score: {cite.reranker_score.toFixed(3)}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => handleCopy(cite.citation_id, cite.source_text)}
                    className="p-1 rounded text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    title="Copy passage text"
                    aria-label="Copy citation text"
                  >
                    {isCopied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </div>
              </div>

              <blockquote className="text-slate-700 leading-relaxed bg-slate-50/70 p-3 rounded-lg border-l-2 border-indigo-400 font-normal italic">
                "{cite.source_text}"
              </blockquote>

              <div className="mt-2.5 flex items-center justify-between text-[11px] text-slate-400">
                <span className="font-mono truncate max-w-xs">ID: {cite.chunk_id}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
