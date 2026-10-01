import React from 'react';
import { BookOpen, FileText, Calendar, ArrowRight, MessageSquare, Trash2 } from 'lucide-react';
import { PersistedDocument } from '@/types/document';
import { ProcessingStatusBadge } from '@/components/processing/ProcessingStatusBadge';
import { Link } from '@/router';

export interface BookCardProps {
  document: PersistedDocument;
  onDelete?: (documentId: string) => void;
  className?: string;
}

export const BookCard: React.FC<BookCardProps> = ({
  document,
  onDelete,
  className = '',
}) => {
  const isProcessed = document.status === 'processed';

  const formattedDate = document.created_at
    ? new Date(document.created_at).toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      })
    : null;

  return (
    <div
      className={`group relative flex flex-col justify-between rounded-2xl border border-slate-200/80 bg-white p-6 transition-all duration-200 hover:border-indigo-300 hover:shadow-md hover:shadow-indigo-50/50 ${className}`}
    >
      <div>
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="w-12 h-12 rounded-xl bg-indigo-50 border border-indigo-100/60 text-indigo-600 flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
            <BookOpen className="w-6 h-6" />
          </div>
          <ProcessingStatusBadge status={document.status} />
        </div>

        <h3 className="font-semibold text-slate-900 line-clamp-2 text-base group-hover:text-indigo-600 transition-colors">
          {document.title || document.filename}
        </h3>

        {document.author && (
          <p className="text-xs font-medium text-slate-500 mt-1 line-clamp-1">
            by {document.author}
          </p>
        )}

        {/* Metadata pills */}
        <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-500">
          <div className="inline-flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5 text-slate-400" />
            <span>
              {document.page_count > 0 ? `${document.page_count} pages` : 'Pending pages'}
            </span>
          </div>
          {formattedDate && (
            <div className="inline-flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-slate-400" />
              <span>{formattedDate}</span>
            </div>
          )}
        </div>
      </div>

      <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between gap-2">
        {isProcessed ? (
          <Link
            to={`/books/${document.document_id}`}
            className="inline-flex items-center gap-2 text-xs font-semibold text-indigo-600 hover:text-indigo-800 transition focus:outline-none focus:ring-2 focus:ring-indigo-500 rounded-lg p-1"
          >
            <MessageSquare className="w-4 h-4" />
            <span>Ask Questions</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </Link>
        ) : (
          <span className="text-xs text-slate-400 italic">Processing in background...</span>
        )}

        {onDelete && (
          <button
            type="button"
            onClick={() => onDelete(document.document_id)}
            className="text-slate-400 hover:text-rose-600 p-1.5 rounded-lg transition hover:bg-rose-50 focus:outline-none focus:ring-2 focus:ring-rose-500"
            title="Delete book"
            aria-label={`Delete ${document.title || document.filename}`}
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};
