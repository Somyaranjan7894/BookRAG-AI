import React from 'react';
import { BookOpen, Upload } from 'lucide-react';

export interface BookEmptyStateProps {
  onUploadClick?: () => void;
  className?: string;
}

export const BookEmptyState: React.FC<BookEmptyStateProps> = ({
  onUploadClick,
  className = '',
}) => {
  return (
    <div
      role="region"
      aria-label="No documents found"
      className={`text-center py-16 px-6 rounded-2xl border border-dashed border-slate-300 bg-white/60 ${className}`}
    >
      <div className="w-16 h-16 mx-auto rounded-2xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center mb-4 shadow-sm">
        <BookOpen className="w-8 h-8" />
      </div>
      <h3 className="text-base font-semibold text-slate-900 mb-1">Your Library is Empty</h3>
      <p className="text-sm text-slate-500 max-w-sm mx-auto mb-6">
        Upload complete PDF books to start building your grounded semantic knowledge base.
      </p>
      {onUploadClick && (
        <button
          type="button"
          onClick={onUploadClick}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold bg-indigo-600 text-white hover:bg-indigo-700 transition shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          <Upload className="w-4 h-4" />
          Upload First Book
        </button>
      )}
    </div>
  );
};
