import React from 'react';
import { PersistedDocument } from '@/types/document';
import { BookCard } from './BookCard';
import { BookEmptyState } from './BookEmptyState';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';

export interface BookListProps {
  documents: PersistedDocument[];
  isLoading: boolean;
  onDelete?: (documentId: string) => void;
  onUploadClick?: () => void;
  className?: string;
}

export const BookList: React.FC<BookListProps> = ({
  documents,
  isLoading,
  onDelete,
  onUploadClick,
  className = '',
}) => {
  if (isLoading && documents.length === 0) {
    return (
      <div className="py-20 flex justify-center items-center">
        <LoadingSpinner size="lg" label="Loading books from library..." />
      </div>
    );
  }

  if (documents.length === 0) {
    return <BookEmptyState onUploadClick={onUploadClick} className={className} />;
  }

  return (
    <div
      className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 ${className}`}
      data-testid="book-grid"
    >
      {documents.map((doc) => (
        <BookCard key={doc.document_id} document={doc} onDelete={onDelete} />
      ))}
    </div>
  );
};
