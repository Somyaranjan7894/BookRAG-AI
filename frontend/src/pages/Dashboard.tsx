import React, { useState, useEffect, useCallback } from 'react';
import { BookOpen, RefreshCw, Plus, Sparkles, Database } from 'lucide-react';
import { documentApi } from '@/api/documents';
import { PersistedDocument, DocumentUploadResponse } from '@/types/document';
import { ApiError } from '@/types/api';
import { PDFUploadZone } from '@/components/upload/PDFUploadZone';
import { ProcessingProgress } from '@/components/processing/ProcessingProgress';
import { BookList } from '@/components/books/BookList';
import { ErrorAlert } from '@/components/common/ErrorAlert';

export const Dashboard: React.FC = () => {
  const [documents, setDocuments] = useState<PersistedDocument[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<ApiError | null>(null);
  const [activeUpload, setActiveUpload] = useState<DocumentUploadResponse | null>(null);
  const [showUploadZone, setShowUploadZone] = useState<boolean>(false);

  const fetchDocuments = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await documentApi.list(0, 50);
      setDocuments(response.documents || []);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err);
      } else {
        setError(
          new ApiError({
            code: 500,
            message: err instanceof Error ? err.message : 'Failed to fetch documents',
            error_type: 'FETCH_DOCUMENTS_FAILED',
          })
        );
      }
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleUploadAccepted = (response: DocumentUploadResponse) => {
    setActiveUpload(response);
    // Refresh documents to include the queued entry
    fetchDocuments();
  };

  const handleProcessingFinished = () => {
    // Refresh book list to reflect processed status and page count
    fetchDocuments();
  };

  const handleDeleteDocument = async (documentId: string) => {
    if (!window.confirm('Are you sure you want to delete this book from the library?')) {
      return;
    }

    try {
      await documentApi.delete(documentId);
      setDocuments((prev) => prev.filter((doc) => doc.document_id !== documentId));
      if (activeUpload?.document_id === documentId) {
        setActiveUpload(null);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err);
      }
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Top Banner / Hero */}
      <div className="relative rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white p-8 sm:p-10 overflow-hidden shadow-lg">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-80 h-80 rounded-full bg-indigo-500/10 blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 -mb-16 w-60 h-60 rounded-full bg-purple-500/10 blur-2xl pointer-events-none" />

        <div className="relative z-10 max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-sm border border-white/10 text-xs font-medium text-indigo-200 mb-4">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>Production RAG Architecture</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Book Knowledge Base
          </h1>
          <p className="mt-3 text-sm sm:text-base text-slate-300 leading-relaxed">
            Upload PDF books, observe background chunking and vector indexing, and query with verified NLI groundedness and sentence-level citations.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => setShowUploadZone(!showUploadZone)}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-semibold bg-indigo-600 hover:bg-indigo-500 text-white transition shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            >
              <Plus className="w-4 h-4" />
              {showUploadZone ? 'Close Upload' : 'Upload New Book'}
            </button>

            <button
              type="button"
              onClick={fetchDocuments}
              className="inline-flex items-center gap-2 px-3.5 py-2.5 rounded-xl text-sm font-medium bg-white/10 hover:bg-white/15 text-white backdrop-blur-sm border border-white/10 transition focus:outline-none focus:ring-2 focus:ring-white/40"
              title="Refresh book list"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Upload Zone (collapsible or displayed when triggered) */}
      {(showUploadZone || documents.length === 0) && (
        <div className="rounded-2xl border border-slate-200/90 bg-white p-6 shadow-sm">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-700 mb-4 flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-indigo-600" />
            Upload Book to Library
          </h2>
          <PDFUploadZone onUploadAccepted={handleUploadAccepted} />
        </div>
      )}

      {/* Active Processing Card */}
      {activeUpload && (
        <ProcessingProgress
          documentId={activeUpload.document_id}
          onFinished={handleProcessingFinished}
        />
      )}

      {/* Error Alert */}
      {error && <ErrorAlert error={error} onDismiss={() => setError(null)} onRetry={fetchDocuments} />}

      {/* Book Library Header */}
      <div>
        <div className="flex items-center justify-between pb-4 border-b border-slate-200">
          <div>
            <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
              <Database className="w-5 h-5 text-indigo-600" />
              Document Library
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              {documents.length} {documents.length === 1 ? 'book' : 'books'} indexed in PostgreSQL
            </p>
          </div>
        </div>

        {/* Book Cards Grid */}
        <div className="mt-6">
          <BookList
            documents={documents}
            isLoading={isLoading}
            onDelete={handleDeleteDocument}
            onUploadClick={() => setShowUploadZone(true)}
          />
        </div>
      </div>
    </div>
  );
};
