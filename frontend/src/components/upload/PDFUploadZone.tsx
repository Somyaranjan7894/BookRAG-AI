import React, { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { UploadCloud, FileText, CheckCircle2, AlertTriangle, ArrowRight } from 'lucide-react';
import { documentApi } from '@/api/documents';
import { DocumentUploadResponse } from '@/types/document';
import { ApiError } from '@/types/api';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';

export interface PDFUploadZoneProps {
  onUploadAccepted?: (response: DocumentUploadResponse) => void;
  className?: string;
}

export const PDFUploadZone: React.FC<PDFUploadZoneProps> = ({
  onUploadAccepted,
  className = '',
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [acceptedResponse, setAcceptedResponse] = useState<DocumentUploadResponse | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateAndUpload = async (file: File) => {
    setValidationError(null);
    setAcceptedResponse(null);

    // Strict PDF validation by MIME type and filename extension
    const isPdfMime = file.type === 'application/pdf';
    const isPdfExt = file.name.toLowerCase().endsWith('.pdf');

    if (!isPdfMime && !isPdfExt) {
      setValidationError(`Invalid file type "${file.name}". Only PDF documents (.pdf) are supported.`);
      return;
    }

    if (file.size === 0) {
      setValidationError('The selected PDF file is empty (0 bytes).');
      return;
    }

    setIsUploading(true);

    try {
      const response = await documentApi.upload(file);
      setAcceptedResponse(response);
      if (onUploadAccepted) {
        onUploadAccepted(response);
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setValidationError(err.message);
      } else {
        setValidationError(err instanceof Error ? err.message : 'An error occurred during upload.');
      }
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndUpload(file);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndUpload(file);
    }
  };

  const triggerPicker = () => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className={`w-full ${className}`}>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerPicker}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            triggerPicker();
          }
        }}
        aria-label="Upload PDF book"
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 ${
          isDragging
            ? 'border-indigo-500 bg-indigo-50/70 scale-[1.01]'
            : isUploading
            ? 'border-slate-300 bg-slate-50 cursor-not-allowed'
            : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50/80 bg-white'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleFileInputChange}
          className="hidden"
          disabled={isUploading}
          data-testid="pdf-file-input"
        />

        <div className="flex flex-col items-center justify-center max-w-md mx-auto">
          {isUploading ? (
            <div className="py-4">
              <LoadingSpinner size="lg" label="Uploading book to server..." />
              <p className="mt-3 text-xs text-slate-500">
                Transferring PDF and initiating asynchronous processing pipeline...
              </p>
            </div>
          ) : (
            <>
              <div className="w-14 h-14 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-4 shadow-sm group-hover:scale-110 transition-transform">
                <UploadCloud className="w-7 h-7" />
              </div>
              <h3 className="text-base font-semibold text-slate-900 mb-1">
                Upload a Complete PDF Book
              </h3>
              <p className="text-sm text-slate-500 mb-4">
                Drag and drop your file here, or{' '}
                <span className="text-indigo-600 font-semibold underline underline-offset-2">
                  browse files
                </span>
              </p>
              <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-100/70 px-3 py-1.5 rounded-full">
                <FileText className="w-3.5 h-3.5" />
                <span>PDF format only • Full books supported</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Validation / Format Error Banner */}
      {validationError && (
        <div
          role="alert"
          className="mt-3 p-3.5 rounded-xl border border-rose-200 bg-rose-50 text-rose-800 text-sm flex items-start gap-2.5 animate-fadeIn"
        >
          <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <span className="font-medium">Upload Rejected: </span>
            {validationError}
          </div>
        </div>
      )}

      {/* Upload Accepted (HTTP 202) Notification */}
      {acceptedResponse && !isUploading && (
        <div
          role="status"
          aria-live="polite"
          className="mt-4 p-4 rounded-xl border border-indigo-200 bg-gradient-to-r from-indigo-50/90 to-purple-50/90 text-indigo-950 text-sm flex items-start gap-3 shadow-sm animate-fadeIn"
        >
          <CheckCircle2 className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h4 className="font-semibold text-indigo-950">
                Upload Accepted — The book is being processed.
              </h4>
              <span className="text-[11px] font-mono font-medium px-2 py-0.5 bg-indigo-100 text-indigo-800 rounded">
                HTTP 202
              </span>
            </div>
            <p className="mt-1 text-xs text-indigo-800 leading-relaxed">
              Your document ID is <code className="font-semibold">{acceptedResponse.document_id}</code>.
              Celery background workers are parsing pages, generating semantic chunks, and building vector indices.
            </p>
            <div className="mt-2 text-xs font-medium text-indigo-700 flex items-center gap-1">
              <span>Tracking live status below</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
