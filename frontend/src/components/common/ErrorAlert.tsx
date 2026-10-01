import React, { useState } from 'react';
import { AlertCircle, ChevronDown, ChevronUp, X, RefreshCw } from 'lucide-react';
import { ApiError } from '@/types/api';

export interface ErrorAlertProps {
  error: ApiError | Error | string | null | undefined;
  title?: string;
  onDismiss?: () => void;
  onRetry?: () => void;
  className?: string;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  error,
  title,
  onDismiss,
  onRetry,
  className = '',
}) => {
  const [showDetails, setShowDetails] = useState(false);

  if (!error) return null;

  const isApiError = error instanceof ApiError;
  const message = typeof error === 'string' ? error : error.message;
  const errorType = isApiError ? error.error_type : undefined;
  const requestId = isApiError ? error.request_id : undefined;
  const statusCode = isApiError ? error.code : undefined;
  const details = isApiError ? error.details : undefined;

  // Compute a user-friendly heading if not provided
  let computedTitle = title;
  if (!computedTitle) {
    if (statusCode === 404) computedTitle = 'Resource Not Found';
    else if (statusCode === 400 || statusCode === 422) computedTitle = 'Validation Error';
    else if (statusCode === 408) computedTitle = 'Request Timed Out';
    else if (statusCode === 0 || errorType === 'NETWORK_ERROR') computedTitle = 'Network Connection Error';
    else computedTitle = 'An Error Occurred';
  }

  return (
    <div
      role="alert"
      className={`rounded-xl border border-rose-200 bg-rose-50/80 p-4 text-rose-900 shadow-sm transition-all ${className}`}
    >
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" aria-hidden="true" />

        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-sm font-semibold text-rose-900">{computedTitle}</h4>
            {onDismiss && (
              <button
                type="button"
                onClick={onDismiss}
                className="text-rose-400 hover:text-rose-700 p-0.5 rounded focus:outline-none focus:ring-2 focus:ring-rose-500"
                aria-label="Dismiss error"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          <p className="mt-1 text-sm text-rose-800 leading-relaxed break-words">{message}</p>

          {/* Action buttons if available */}
          <div className="mt-3 flex flex-wrap items-center gap-3">
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-600 text-white hover:bg-rose-700 transition shadow-sm focus:outline-none focus:ring-2 focus:ring-rose-500"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Try Again
              </button>
            )}

            {(requestId || errorType || details) && (
              <button
                type="button"
                onClick={() => setShowDetails(!showDetails)}
                className="inline-flex items-center gap-1 text-xs font-medium text-rose-700 hover:text-rose-900 underline focus:outline-none"
              >
                {showDetails ? (
                  <>
                    <ChevronUp className="w-3.5 h-3.5" />
                    Hide technical details
                  </>
                ) : (
                  <>
                    <ChevronDown className="w-3.5 h-3.5" />
                    View technical details
                  </>
                )}
              </button>
            )}
          </div>

          {/* Expandable details area (no stack traces) */}
          {showDetails && (
            <div className="mt-3 pt-3 border-t border-rose-200/60 text-xs font-mono text-rose-800 space-y-1 bg-rose-100/50 p-2.5 rounded-lg">
              {errorType && (
                <div>
                  <span className="font-semibold text-rose-900">Type: </span>
                  {errorType}
                </div>
              )}
              {statusCode !== undefined && statusCode > 0 && (
                <div>
                  <span className="font-semibold text-rose-900">Status: </span>
                  {statusCode}
                </div>
              )}
              {requestId && (
                <div>
                  <span className="font-semibold text-rose-900">Request ID: </span>
                  <span className="select-all bg-rose-200/60 px-1 py-0.5 rounded">{requestId}</span>
                </div>
              )}
              {details && (
                <div className="mt-1">
                  <span className="font-semibold text-rose-900">Details: </span>
                  <pre className="mt-0.5 whitespace-pre-wrap text-[11px] max-h-32 overflow-y-auto">
                    {typeof details === 'object' ? JSON.stringify(details, null, 2) : String(details)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
