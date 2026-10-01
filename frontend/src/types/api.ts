/**
 * Centralized API error contracts matching Phase 16 backend error structure.
 */

export interface ApiErrorDetail {
  loc?: (string | number)[];
  msg?: string;
  type?: string;
  ctx?: Record<string, string>;
  [key: string]: unknown;
}

export interface ApiErrorEnvelope {
  code: number;
  message: string;
  request_id?: string;
  error_type?: string;
  details?: ApiErrorDetail[] | string | Record<string, unknown>;
}

export interface BackendErrorResponse {
  error: ApiErrorEnvelope;
}

export class ApiError extends Error {
  public readonly code: number;
  public readonly requestId?: string;
  public readonly request_id?: string;
  public readonly errorType?: string;
  public readonly error_type?: string;
  public readonly details?: ApiErrorDetail[] | string | Record<string, unknown>;

  constructor(envelope: ApiErrorEnvelope) {
    super(envelope.message || 'An unexpected API error occurred');
    this.name = 'ApiError';
    this.code = envelope.code;
    this.requestId = envelope.request_id;
    this.request_id = envelope.request_id;
    this.errorType = envelope.error_type;
    this.error_type = envelope.error_type;
    this.details = envelope.details;
  }
}
