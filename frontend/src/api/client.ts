import { ApiError, BackendErrorResponse } from '@/types/api';

/**
 * Configurable base URL.
 * Defaults to empty string to leverage the Vite reverse proxy in development,
 * or can be set via VITE_API_BASE_URL for dedicated deployments.
 */
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

/**
 * Central HTTP client with error sanitization and structured ApiError mapping.
 */
export async function apiClient<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = 600000, headers = {}, ...rest } = options;


  const url = `${API_BASE_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...rest,
      headers: {
        Accept: 'application/json',
        ...headers,
      },
      signal: options.signal || controller.signal,
    });

    clearTimeout(timeoutId);

    // Parse response body
    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');
    const data = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      if (isJson && data && typeof data === 'object' && 'error' in data) {
        const backendError = data as BackendErrorResponse;
        throw new ApiError(backendError.error);
      }

      // Fallback for non-standard error responses
      throw new ApiError({
        code: response.status,
        message: typeof data === 'string' && data ? data : response.statusText || 'Request failed',
        request_id: response.headers.get('x-request-id') || undefined,
        error_type: `HTTP_${response.status}`,
      });
    }

    return data as T;
  } catch (error) {
    clearTimeout(timeoutId);

    if (error instanceof ApiError) {
      throw error;
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError({
        code: 408,
        message: 'Request timed out after waiting for server response.',
        error_type: 'REQUEST_TIMEOUT',
      });
    }

    // Network error or unexpected exception
    throw new ApiError({
      code: 0,
      message: error instanceof Error ? error.message : 'Network error or unreachable backend.',
      error_type: 'NETWORK_ERROR',
    });
  }
}
