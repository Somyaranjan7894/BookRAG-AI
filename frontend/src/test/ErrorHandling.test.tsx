import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ErrorAlert } from '@/components/common/ErrorAlert';
import { ApiError } from '@/types/api';

describe('Error Handling and Sanitization Tests', () => {
  it('14. Backend error is rendered safely without exposing stack traces, preserving request ID', () => {
    const backendError = new ApiError({
      code: 422,
      message: 'Document processing worker rejected payload: Unsupported PDF encryption.',
      error_type: 'UNPROCESSABLE_DOCUMENT',
      request_id: 'req-prod-987654321',
      details: [
        {
          loc: ['body', 'password'],
          msg: 'Password-protected PDFs cannot be indexed without a key',
          type: 'value_error',
        },
      ],
    });

    render(<ErrorAlert error={backendError} />);

    // Verify user-friendly title and sanitized message
    expect(screen.getByText('Validation Error')).toBeInTheDocument();
    expect(
      screen.getByText('Document processing worker rejected payload: Unsupported PDF encryption.')
    ).toBeInTheDocument();

    // Verify stack traces are NOT displayed anywhere
    expect(screen.queryByText(/Traceback/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/File "/i)).not.toBeInTheDocument();

    // Toggle technical details to see Request ID
    const toggleButton = screen.getByText(/View technical details/i);
    expect(toggleButton).toBeInTheDocument();
    fireEvent.click(toggleButton);

    // Verify request ID is displayed for debugging
    expect(screen.getByText('req-prod-987654321')).toBeInTheDocument();
    expect(screen.getByText('UNPROCESSABLE_DOCUMENT')).toBeInTheDocument();
  });

  it('15. Network error renders safely with clear messaging and retry action', () => {
    const networkError = new ApiError({
      code: 0,
      message: 'Network error or unreachable backend.',
      error_type: 'NETWORK_ERROR',
    });

    const onRetry = vi.fn();
    render(<ErrorAlert error={networkError} onRetry={onRetry} />);

    // Verify user-friendly title
    expect(screen.getByText('Network Connection Error')).toBeInTheDocument();
    expect(screen.getByText('Network error or unreachable backend.')).toBeInTheDocument();

    // Verify retry button exists and triggers callback
    const retryBtn = screen.getByRole('button', { name: /try again/i });
    expect(retryBtn).toBeInTheDocument();
    fireEvent.click(retryBtn);
    expect(onRetry).toHaveBeenCalledTimes(1);

    // Verify stack traces are not present
    expect(screen.queryByText(/Traceback/i)).not.toBeInTheDocument();
  });
});

