import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { Router } from '@/router';
import { ProcessingProgress } from '@/components/processing/ProcessingProgress';
import { PersistedDocument } from '@/types/document';

const { mockDocApi } = vi.hoisted(() => {
  return {
    mockDocApi: {
      get: vi.fn(),
      list: vi.fn(),
      upload: vi.fn(),
      delete: vi.fn(),
    },
  };
});

vi.mock('@/api/documents', () => ({
  documentsApi: mockDocApi,
  documentApi: mockDocApi,
}));

describe('Processing Status Polling Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('6. Status polling starts correctly and tracks document stage', async () => {
    const queuedDoc: PersistedDocument = {
      document_id: 'doc-poll-start',
      filename: 'linear_algebra.pdf',
      title: 'Linear Algebra Done Right',
      author: 'Sheldon Axler',
      page_count: 0,
      status: 'queued',
      processing_stage: 'queued',
      error_message: null,
      created_at: '2026-09-30T12:00:00Z',
      updated_at: null,
    };

    const processingDoc: PersistedDocument = {
      ...queuedDoc,
      status: 'processing',
      processing_stage: 'parsing',
      page_count: 340,
    };

    mockDocApi.get.mockResolvedValueOnce(queuedDoc).mockResolvedValueOnce(processingDoc);

    render(
      <Router initialPath="/">
        <ProcessingProgress
          documentId="doc-poll-start"
          filename="linear_algebra.pdf"
          pollingIntervalMs={50}
        />
      </Router>
    );

    expect(screen.getByText('linear_algebra.pdf')).toBeInTheDocument();

    await waitFor(() => {
      expect(mockDocApi.get).toHaveBeenCalledWith('doc-poll-start');
      expect(screen.getByText(/Extracting text and structure from PDF pages/i)).toBeInTheDocument();
    });
  });

  it('7. Polling stops after processed terminal state is reached', async () => {
    const queuedDoc: PersistedDocument = {
      document_id: 'doc-poll-1',
      filename: 'pattern_recognition.pdf',
      title: 'PRML',
      author: 'Christopher Bishop',
      page_count: 0,
      status: 'queued',
      processing_stage: 'queued',
      error_message: null,
      created_at: '2026-09-30T12:00:00Z',
      updated_at: null,
    };

    const processingDoc: PersistedDocument = {
      ...queuedDoc,
      status: 'processing',
      processing_stage: 'indexing',
      page_count: 738,
    };

    const completedDoc: PersistedDocument = {
      ...queuedDoc,
      status: 'processed',
      processing_stage: 'completed',
      page_count: 738,
    };

    // First call returns queued, second call returns indexing, third call returns processed
    mockDocApi.get
      .mockResolvedValueOnce(queuedDoc)
      .mockResolvedValueOnce(processingDoc)
      .mockResolvedValue(completedDoc);

    const onFinished = vi.fn();

    render(
      <Router initialPath="/">
        <ProcessingProgress
          documentId="doc-poll-1"
          filename="pattern_recognition.pdf"
          pollingIntervalMs={50}
          onFinished={onFinished}
        />
      </Router>
    );

    // Initial state
    expect(screen.getByText('pattern_recognition.pdf')).toBeInTheDocument();

    // Eventually completes and renders the Open Book action
    await waitFor(
      () => {
        expect(screen.getByText('Open Book')).toBeInTheDocument();
        expect(
          screen.getByText('Document fully indexed and ready for grounded retrieval!')
        ).toBeInTheDocument();
        expect(onFinished).toHaveBeenCalled();
      },
      { timeout: 3000 }
    );

    const callCountAfterComplete = mockDocApi.get.mock.calls.length;
    // Wait past an interval to verify polling has stopped
    await new Promise((r) => setTimeout(r, 120));
    expect(mockDocApi.get.mock.calls.length).toBe(callCountAfterComplete);
  });

  it('8. Polling stops after failed terminal state', async () => {
    const failedDoc: PersistedDocument = {
      document_id: 'doc-fail-1',
      filename: 'corrupted_file.pdf',
      title: null,
      author: null,
      page_count: 0,
      status: 'failed',
      processing_stage: 'failed',
      error_message: 'PDF page extraction failed: Invalid PDF xref table or encrypted structure.',
      created_at: '2026-09-30T12:00:00Z',
      updated_at: null,
    };

    mockDocApi.get.mockResolvedValue(failedDoc);

    render(
      <Router initialPath="/">
        <ProcessingProgress
          documentId="doc-fail-1"
          filename="corrupted_file.pdf"
          pollingIntervalMs={50}
        />
      </Router>
    );

    await waitFor(
      () => {
        expect(screen.getByText(/Processing Failed:/i)).toBeInTheDocument();
        expect(
          screen.getAllByText(/Invalid PDF xref table or encrypted structure/i).length
        ).toBeGreaterThan(0);
      },
      { timeout: 3000 }
    );

    const callCountAfterFail = mockDocApi.get.mock.calls.length;
    await new Promise((r) => setTimeout(r, 120));
    expect(mockDocApi.get.mock.calls.length).toBe(callCountAfterFail);
  });
});
