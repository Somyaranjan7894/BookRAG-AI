import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Router } from '@/router';
import { Dashboard } from '@/pages/Dashboard';
import { DocumentListResponse, DocumentUploadResponse } from '@/types/document';

const { mockDocApi } = vi.hoisted(() => {
  return {
    mockDocApi: {
      list: vi.fn(),
      upload: vi.fn(),
      get: vi.fn(),
      delete: vi.fn(),
    },
  };
});

// Mock documentsApi & documentApi
vi.mock('@/api/documents', () => ({
  documentsApi: mockDocApi,
  documentApi: mockDocApi,
}));

describe('Dashboard and Upload Component Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('1. Dashboard renders successfully with title and hero', async () => {
    const mockList: DocumentListResponse = {
      total: 1,
      documents: [
        {
          document_id: 'doc-123',
          filename: 'deep_learning.pdf',
          title: 'Deep Learning Textbook',
          author: 'Ian Goodfellow',
          page_count: 800,
          status: 'processed',
          processing_stage: 'completed',
          error_message: null,
          created_at: '2026-09-30T10:00:00Z',
          updated_at: '2026-09-30T10:05:00Z',
        },
      ],
    };

    mockDocApi.list.mockResolvedValue(mockList);

    render(
      <Router initialPath="/">
        <Dashboard />
      </Router>
    );

    // Verify hero and title
    expect(screen.getByText('Book Knowledge Base')).toBeInTheDocument();
    expect(screen.getByText(/Upload PDF books, observe background chunking/i)).toBeInTheDocument();

    // Verify book card rendered
    await waitFor(() => {
      expect(screen.getByText('Deep Learning Textbook')).toBeInTheDocument();
      expect(screen.getByText('by Ian Goodfellow')).toBeInTheDocument();
      expect(screen.getByText('800 pages')).toBeInTheDocument();
      expect(screen.getByText('Ready')).toBeInTheDocument();
    });
  });

  it('2. Empty library state renders when no books exist', async () => {
    const mockEmptyList: DocumentListResponse = {
      total: 0,
      documents: [],
    };

    mockDocApi.list.mockResolvedValue(mockEmptyList);

    render(
      <Router initialPath="/">
        <Dashboard />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByText('Your Library is Empty')).toBeInTheDocument();
      expect(
        screen.getByText(/Upload complete PDF books to start building your grounded semantic knowledge base/i)
      ).toBeInTheDocument();
      expect(screen.getByText('Upload First Book')).toBeInTheDocument();
    });
  });

  it('3. Upload validation rejects non-PDF input', async () => {
    const mockEmptyList: DocumentListResponse = {
      total: 0,
      documents: [],
    };
    mockDocApi.list.mockResolvedValue(mockEmptyList);

    render(
      <Router initialPath="/">
        <Dashboard />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-file-input')).toBeInTheDocument();
    });

    const fileInput = screen.getByTestId('pdf-file-input');

    // Create a non-PDF text file
    const invalidFile = new File(['hello world'], 'notes.txt', { type: 'text/plain' });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    await waitFor(() => {
      expect(
        screen.getByText(/Invalid file type "notes.txt". Only PDF documents \(.pdf\) are supported./i)
      ).toBeInTheDocument();
    });

    // Verify api was not called
    expect(mockDocApi.upload).not.toHaveBeenCalled();
  });

  it('4. PDF upload succeeds and sends multipart request to API', async () => {
    const mockList: DocumentListResponse = {
      total: 0,
      documents: [],
    };
    mockDocApi.list.mockResolvedValue(mockList);

    const uploadResponse: DocumentUploadResponse = {
      document_id: 'doc-async-999',
      task_id: 'celery-task-777',
      status: 'queued',
      message: 'PDF accepted for asynchronous processing.',
    };
    mockDocApi.upload.mockResolvedValue(uploadResponse);

    render(
      <Router initialPath="/">
        <Dashboard />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-file-input')).toBeInTheDocument();
    });

    const fileInput = screen.getByTestId('pdf-file-input');
    const validPdf = new File(['%PDF-1.4 mock content'], 'reinforcement_learning.pdf', {
      type: 'application/pdf',
    });

    fireEvent.change(fileInput, { target: { files: [validPdf] } });

    await waitFor(() => {
      expect(mockDocApi.upload).toHaveBeenCalledWith(validPdf);
    });
  });

  it('5. HTTP 202 processing response is handled with queued progress state', async () => {
    const mockList: DocumentListResponse = {
      total: 0,
      documents: [],
    };
    mockDocApi.list.mockResolvedValue(mockList);

    const uploadResponse: DocumentUploadResponse = {
      document_id: 'doc-async-999',
      task_id: 'celery-task-777',
      status: 'queued',
      message: 'PDF accepted for asynchronous processing.',
    };
    mockDocApi.upload.mockResolvedValue(uploadResponse);
    mockDocApi.get.mockResolvedValue({
      document_id: 'doc-async-999',
      filename: 'reinforcement_learning.pdf',
      title: 'Reinforcement Learning',
      status: 'queued',
      processing_stage: 'queued',
      page_count: 0,
      error_message: null,
      created_at: null,
      updated_at: null,
    });

    render(
      <Router initialPath="/">
        <Dashboard />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-file-input')).toBeInTheDocument();
    });

    const fileInput = screen.getByTestId('pdf-file-input');
    const validPdf = new File(['%PDF-1.4 mock content'], 'reinforcement_learning.pdf', {
      type: 'application/pdf',
    });

    fireEvent.change(fileInput, { target: { files: [validPdf] } });

    await waitFor(() => {
      expect(
        screen.getByText(/Upload Accepted — The book is being processed/i)
      ).toBeInTheDocument();
      expect(screen.getByText('HTTP 202')).toBeInTheDocument();
      expect(screen.getByText(/doc-async-999/)).toBeInTheDocument();
    });
  });
});
