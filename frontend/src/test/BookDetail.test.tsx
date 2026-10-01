import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { Router } from '@/router';
import { BookDetail } from '@/pages/BookDetail';
import { qaApi } from '@/api/qa';
import { PersistedDocument } from '@/types/document';
import { GroundedAnswerResponse } from '@/types/qa';

const { mockDocApi, mockQaApi } = vi.hoisted(() => {
  return {
    mockDocApi: {
      get: vi.fn(),
      getPages: vi.fn(),
      getChunks: vi.fn(),
      delete: vi.fn(),
    },
    mockQaApi: {
      askGroundedQuestion: vi.fn(),
      askExtractiveQuestion: vi.fn(),
    },
  };
});

vi.mock('@/api/documents', () => ({
  documentsApi: mockDocApi,
  documentApi: mockDocApi,
}));

vi.mock('@/api/qa', () => ({
  qaApi: mockQaApi,
}));

describe('Book Detail and Question Answering Tests', () => {
  const mockDoc: PersistedDocument = {
    document_id: 'book-ai-101',
    filename: 'artificial_intelligence.pdf',
    title: 'Artificial Intelligence: A Modern Approach',
    author: 'Stuart Russell and Peter Norvig',
    page_count: 1152,
    status: 'processed',
    processing_stage: 'completed',
    error_message: null,
    created_at: '2026-09-30T10:00:00Z',
    updated_at: '2026-09-30T10:15:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockDocApi.get.mockResolvedValue(mockDoc);
    mockDocApi.getPages.mockResolvedValue([]);
    mockDocApi.getChunks.mockResolvedValue([]);
  });

  it('9. Book detail renders title, metadata, and QA controls', async () => {
    render(
      <Router initialPath="/books/book-ai-101">
        <BookDetail />
      </Router>
    );

    await waitFor(() => {
      expect(
        screen.getByText('Artificial Intelligence: A Modern Approach')
      ).toBeInTheDocument();
      expect(screen.getByText('Author: Stuart Russell and Peter Norvig')).toBeInTheDocument();
      expect(screen.getByText('1152 Pages')).toBeInTheDocument();
      expect(screen.getByText('Ask This Book')).toBeInTheDocument();
      expect(screen.getByText('Grounded Synthesis (NLI)')).toBeInTheDocument();
    });
  });

  it('10. Empty question is rejected before submission', async () => {
    render(
      <Router initialPath="/books/book-ai-101">
        <BookDetail />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByText('Ask This Book')).toBeInTheDocument();
    });

    const submitBtn = screen.getByRole('button', { name: /ask/i });
    // The button is disabled when empty, and clicking it does not call API
    expect(submitBtn).toBeDisabled();

    fireEvent.click(submitBtn);
    expect(qaApi.askGroundedQuestion).not.toHaveBeenCalled();
  });

  it('11. Question loading state appears on submission', async () => {
    let resolveQA: (val: any) => void;
    const qaPromise = new Promise((resolve) => {
      resolveQA = resolve;
    });

    mockQaApi.askGroundedQuestion.mockReturnValue(qaPromise);

    render(
      <Router initialPath="/books/book-ai-101">
        <BookDetail />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Ask a question about/i)).toBeInTheDocument();
    });

    const textarea = screen.getByPlaceholderText(/Ask a question about/i);
    fireEvent.change(textarea, { target: { value: 'What is an admissible heuristic in A* search?' } });
    fireEvent.submit(textarea.closest('form')!);

    // Verify loading indicator appears
    await waitFor(() => {
      expect(
        screen.getByText(/Searching semantic chunks & verifying claims/i)
      ).toBeInTheDocument();
    });

    expect(mockQaApi.askGroundedQuestion).toHaveBeenCalledWith(
      expect.objectContaining({
        query: 'What is an admissible heuristic in A* search?',
        document_id: 'book-ai-101',
      })
    );

    // Clean up pending promise
    resolveQA!({
      query: 'What is an admissible heuristic in A* search?',
      answer: 'Done',
      answerable: true,
      grounded: true,
      groundedness_score: 1.0,
      grounding_status: 'grounded',
      claims: [],
      evidence: [],
      citations: [],
    });

    await waitFor(() => {
      expect(screen.getByText('Done')).toBeInTheDocument();
    });
  });

  it('12. Successful answer renders from backend response', async () => {
    const mockAnswer: GroundedAnswerResponse = {
      query: 'Define Alpha-Beta pruning',
      answer: 'Alpha-Beta pruning is an optimization technique for the minimax algorithm that eliminates branches that cannot influence the final decision.',
      answerable: true,
      grounded: true,
      groundedness_score: 0.95,
      grounding_status: 'grounded',
      claims: [],
      evidence: [],
      citations: [],
      model_name: 'test-nli-model',
    };

    mockQaApi.askGroundedQuestion.mockResolvedValue(mockAnswer);

    render(
      <Router initialPath="/books/book-ai-101">
        <BookDetail />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Ask a question about/i)).toBeInTheDocument();
    });

    const textarea = screen.getByPlaceholderText(/Ask a question about/i);
    fireEvent.change(textarea, { target: { value: 'Define Alpha-Beta pruning' } });
    fireEvent.submit(textarea.closest('form')!);

    await waitFor(() => {
      expect(
        screen.getByText(/Alpha-Beta pruning is an optimization technique for the minimax algorithm/i)
      ).toBeInTheDocument();
      expect(screen.getByText('(95%)')).toBeInTheDocument();
      expect(screen.getByText('grounded')).toBeInTheDocument();
    });
  });

  it('13. Evidence and citations render with page numbers and text provenance', async () => {
    const mockAnswerWithCitations: GroundedAnswerResponse = {
      query: 'Define Alpha-Beta pruning',
      answer: 'Alpha-Beta pruning prunes branches.',
      answerable: true,
      grounded: true,
      groundedness_score: 0.95,
      grounding_status: 'grounded',
      claims: [],
      evidence: [
        {
          rank: 1,
          chunk_id: 'chunk-ab-1',
          document_id: 'book-ai-101',
          page_number: 167,
          chunk_index: 4,
          source_text: 'Alpha-beta pruning returns the same move as minimax, but prunes away branches that cannot possibly influence the final decision.',
          similarity_score: 0.89,
          reranker_score: 0.96,
        },
      ],
      citations: [
        {
          citation_id: 'cite-ab-1',
          document_id: 'book-ai-101',
          chunk_id: 'chunk-ab-1',
          page_number: 167,
          chunk_index: 4,
          source_text: 'Alpha-beta pruning returns the same move as minimax...',
          reranker_score: 0.96,
        },
      ],
      model_name: 'test-nli-model',
    };

    mockQaApi.askGroundedQuestion.mockResolvedValue(mockAnswerWithCitations);

    render(
      <Router initialPath="/books/book-ai-101">
        <BookDetail />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Ask a question about/i)).toBeInTheDocument();
    });

    const textarea = screen.getByPlaceholderText(/Ask a question about/i);
    fireEvent.change(textarea, { target: { value: 'Define Alpha-Beta pruning' } });
    fireEvent.submit(textarea.closest('form')!);

    await waitFor(() => {
      // Verify citations and evidence rendered with page number and chunk
      const pageTags = screen.getAllByText(/Page 167/i);
      expect(pageTags.length).toBeGreaterThanOrEqual(1);
      const chunkTags = screen.getAllByText(/Chunk #4/i);
      expect(chunkTags.length).toBeGreaterThanOrEqual(1);

      // Verify evidence card rendered
      expect(screen.getByText(/GROUNDING PROVENANCE & RETRIEVAL CONTEXT/i)).toBeInTheDocument();
      const evidencePassages = screen.getAllByText(/Alpha-beta pruning returns the same move as minimax/i);
      expect(evidencePassages.length).toBe(2); // One in citations, one in evidence cards
    });
  });
});
