import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MatchingBoard } from '@/components/matching/MatchingBoard';
import { AnswerDisplay } from '@/components/qa/AnswerDisplay';
import { BookDetail } from '@/pages/BookDetail';
import { Router } from '@/router';
import { SearchResult } from '@/types/search';
import { GenerationEvidenceItem, GroundedAnswerResponse, Citation } from '@/types/qa';
import { PersistedDocument } from '@/types/document';
import { ApiError } from '@/types/api';

const { mockDocApi, mockQaApi, mockSearchApi } = vi.hoisted(() => ({
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
  mockSearchApi: {
    search: vi.fn(),
  },
}));

vi.mock('@/api/documents', () => ({
  documentsApi: mockDocApi,
  documentApi: mockDocApi,
}));

vi.mock('@/api/qa', () => ({
  qaApi: mockQaApi,
}));

vi.mock('@/api/search', () => ({
  searchApi: mockSearchApi,
}));

describe('Phase 18 — Matching Board & Retrieval Transparency Tests', () => {
  const mockResults: SearchResult[] = [
    {
      rank: 1,
      original_rank: 7,
      chunk_id: 'chunk_alpha_001',
      document_id: 'deep_learning_handbook',
      page_number: 42,
      text: 'Backpropagation computes the gradient of the loss function with respect to weights using the multivariate chain rule across layers.',
      similarity_score: 0.8245,
      reranker_score: 5.8123,
      chunk_index: 3,
    },
    {
      rank: 2,
      original_rank: 1,
      chunk_id: 'chunk_beta_002',
      document_id: 'deep_learning_handbook',
      page_number: 12,
      text: 'Stochastic gradient descent updates network parameters iteratively using small minibatches.',
      similarity_score: 0.912,
      reranker_score: 3.4567,
      chunk_index: 1,
    },
    {
      rank: 3,
      original_rank: 3,
      chunk_id: 'chunk_gamma_003',
      document_id: 'deep_learning_handbook',
      page_number: 88,
      text: 'Learning rate warmup stabilizes transformer training dynamics during early gradient updates.',
      similarity_score: 0.7654,
      reranker_score: 2.1098,
      chunk_index: 8,
    },
    {
      rank: 4,
      original_rank: null,
      chunk_id: 'chunk_delta_004',
      document_id: 'deep_learning_handbook',
      page_number: 104,
      text: 'Momentum accelerates stochastic gradient descent by damping oscillations in steep ravines.',
      similarity_score: 0.7012,
      reranker_score: null,
      chunk_index: 10,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  // 1. Matching Board renders query
  it('1. Matching Board renders submitted query', () => {
    render(
      <MatchingBoard
        query="How does backpropagation calculate gradients?"
        results={mockResults}
        candidateCount={20}
      />
    );

    expect(screen.getByText('Submitted Query')).toBeInTheDocument();
    expect(screen.getByText('How does backpropagation calculate gradients?')).toBeInTheDocument();
  });

  // 2. Candidate count renders
  it('2. Candidate count renders from candidate pool metadata', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={25}
      />
    );

    expect(screen.getByText('25')).toBeInTheDocument();
    expect(screen.getByText('candidates retrieved')).toBeInTheDocument();
    expect(screen.getByText('Candidate Pool')).toBeInTheDocument();
  });

  // 3. Evidence count renders
  it('3. Evidence count renders for selected evaluated passages', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    expect(screen.getByText('Selected Evidence')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText('passages selected')).toBeInTheDocument();
  });

  // 4. Result rank renders
  it('4. Result final rank renders on card', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    expect(screen.getByText('#1')).toBeInTheDocument();
    expect(screen.getByText('#2')).toBeInTheDocument();
    expect(screen.getByText('#3')).toBeInTheDocument();
    expect(screen.getByText('#4')).toBeInTheDocument();
  });

  // 5. Original rank renders
  it('5. Original first-stage rank renders on card', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    expect(screen.getByText('Original #7')).toBeInTheDocument();
    expect(screen.getByText('Original #1')).toBeInTheDocument();
    expect(screen.getByText('Original #3')).toBeInTheDocument();
  });

  // 6. Page number renders
  it('6. Page number renders for provenance tracking', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    expect(screen.getByText('Page 42')).toBeInTheDocument();
    expect(screen.getByText('Page 12')).toBeInTheDocument();
    expect(screen.getByText('Page 88')).toBeInTheDocument();
    expect(screen.getByText('Page 104')).toBeInTheDocument();
  });

  // 7. Semantic score renders
  it('7. Semantic relevance score renders with neutral decimal representation', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    // Verify label and calibrated decimal score
    const semanticLabels = screen.getAllByText('Semantic relevance');
    expect(semanticLabels.length).toBeGreaterThan(0);
    expect(screen.getByText('0.8245')).toBeInTheDocument();
    expect(screen.getByText('0.9120')).toBeInTheDocument();
  });

  // 8. Reranker score renders
  it('8. Reranker relevance score renders with Cross-Encoder label', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    const rerankerLabels = screen.getAllByText('Reranker relevance');
    expect(rerankerLabels.length).toBeGreaterThan(0);
    expect(screen.getByText('5.8123')).toBeInTheDocument();
    expect(screen.getByText('3.4567')).toBeInTheDocument();
  });

  // 9. Source passage renders
  it('9. Source passage snippet renders in evidence block', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    expect(
      screen.getByText(/Backpropagation computes the gradient of the loss function/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Stochastic gradient descent updates network parameters/i)
    ).toBeInTheDocument();
  });

  // 10. Expand/collapse works
  it('10. Expand and collapse works on long passage text', () => {
    const longPassage =
      'This is an extensive technical passage detailing the mathematical formulation of backpropagation across multi-layer perceptrons. '.repeat(
        4
      );

    const longResult: SearchResult[] = [
      {
        rank: 1,
        original_rank: 5,
        chunk_id: 'chunk_long_001',
        document_id: 'doc_1',
        page_number: 15,
        text: longPassage,
        similarity_score: 0.85,
        reranker_score: 4.12,
        chunk_index: 0,
      },
    ];

    render(
      <MatchingBoard
        query="test query"
        results={longResult}
        candidateCount={10}
      />
    );

    const expandBtn = screen.getByRole('button', { name: /Show full passage/i });
    expect(expandBtn).toBeInTheDocument();
    expect(expandBtn).toHaveAttribute('aria-expanded', 'false');

    // Click to expand
    fireEvent.click(expandBtn);
    expect(screen.getByRole('button', { name: /Show less/i })).toHaveAttribute('aria-expanded', 'true');

    // Click to collapse
    fireEvent.click(screen.getByRole('button', { name: /Show less/i }));
    expect(screen.getByRole('button', { name: /Show full passage/i })).toHaveAttribute('aria-expanded', 'false');
  });

  // 11. Ranking movement is calculated correctly: positive movement (Original #7 -> Final #1)
  it('11. Ranking movement calculates upward movement correctly (↑ positions)', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    // Original #7 -> Final #1 => delta = 6 positions upward
    expect(screen.getByText('↑ 6 positions')).toBeInTheDocument();
  });

  // 12. Negative/downward movement is handled: Original #1 -> Final #2
  it('12. Negative downward ranking movement is handled correctly (↓ positions)', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    // Original #1 -> Final #2 => delta = -1 position downward
    expect(screen.getByText('↓ 1 position')).toBeInTheDocument();
  });

  // 13. Unchanged rank is handled: Original #3 -> Final #3
  it('13. Unchanged rank is handled correctly (— Unchanged)', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    // Original #3 -> Final #3 => delta = 0
    expect(screen.getByText('— Unchanged')).toBeInTheDocument();
  });

  // 14. Missing optional reranker score is handled gracefully
  it('14. Missing optional reranker score is handled gracefully without crash', () => {
    render(
      <MatchingBoard
        query="gradient descent"
        results={mockResults}
        candidateCount={20}
      />
    );

    // Fourth card has reranker_score = null
    expect(screen.getByText('Not reranked')).toBeInTheDocument();
    expect(screen.getByText('No initial rank')).toBeInTheDocument();
  });

  // 15. "View Matching Board" integration works from AnswerDisplay
  it('15. "View Matching Board" action in AnswerDisplay opens the modal transparency layer', () => {
    const mockEvidence: GenerationEvidenceItem[] = [
      {
        rank: 1,
        original_rank: 5,
        chunk_id: 'chunk_alpha_001',
        document_id: 'deep_learning_handbook',
        page_number: 42,
        chunk_index: 3,
        source_text: 'Backpropagation computes the gradient of the loss function.',
        similarity_score: 0.8245,
        reranker_score: 5.8123,
      },
    ];

    const mockCitations: Citation[] = [
      {
        citation_id: 'cite_1',
        document_id: 'deep_learning_handbook',
        chunk_id: 'chunk_alpha_001',
        page_number: 42,
        chunk_index: 3,
        source_text: 'Backpropagation computes the gradient of the loss function.',
      },
    ];

    const mockGroundedResponse: GroundedAnswerResponse = {
      query: 'What is backpropagation?',
      answer: 'Backpropagation calculates gradients of the loss function.',
      answerable: true,
      grounded: true,
      groundedness_score: 1.0,
      grounding_status: 'grounded',
      claims: [],
      evidence: mockEvidence,
      citations: mockCitations,
      candidate_count: 20,
      reranking_applied: true,
    };

    render(
      <AnswerDisplay
        query="What is backpropagation?"
        groundedResult={mockGroundedResponse}
        extractiveResult={null}
      />
    );

    // Button should be visible
    const viewBoardBtn = screen.getByRole('button', { name: /View Matching Board/i });
    expect(viewBoardBtn).toBeInTheDocument();

    // Click button to open modal
    fireEvent.click(viewBoardBtn);

    // Modal dialog should now be in document
    const dialog = screen.getByRole('dialog');
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText('Relevant Matching Board')).toBeInTheDocument();
    expect(screen.getByText('20')).toBeInTheDocument();
    expect(screen.getByText('candidates retrieved')).toBeInTheDocument();
    expect(screen.getByText('↑ 4 positions')).toBeInTheDocument();
    expect(screen.getByText('[cite_1]')).toBeInTheDocument();

    // Close modal
    const closeBtn = screen.getByRole('button', { name: /Close Matching Board modal/i });
    fireEvent.click(closeBtn);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  // 16. API error is handled safely
  it('16. Standalone Matching Board handles API search error safely in BookDetail', async () => {
    const mockDoc: PersistedDocument = {
      document_id: 'book-ai-101',
      filename: 'ai_handbook.pdf',
      title: 'AI Handbook',
      author: 'Russell',
      page_count: 500,
      status: 'processed',
      processing_stage: 'completed',
      error_message: null,
      created_at: '2026-09-30T10:00:00Z',
      updated_at: '2026-09-30T10:15:00Z',
    };

    mockDocApi.get.mockResolvedValue(mockDoc);
    mockDocApi.getPages.mockResolvedValue([]);
    mockDocApi.getChunks.mockResolvedValue([]);
    mockSearchApi.search.mockRejectedValue(
      new ApiError({
        code: 500,
        message: 'Search vector index temporarily unavailable.',
        error_type: 'SEARCH_FAILED',
        request_id: 'req_search_err_789',
      })
    );

    render(
      <Router initialPath="/books/book-ai-101">
        <BookDetail />
      </Router>
    );

    await waitFor(() => {
      expect(screen.getByText('AI Handbook')).toBeInTheDocument();
    });

    // Switch to Matching Board mode
    const matchingTab = screen.getByRole('button', { name: /Matching Board/i });
    fireEvent.click(matchingTab);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Search semantic candidates/i)).toBeInTheDocument();
    });

    // Submit a search query
    const textarea = screen.getByPlaceholderText(/Search semantic candidates/i);
    fireEvent.change(textarea, { target: { value: 'quantum neural networks' } });
    expect((textarea as HTMLTextAreaElement).value).toBe('quantum neural networks');

    const submitBtn = screen.getByRole('button', { name: /Ask Question/i });
    expect(submitBtn).not.toBeDisabled();
    fireEvent.click(submitBtn);

    // Verify error is displayed safely
    await waitFor(() => {
      expect(screen.getByText('Failed to Search Book')).toBeInTheDocument();
      expect(screen.getByText('Search vector index temporarily unavailable.')).toBeInTheDocument();
    });

    // Expand technical details and verify request ID
    const detailsBtn = screen.getByRole('button', { name: /View technical details/i });
    fireEvent.click(detailsBtn);
    expect(screen.getByText(/req_search_err_789/i)).toBeInTheDocument();
  });
});
