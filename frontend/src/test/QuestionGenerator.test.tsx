import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QuestionGenerator } from '@/components/questions/QuestionGenerator';
import { questionsApi } from '@/api/questions';
import { QuestionGenerationResponse } from '@/types/questionGeneration';

vi.mock('@/api/questions', () => ({
  questionsApi: {
    generateQuestions: vi.fn(),
  },
}));

describe('QuestionGenerator Frontend Component Tests', () => {
  const mockResponse: QuestionGenerationResponse = {
    document_id: 'doc-ai-101',
    requested_count: 3,
    generated_candidates: 6,
    validated_count: 2,
    returned_count: 2,
    latency_ms: 185.4,
    questions: [
      {
        question_id: 'qgen_doc-ai-101_c1_001',
        question: 'What is meant by gradient descent?',
        answer: 'an optimization algorithm to minimize loss',
        question_type: 'definition',
        difficulty: 'medium',
        document_id: 'doc-ai-101',
        chunk_id: 'c1',
        chunk_ids: ['c1'],
        page_number: 4,
        page_numbers: [4],
        source_text: 'Gradient descent is defined as an optimization algorithm to minimize loss.',
        validation_status: 'validated',
        qa_predicted_answer: 'an optimization algorithm to minimize loss',
        qa_confidence_score: 0.94,
      },
      {
        question_id: 'qgen_doc-ai-101_c2_002',
        question: 'How do CNNs compare with Transformers across pages?',
        answer: 'Transformers process global attention while CNNs process local features',
        question_type: 'multi_page_synthesis',
        difficulty: 'hard',
        document_id: 'doc-ai-101',
        chunk_id: 'c2',
        chunk_ids: ['c2', 'c3'],
        page_number: 8,
        page_numbers: [8, 9],
        source_text: 'Page 8 discusses CNNs while Page 9 discusses Transformers.',
        validation_status: 'validated',
        qa_predicted_answer: 'Transformers process global attention while CNNs process local features',
        qa_confidence_score: 0.88,
      },
    ],
    rejected_candidates: [
      {
        candidate_id: 'rej_1',
        question_text: 'What is this?',
        answer_text: 'loss',
        rejection_reason: 'Question is too vague or generic to be answerable.',
        rejection_category: 'quality_failure',
        chunk_id: 'c1',
        chunk_ids: ['c1'],
        page_number: 4,
        page_numbers: [4],
        question_type: 'what',
      },
    ],
    rejection_summary: {
      quality_failure: 1,
      duplicate: 2,
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('1. Renders empty initial state with controls and button', () => {
    render(
      <QuestionGenerator
        documentId="doc-ai-101"
        bookTitle="Artificial Intelligence Handbook"
      />
    );

    expect(screen.getByText(/Generate Reading Comprehension Questions/i)).toBeInTheDocument();
    expect(screen.getByText(/No questions generated yet/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Generate Questions/i })).toBeInTheDocument();
  });

  it('2. Clicking generate triggers API and renders validated questions with provenance', async () => {
    (questionsApi.generateQuestions as any).mockResolvedValue(mockResponse);

    render(
      <QuestionGenerator
        documentId="doc-ai-101"
        bookTitle="Artificial Intelligence Handbook"
      />
    );

    const button = screen.getByRole('button', { name: /Generate Questions/i });
    fireEvent.click(button);

    await waitFor(() => {
      expect(questionsApi.generateQuestions).toHaveBeenCalledWith({
        document_id: 'doc-ai-101',
        count: 5,
        question_type: undefined,
        difficulty: undefined,
        include_rejected: false,
        ensure_diversity: true,
      });
    });

    await waitFor(() => {
      expect(screen.getByText('What is meant by gradient descent?')).toBeInTheDocument();
      expect(screen.getByText('an optimization algorithm to minimize loss')).toBeInTheDocument();
      expect(screen.getByText('Page 4')).toBeInTheDocument();
      expect(screen.getByText('Pages 8, 9')).toBeInTheDocument();
      expect(screen.getByText('185ms')).toBeInTheDocument();
    });
  });

  it('3. Inspects and toggles source evidence passage', async () => {
    (questionsApi.generateQuestions as any).mockResolvedValue(mockResponse);

    render(
      <QuestionGenerator
        documentId="doc-ai-101"
        bookTitle="Artificial Intelligence Handbook"
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /Generate Questions/i }));

    await waitFor(() => {
      expect(screen.getByText('What is meant by gradient descent?')).toBeInTheDocument();
    });

    const toggleButton = screen.getAllByRole('button', { name: /Inspect Source Book Evidence/i })[0];
    fireEvent.click(toggleButton);

    expect(screen.getByText(/Gradient descent is defined as an optimization algorithm to minimize loss./i)).toBeInTheDocument();
  });

  it('4. Renders diagnostic rejections when show diagnostics is enabled', async () => {
    (questionsApi.generateQuestions as any).mockResolvedValue(mockResponse);

    render(
      <QuestionGenerator
        documentId="doc-ai-101"
        bookTitle="Artificial Intelligence Handbook"
      />
    );

    const diagCheckbox = screen.getByLabelText(/Show Diagnostics/i);
    fireEvent.click(diagCheckbox);

    fireEvent.click(screen.getByRole('button', { name: /Generate Questions/i }));

    await waitFor(() => {
      expect(questionsApi.generateQuestions).toHaveBeenCalledWith(
        expect.objectContaining({
          include_rejected: true,
        })
      );
    });

    await waitFor(() => {
      expect(screen.getByText(/Diagnostic Rejections/i)).toBeInTheDocument();
      expect(screen.getByText('What is this?')).toBeInTheDocument();
      expect(screen.getByText(/Question is too vague/i)).toBeInTheDocument();
    });
  });

  it('5. Handles API error gracefully', async () => {
    (questionsApi.generateQuestions as any).mockRejectedValue(new Error('Network error during generation'));

    render(
      <QuestionGenerator
        documentId="doc-ai-101"
        bookTitle="Artificial Intelligence Handbook"
      />
    );

    fireEvent.click(screen.getByRole('button', { name: /Generate Questions/i }));

    await waitFor(() => {
      expect(screen.getByText(/Network error during generation/i)).toBeInTheDocument();
    });
  });
});
