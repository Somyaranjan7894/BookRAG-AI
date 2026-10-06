"""Sample technical document fixture for reproducible evaluation.

Defines the text, chunks, and metadata for 'Foundations of Machine Learning & Neural Networks'
(document_id: 'doc_ai_handbook_01') corresponding to the dev_sample.json benchmark.
"""

from typing import List
from app.schemas.chunk import Chunk


DOCUMENT_ID = "doc_ai_handbook_01"
DOCUMENT_TITLE = "Foundations of Machine Learning & Neural Networks"
DOCUMENT_AUTHOR = "BookRAG AI Research Benchmark"

SAMPLE_PAGES = [
    (
        1,
        "Supervised learning algorithms optimize parametric predictive models using explicit pairs of input features and ground-truth target labels. Common optimization methods include stochastic gradient descent and adaptive learning rate schedules such as Adam. In contrast, unsupervised learning algorithms discover latent structures, cluster partitions, and manifold representations without external supervisory guidance or target labels."
    ),
    (
        2,
        "The backpropagation algorithm computes the exact mathematical gradients of a scalar loss function with respect to all trainable network weights by recursively applying the multivariate chain rule across the nodes of a computational graph. Errors are propagated backward layer by layer from output activations to input features, enabling efficient reverse-mode automatic differentiation in deep neural architectures."
    ),
    (
        3,
        "Non-linear activation functions enable deep neural networks to approximate complex mathematical functions. The Rectified Linear Unit (ReLU) activation function mitigates the vanishing gradient problem in deep multilayer perceptrons because its derivative is a constant 1 for all positive inputs, preventing the exponential gradient decay observed in saturating activation functions such as sigmoid and hyperbolic tangent."
    ),
    (
        4,
        "Convolutional Neural Networks (CNNs) exploit the local spatial geometry and translation invariance of image grids through weight sharing and localized receptive field convolution kernels. Subsampling and pooling layers progressively compress spatial dimensions while hierarchically expanding channel feature capacity from low-level edges to semantic object representations."
    ),
    (
        5,
        "The transformer architecture relies on scaled dot-product attention operating over query, key, and value vectors. The dot products between query and key vectors are explicitly divided by the square root of the key dimension to counteract exponential magnitude growth in high-dimensional vector spaces, which would otherwise push the softmax function into regions with near-zero vanishing gradients."
    ),
    (
        6,
        "In modern neural information retrieval, bi-encoders encode queries and document passages independently into isolated dense vector representations, enabling sub-millisecond approximate nearest-neighbor vector retrieval via cosine similarity. Conversely, cross-encoders concatenate the query and candidate passage into a single unified sequence, allowing full bidirectional token cross-attention across all layers to achieve significantly sharper relevance discrimination at the cost of higher per-query inference latency."
    ),
    (
        7,
        "Natural Language Inference (NLI) classifies the semantic relationship between a premise text and a hypothesis proposition into three fundamental categories: entailment, contradiction, and neutral. In grounded Retrieval-Augmented Generation (RAG), NLI models verify whether each sentence-level claim synthesized by an abstractive generator is strictly entailed by the retrieved source context, systematically identifying and suppressing ungrounded hallucinations."
    ),
    (
        8,
        "Retrieval quality is measured through standardized ranking evaluation metrics. Mean Reciprocal Rank (MRR) evaluates the reciprocal of the rank position of the first relevant retrieved passage averaged across all queries. Precision at K measures the fraction of top-K retrieved passages that are relevant, while Recall at K quantifies the fraction of all relevant target passages successfully retrieved within the top-K ranking window."
    ),
]


def get_sample_chunks() -> List[Chunk]:
    """Return deterministic Chunk objects for the sample document."""
    chunks = []
    for page_num, text in SAMPLE_PAGES:
        chunk_id = f"{DOCUMENT_ID}_p{page_num:03d}_c{page_num:04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=DOCUMENT_ID,
                page_number=page_num,
                chunk_index=page_num - 1,
                text=text,
                char_count=len(text),
                word_count=len(text.split()),
            )
        )
    return chunks
