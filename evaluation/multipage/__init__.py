"""Multi-page synthesis evaluation module."""

from evaluation.multipage.evaluator import (
    AggregatedMultiPageMetrics,
    MultiPageQueryEvaluation,
    evaluate_multipage_synthesis,
)

__all__ = [
    "AggregatedMultiPageMetrics",
    "MultiPageQueryEvaluation",
    "evaluate_multipage_synthesis",
]
