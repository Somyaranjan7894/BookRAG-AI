"""Celery tasks for asynchronous background document processing."""

from typing import Any, Dict

from app.core.config import settings
from app.core.device import empty_cache
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.services.document_processing.exceptions import (
    PermanentProcessingError,
    TransientProcessingError,
)
from app.services.document_processing.service import DocumentProcessingService
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    bind=True,
    name="app.workers.tasks.process_document_task",
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=settings.CELERY_TASK_DEFAULT_RETRY_DELAY,
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    acks_late=True,
)
def process_document_task(self, document_id: str, file_path: str) -> Dict[str, Any]:
    """Execute asynchronous document processing in the background.

    Args:
        document_id: Deterministic document identifier.
        file_path: Path to source PDF file on shared storage.

    Returns:
        Compact summary dict with status and counts.

    Raises:
        Retry: When transient retryable errors occur (up to max_retries).
    """
    task_id = getattr(self.request, "id", "local_direct")
    retries = getattr(self.request, "retries", 0)

    logger.info(
        "Celery worker received process_document_task: task_id=%s, document_id=%s, file_path=%s, attempt=%d",
        task_id,
        document_id,
        file_path,
        retries + 1,
    )

    session_factory = get_session_factory()
    try:
        with session_factory() as session:
            service = DocumentProcessingService(session=session)
            try:
                result = service.process(document_id=document_id, file_path=file_path)
                logger.info(
                    "process_document_task completed successfully: task_id=%s, doc_id=%s",
                    task_id,
                    document_id,
                )
                return result
            except TransientProcessingError as exc:
                logger.warning(
                    "Transient failure processing document '%s' (task_id=%s, attempt=%d/%d): %s",
                    document_id,
                    task_id,
                    retries + 1,
                    self.max_retries,
                    exc,
                )
                raise self.retry(exc=exc)
            except PermanentProcessingError as exc:
                logger.error(
                    "Permanent unretryable failure processing document '%s' (task_id=%s): %s",
                    document_id,
                    task_id,
                    exc,
                )
                raise
            except Exception as exc:
                logger.exception(
                    "Unhandled failure processing document '%s' (task_id=%s): %s",
                    document_id,
                    task_id,
                    exc,
                )
                raise
    finally:
        empty_cache()
