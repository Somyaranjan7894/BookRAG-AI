"""Service responsible for validating document uploads and enqueueing background processing tasks."""

import hashlib
import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import Document, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import DocumentUploadResponse

logger = get_logger(__name__)


class DocumentUploadService:
    """Handles incoming PDF uploads or file references, persists initial queued state,

    and enqueues the Celery background processing task.
    """

    def __init__(self, session: Optional[Session] = None) -> None:
        self.session = session
        self.doc_repo = DocumentRepository(session) if session is not None else None

    async def handle_upload_or_enqueue(self, request: Request) -> DocumentUploadResponse:
        """Process incoming HTTP upload request and enqueue background task.

        Args:
            request: FastAPI request containing multipart form data or JSON body.

        Returns:
            DocumentUploadResponse with document ID, task ID, and status 'queued'.

        Raises:
            HTTPException: On validation failure or task enqueueing error.
        """
        content_type = request.headers.get("content-type", "")
        os.makedirs(settings.UPLOAD_STORAGE_DIR, exist_ok=True)

        dest_path: str
        filename: str
        custom_doc_id: Optional[str] = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            uploaded_file = form.get("file")
            if not uploaded_file or not hasattr(uploaded_file, "filename"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Form data missing required 'file' upload",
                )
            filename = uploaded_file.filename or "upload.pdf"
            custom_doc_id = form.get("document_id")  # type: ignore[assignment]
            file_bytes = await uploaded_file.read()  # type: ignore[misc]
            if not file_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file is empty",
                )
            dest_filename = f"{hashlib.sha256(file_bytes).hexdigest()[:12]}_{filename}"
            dest_path = os.path.join(settings.UPLOAD_STORAGE_DIR, dest_filename)
            with open(dest_path, "wb") as f:
                f.write(file_bytes)
        else:
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid request payload; expected multipart form-data or JSON with 'file_path'",
                )
            raw_file_path = body.get("file_path")
            if not raw_file_path or not isinstance(raw_file_path, str) or not raw_file_path.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON payload missing required 'file_path'",
                )
            source_path = Path(raw_file_path.strip())
            if not source_path.is_file():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Source file not found at '{raw_file_path}'",
                )
            filename = source_path.name
            dest_path = str(source_path.resolve())
            custom_doc_id = body.get("document_id")

        # Determine stable deterministic document ID if not explicitly provided
        if custom_doc_id and custom_doc_id.strip():
            document_id = custom_doc_id.strip()
        else:
            with open(dest_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            base_name = Path(filename).stem
            safe_stem = "".join(c if c.isalnum() else "_" for c in base_name)[:20].strip("_")
            document_id = f"doc_{safe_stem}_{file_hash}"

        # Register initial durable queued state in PostgreSQL
        if self.session is not None and self.doc_repo is not None:
            doc = self.doc_repo.get_by_id(document_id)
            if doc is None:
                doc = Document(
                    document_id=document_id,
                    filename=filename,
                    status=DocumentStatus.QUEUED.value,
                    processing_stage="queued",
                )
                self.doc_repo.create(doc)
            else:
                self.doc_repo.update_status(
                    document_id,
                    status=DocumentStatus.QUEUED,
                    stage="queued",
                    error_message=None,
                )
            self.session.commit()

        # Enqueue Celery task
        task_id: Optional[str] = None
        try:
            from app.workers.tasks import process_document_task

            task = process_document_task.delay(document_id=document_id, file_path=dest_path)
            task_id = task.id
            logger.info("Enqueued process_document_task for document '%s' with task_id=%s", document_id, task_id)
        except Exception as exc:
            logger.exception("Failed to enqueue Celery task for document '%s': %s", document_id, exc)
            if self.session is not None and self.doc_repo is not None:
                self.doc_repo.update_status(
                    document_id,
                    status=DocumentStatus.FAILED,
                    stage="queue_failed",
                    error_message=str(exc),
                )
                self.session.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to enqueue background processing task: {exc}",
            )

        return DocumentUploadResponse(
            document_id=document_id,
            task_id=task_id,
            status=DocumentStatus.QUEUED.value,
            message="Document processing task enqueued successfully.",
        )
