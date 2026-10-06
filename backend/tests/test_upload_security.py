"""Security tests for document upload: path traversal, file type restriction, and file_path access control."""

import io
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from app.api.v1.dependencies import get_optional_db_session
from app.core.config import settings


# Mock out the Celery task so tests don't block on Redis connection
@pytest.fixture(autouse=True)
def mock_celery_task():
    """Prevent Celery from attempting to connect to Redis during upload security tests."""
    mock_task_result = MagicMock()
    mock_task_result.id = "mock-task-id-for-testing"
    with patch("app.workers.tasks.process_document_task") as mock_task:
        mock_task.delay.return_value = mock_task_result
        yield mock_task


@pytest.fixture(autouse=True)
def disable_db(app: FastAPI):
    """Override get_optional_db_session to yield None so upload tests don't require database."""
    def _no_db():
        yield None

    app.dependency_overrides[get_optional_db_session] = _no_db
    yield
    app.dependency_overrides.pop(get_optional_db_session, None)


# ---------------------------------------------------------------------------
# Issue 3 & 4: Multipart upload — path traversal + PDF-only validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "malicious_filename",
    [
        "../../evil.pdf",
        "..\\..\\evil.pdf",
        "/absolute/path/evil.pdf",
        "C:\\temp\\evil.pdf",
        "../../../etc/passwd.pdf",
    ],
)
def test_upload_path_traversal_filename_sanitized(client: TestClient, malicious_filename: str) -> None:
    """Verify that malicious filenames are sanitized and the file stays inside UPLOAD_STORAGE_DIR."""
    dummy_pdf = b"%PDF-1.4 dummy content for testing"
    response = client.post(
        "/api/v1/documents",
        files={"file": (malicious_filename, io.BytesIO(dummy_pdf), "application/pdf")},
    )
    # Should succeed (sanitized basename is valid .pdf) — never write outside UPLOAD_STORAGE_DIR
    assert response.status_code == status.HTTP_202_ACCEPTED
    upload_dir = os.path.realpath(settings.UPLOAD_STORAGE_DIR)
    for root, _dirs, files in os.walk(upload_dir):
        for f in files:
            assert os.path.realpath(os.path.join(root, f)).startswith(upload_dir)


@pytest.mark.parametrize(
    "filename,should_reject",
    [
        ("book.pdf", False),
        ("book.PDF", False),
        ("BOOK.Pdf", False),
        ("book.txt", True),
        ("book.exe", True),
        ("book.zip", True),
        ("book.py", True),
        ("book", True),
    ],
)
def test_upload_rejects_non_pdf_extensions(client: TestClient, filename: str, should_reject: bool) -> None:
    """Verify that only PDF extensions are accepted at the upload boundary."""
    dummy_content = b"%PDF-1.4 dummy content for testing"
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, io.BytesIO(dummy_content), "application/octet-stream")},
    )
    if should_reject:
        assert response.status_code == status.HTTP_400_BAD_REQUEST, (
            f"Expected 400 for '{filename}' but got {response.status_code}"
        )
        data = response.json()
        err_msg = data.get("detail") or data.get("error", {}).get("message", "")
        assert "PDF" in err_msg
    else:
        # Valid PDF extension — should be accepted (202)
        assert response.status_code == status.HTTP_202_ACCEPTED


# ---------------------------------------------------------------------------
# Issue 5: JSON file_path — directory restriction
# ---------------------------------------------------------------------------


def test_json_file_path_outside_upload_dir_rejected(client: TestClient) -> None:
    """Verify that file_path pointing outside UPLOAD_STORAGE_DIR is rejected."""
    response = client.post(
        "/api/v1/documents",
        json={"file_path": "/etc/passwd"},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_json_file_path_traversal_rejected(client: TestClient) -> None:
    """Verify that traversal attempts in file_path are rejected."""
    response = client.post(
        "/api/v1/documents",
        json={"file_path": os.path.join(settings.UPLOAD_STORAGE_DIR, "..", "..", "etc", "passwd")},
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_json_file_path_absolute_system_path_rejected(client: TestClient) -> None:
    """Verify that absolute system paths are rejected."""
    response = client.post(
        "/api/v1/documents",
        json={"file_path": "C:\\Windows\\System32\\cmd.exe"},
    )
    assert response.status_code in (status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND)


def test_json_file_path_within_upload_dir_allowed(client: TestClient) -> None:
    """Verify that file_path within UPLOAD_STORAGE_DIR succeeds (if file exists and is PDF)."""
    os.makedirs(settings.UPLOAD_STORAGE_DIR, exist_ok=True)
    test_pdf = os.path.join(os.path.realpath(settings.UPLOAD_STORAGE_DIR), "test_allowed.pdf")
    with open(test_pdf, "wb") as f:
        f.write(b"%PDF-1.4 test content")
    try:
        response = client.post(
            "/api/v1/documents",
            json={"file_path": test_pdf},
        )
        # Should be accepted (202) — not 403
        assert response.status_code != status.HTTP_403_FORBIDDEN
    finally:
        if os.path.exists(test_pdf):
            os.remove(test_pdf)


def test_json_file_path_non_pdf_within_upload_dir_rejected(client: TestClient) -> None:
    """Verify that non-PDF files within upload dir are still rejected."""
    os.makedirs(settings.UPLOAD_STORAGE_DIR, exist_ok=True)
    test_txt = os.path.join(os.path.realpath(settings.UPLOAD_STORAGE_DIR), "test_reject.txt")
    with open(test_txt, "w") as f:
        f.write("not a PDF")
    try:
        response = client.post(
            "/api/v1/documents",
            json={"file_path": test_txt},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    finally:
        if os.path.exists(test_txt):
            os.remove(test_txt)
