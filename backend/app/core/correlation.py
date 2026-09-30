"""Request correlation and context tracing for BookRAG AI.

Provides request ID generation, propagation, ContextVar tracking, and ASGI middleware.
"""

from contextvars import ContextVar
import re
import time
from typing import Callable, Optional
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import get_logger

logger = get_logger(__name__)

# Request ID ContextVar accessible across all synchronous and asynchronous calls
_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

# Pattern for valid incoming request IDs (sanitized to prevent header injection)
VALID_REQUEST_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")


def get_request_id() -> Optional[str]:
    """Retrieve the current request ID from the active context."""
    return _request_id_ctx.get()


def set_request_id(request_id: Optional[str]) -> None:
    """Set the current request ID in the active context."""
    _request_id_ctx.set(request_id)


class CorrelationIdMiddleware:
    """Pure ASGI middleware for lightweight request correlation tracking.

    - Extracts 'X-Request-ID' from incoming headers or generates a new UUID4.
    - Sanitizes incoming request IDs to prevent header injection.
    - Stores request ID in contextvars and scope state.
    - Adds 'X-Request-ID' and 'X-Process-Time' headers to HTTP responses (without duplication).
    - Emits structured access logs for completed requests.
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Request-ID",
    ) -> None:
        self.app = app
        self.header_name = header_name
        self.header_name_bytes = header_name.lower().encode("latin1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.perf_counter()

        # Extract existing request ID from headers
        raw_request_id: Optional[str] = None
        for name, value in scope.get("headers", []):
            if name.lower() == self.header_name_bytes:
                try:
                    raw_request_id = value.decode("latin1").strip()
                except UnicodeDecodeError:
                    raw_request_id = None
                break

        # Validate or generate request ID
        if raw_request_id and VALID_REQUEST_ID_PATTERN.match(raw_request_id):
            request_id = raw_request_id
        else:
            request_id = str(uuid.uuid4())

        # Set context variables and scope state
        token = _request_id_ctx.set(request_id)
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id

        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                headers = list(message.get("headers", []))
                existing_names = {h[0].lower() for h in headers}

                # Add X-Request-ID header if not already added by error handler
                if self.header_name_bytes not in existing_names:
                    headers.append((self.header_name_bytes, request_id.encode("latin1")))

                # Add X-Process-Time header if not already present
                if b"x-process-time" not in existing_names:
                    headers.append((b"x-process-time", f"{elapsed_ms:.2f}ms".encode("latin1")))

                message["headers"] = headers

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            method = scope.get("method", "UNKNOWN")
            path = scope.get("path", "/")
            logger.info(
                "HTTP %s %s completed in %.2fms with status %s",
                method,
                path,
                elapsed_ms,
                status_code,
            )
            _request_id_ctx.reset(token)
