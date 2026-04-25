import os
import json
import logging
from datetime import datetime, timezone

from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ---------------------------------------------------------------------------
# Security logger — writes to a dedicated security.log file
# ---------------------------------------------------------------------------
_security_log = logging.getLogger("ai-server.security")
_security_log_path = os.getenv("SECURITY_LOG_PATH", "security.log")
try:
    _security_handler: logging.Handler = logging.FileHandler(
        _security_log_path, encoding="utf-8"
    )
except (PermissionError, OSError):
    _security_handler = logging.StreamHandler()
_security_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
_security_log.addHandler(_security_handler)
_security_log.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Forbidden fields — PII / communication / auth data that must never appear
# ---------------------------------------------------------------------------
FORBIDDEN_FIELDS = {
    "email", "password", "encrypted_password", "phone",
    "avatar_url", "security_question", "security_answer",
    "raw_user_meta_data", "raw_app_meta_data",
    "recovery_token", "confirmation_token",
    "content", "body", "message", "subject",
    "storage_path", "public_url", "file_path",
    "token", "session", "cookie", "access_token", "refresh_token",
}


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """Validate the API key sent by the Next.js backend.

    Reads from AI_SERVER_API_KEY env var.
    """
    expected = os.getenv("AI_SERVER_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="AI_SERVER_API_KEY not configured on the server")
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ---------------------------------------------------------------------------
# Payload inspector
# ---------------------------------------------------------------------------
def inspect_payload(data: dict, path: str = "") -> list[str]:
    """Recursively check a payload for forbidden fields.
    Returns violation descriptions (field paths only — never values).
    """
    violations: list[str] = []
    for key, val in data.items():
        full_path = f"{path}.{key}" if path else key
        if key.lower() in FORBIDDEN_FIELDS:
            violations.append(f"Forbidden field detected: {full_path}")
        if isinstance(val, dict):
            violations.extend(inspect_payload(val, full_path))
        if isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    violations.extend(inspect_payload(item, f"{full_path}[{i}]"))
    return violations


class PayloadInspectorMiddleware:
    """Pure-ASGI middleware that rejects POST requests containing forbidden PII.

    Implemented as ASGI (not BaseHTTPMiddleware) because BaseHTTPMiddleware
    buffers response bodies and breaks StreamingResponse / truncates bodies
    behind the HF Spaces proxy.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        # Drain the incoming body
        chunks: list[bytes] = []
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                # Disconnect or unexpected message type — bail out
                await self.app(scope, receive, send)
                return
            chunks.append(message.get("body", b"") or b"")
            more_body = message.get("more_body", False)
        body_bytes = b"".join(chunks)

        # Inspect
        if body_bytes:
            try:
                data = json.loads(body_bytes)
                if isinstance(data, dict):
                    violations = inspect_payload(data)
                    if violations:
                        timestamp = datetime.now(timezone.utc).isoformat()
                        client_host = "unknown"
                        client_info = scope.get("client")
                        if client_info:
                            client_host = client_info[0] if len(client_info) else "unknown"
                        path = scope.get("path", "")
                        _security_log.warning(
                            "PAYLOAD VIOLATION | ip=%s | path=%s | time=%s | fields=%s",
                            client_host, path, timestamp, "; ".join(violations),
                        )
                        body = json.dumps({
                            "error": "payload_violation",
                            "detail": "Request contains forbidden data fields that violate the AI data privacy boundary.",
                            "fields": violations,
                        }).encode("utf-8")
                        await send({
                            "type": "http.response.start",
                            "status": 400,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                            ],
                        })
                        await send({"type": "http.response.body", "body": body})
                        return
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        # Replay the buffered body so the downstream app can read it
        replayed = False

        async def replay_receive():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
