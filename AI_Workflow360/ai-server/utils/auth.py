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
_security_handler = logging.FileHandler("security.log", encoding="utf-8")
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


class PayloadInspectorMiddleware(BaseHTTPMiddleware):
    """Rejects POST requests containing forbidden PII fields.
    Logs violations to security.log (field names only, never values).
    """

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            try:
                body_bytes = await request.body()
                if body_bytes:
                    data = json.loads(body_bytes)
                    if isinstance(data, dict):
                        violations = inspect_payload(data)
                        if violations:
                            timestamp = datetime.now(timezone.utc).isoformat()
                            client = request.client.host if request.client else "unknown"
                            _security_log.warning(
                                "PAYLOAD VIOLATION | ip=%s | path=%s | time=%s | fields=%s",
                                client, request.url.path, timestamp,
                                "; ".join(violations),
                            )
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "error": "payload_violation",
                                    "detail": "Request contains forbidden data fields that violate the AI data privacy boundary.",
                                    "fields": violations,
                                },
                            )

                    async def receive():
                        return {"type": "http.request", "body": body_bytes}
                    request._receive = receive

            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return await call_next(request)
