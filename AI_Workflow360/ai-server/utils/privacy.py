"""Last-resort PII guard — called at the top of every inference function.

If forbidden fields somehow survive the middleware and sanitization layers,
this raises before the model ever sees the data.
"""

import logging

_security_log = logging.getLogger("ai-server.security")

_PII_KEYS = {
    "email", "phone", "password", "avatar", "avatar_url",
    "full_name", "token", "session", "cookie",
    "content", "body", "message", "subject",
    "storage_path", "public_url", "file_path",
    "security_question", "security_answer",
    "encrypted_password", "recovery_token", "confirmation_token",
    "access_token", "refresh_token",
    "raw_user_meta_data", "raw_app_meta_data",
}


def assert_no_pii(data: dict, context: str = "inference") -> None:
    """Hard stop if PII somehow reaches an inference function.

    Raises ValueError so the model never runs. Logs the violation
    to security.log (field names only, never values).
    """
    flat_keys = {k.lower() for k in _flatten_keys(data)}
    found = _PII_KEYS & flat_keys
    if found:
        _security_log.warning(
            "PII BLOCK at %s layer — refused inference. Fields: %s",
            context,
            ", ".join(sorted(found)),
        )
        raise ValueError(
            f"PII block: refused to run {context} with fields: {found}"
        )


def _flatten_keys(data: dict, prefix: str = "") -> set[str]:
    """Recursively collect all keys in a nested dict/list structure."""
    keys: set[str] = set()
    for k, v in data.items():
        keys.add(k)
        if isinstance(v, dict):
            keys |= _flatten_keys(v, f"{prefix}{k}.")
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    keys |= _flatten_keys(item, f"{prefix}{k}[].")
    return keys
