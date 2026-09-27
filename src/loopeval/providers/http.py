from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    """Return a bounded retry delay, honoring Retry-After when supplied."""
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 30.0)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=UTC)
                seconds = (retry_at - datetime.now(UTC)).total_seconds()
                return min(max(seconds, 0.0), 30.0)
            except (TypeError, ValueError, OverflowError):
                pass
    return min(0.25 * (2**attempt), 2.0)
