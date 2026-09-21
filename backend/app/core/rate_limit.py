"""In-process token-bucket limiter.

Single-process app, so an in-memory limiter is sufficient and avoids a Redis
dependency. Protects the two endpoints that are actually expensive or abusable:
sign-in (credential stuffing) and uploads/chat (CPU + LLM quota).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from app.core.errors import RateLimitError


@dataclass
class _Bucket:
    tokens: float
    updated_at: float = field(default_factory=time.monotonic)


class TokenBucketLimiter:
    def __init__(self, rate_per_minute: int, burst: int, name: str = "requests") -> None:
        self.rate_per_second = rate_per_minute / 60.0
        self.burst = float(burst)
        self.name = name
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str, cost: float = 1.0) -> None:
        """Consume `cost` tokens for `key`, or raise RateLimitError."""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.burst, updated_at=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.updated_at
            bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate_per_second)
            bucket.updated_at = now

            if bucket.tokens < cost:
                wait = (cost - bucket.tokens) / self.rate_per_second
                raise RateLimitError(
                    "Too many {} attempts. Try again in {} seconds.".format(
                        self.name, max(1, int(wait))
                    ),
                    detail={"retry_after_seconds": max(1, int(wait))},
                )

            bucket.tokens -= cost
            self._prune(now)

    def _prune(self, now: float) -> None:
        """Drop buckets idle for over an hour so memory cannot grow unbounded."""
        if len(self._buckets) < 512:
            return
        stale = [k for k, b in self._buckets.items() if now - b.updated_at > 3600]
        for key in stale:
            self._buckets.pop(key, None)


login_limiter = TokenBucketLimiter(rate_per_minute=10, burst=10, name="sign-in")
signup_limiter = TokenBucketLimiter(rate_per_minute=5, burst=5, name="sign-up")
upload_limiter = TokenBucketLimiter(rate_per_minute=30, burst=15, name="upload")
chat_limiter = TokenBucketLimiter(rate_per_minute=30, burst=10, name="chat")
