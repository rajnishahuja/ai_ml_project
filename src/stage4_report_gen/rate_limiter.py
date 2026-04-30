"""Token-bucket rate limiter + exponential-backoff helper for Gemini.

The free tier caps Gemini Flash at ~15 requests/minute; we run at 12 RPM
(stage4_config.yaml :: gemini.rate_limit_rpm) to leave headroom. The
limiter is a classic token bucket with capacity = RPM and refill rate =
RPM / 60 tokens per second. acquire() blocks until at least one token is
available.

backoff_delays() yields the configured exponential delays plus a small
random jitter so concurrent retries don't synchronize.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Iterator

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    def __init__(self, rate_per_minute: int) -> None:
        if rate_per_minute <= 0:
            raise ValueError(f"rate_per_minute must be positive, got {rate_per_minute}")
        self.capacity = float(rate_per_minute)
        self.refill_per_sec = rate_per_minute / 60.0
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
            self._last_refill = now

    def acquire(self, *, sleep_fn=time.sleep, _now_fn=time.monotonic) -> float:
        """Block until one token is available; returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                deficit = 1.0 - self._tokens
                wait = deficit / self.refill_per_sec
            sleep_fn(wait)
            waited += wait

    @property
    def available_tokens(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens


def backoff_delays(delays_sec: list[float], *, jitter: float = 0.25) -> Iterator[float]:
    """Yield delays for an exponential-backoff retry loop, with jitter.

    Args:
        delays_sec: Base delay sequence (e.g. [2, 4, 8]).
        jitter: Multiplicative random jitter in [-jitter, +jitter].
    """
    for base in delays_sec:
        factor = 1.0 + random.uniform(-jitter, jitter)
        yield max(0.0, base * factor)
