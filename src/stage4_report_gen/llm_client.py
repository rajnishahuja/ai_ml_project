"""The single Gemini Flash entry point for Stage 4.

Every Gemini call in Stage 4 must come through `GeminiClient.generate(...)`.
Only `explainer.py` and `pattern_deriver.py` (tier-2 fallback) are
permitted to import this module — see the hard rules in
docs/STAGE4_RESUME_HANDOFF.md and docs/STAGE4_DECISION_LOG.md.

Behavior:
  - Loads GEMINI_API_KEY from the project-root .env via python-dotenv.
  - Loads gemini config from configs/stage4_config.yaml (model, rate_limit,
    retries, retry delays, cache_dir).
  - GEMINI_MODEL env var, when set, overrides gemini.model at runtime
    (decision #5 in STAGE4_DECISION_LOG.md).
  - Requests pass through GeminiCache (SHA-256 keyed) → TokenBucketRateLimiter
    (12 RPM) → google.generativeai SDK.
  - On 429 / transient error: exponential backoff up to gemini.max_retries.
  - On terminal failure: returns the soft-fail placeholder string and logs
    a warning. NEVER raises into the report builder — failure isolation
    (hard rule #7).
  - Never logs the API key.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from src.common.utils import load_config
from src.stage4_report_gen.cache import GeminiCache
from src.stage4_report_gen.rate_limiter import TokenBucketRateLimiter, backoff_delays

logger = logging.getLogger(__name__)


SOFT_FAIL_PLACEHOLDER = (
    "[explanation unavailable — Gemini call failed after retries; "
    "review the underlying clause and risk_explanation directly]"
)


def _load_dotenv_once() -> None:
    """Idempotent .env loader — tolerates python-dotenv being missing."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        logger.debug("python-dotenv not installed; relying on process env only.")
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


class GeminiClient:
    def __init__(
        self,
        config_path: str | Path = "configs/stage4_config.yaml",
        *,
        sdk: Any = None,
        api_key: Optional[str] = None,
    ) -> None:
        cfg = load_config(str(config_path))
        gemini_cfg = cfg.get("gemini", {})

        self.model = os.environ.get("GEMINI_MODEL") or gemini_cfg.get(
            "model", "gemini-2.5-flash"
        )
        self.max_retries = int(gemini_cfg.get("max_retries", 3))
        self.retry_delays = list(gemini_cfg.get("retry_delays_sec", [2, 4, 8]))
        rpm = int(gemini_cfg.get("rate_limit_rpm", 12))
        cache_dir = gemini_cfg.get("cache_dir", ".cache/gemini")

        self.cache = GeminiCache(cache_dir)
        self.rate_limiter = TokenBucketRateLimiter(rpm)

        self._sdk = sdk
        self._api_key = api_key
        self._sdk_model = None
        self._sdk_lock = threading.Lock()
        _load_dotenv_once()

    def _resolve_api_key(self) -> Optional[str]:
        return self._api_key or os.environ.get("GEMINI_API_KEY")

    def _get_sdk_model(self) -> Any:
        if self._sdk_model is not None:
            return self._sdk_model
        with self._sdk_lock:
            if self._sdk_model is not None:
                return self._sdk_model
            if self._sdk is None:
                import google.generativeai as genai  # type: ignore
                self._sdk = genai
            api_key = self._resolve_api_key()
            if not api_key or api_key == "PASTE_KEY_HERE":
                raise RuntimeError(
                    "GEMINI_API_KEY is not set (check .env). "
                    "Cannot make live Gemini calls."
                )
            self._sdk.configure(api_key=api_key)
            self._sdk_model = self._sdk.GenerativeModel(self.model)
            return self._sdk_model

    def _call_sdk(self, prompt: str) -> str:
        model = self._get_sdk_model()
        result = model.generate_content(prompt)
        text = getattr(result, "text", None)
        if text is None:
            raise RuntimeError("Gemini response had no .text attribute")
        return str(text)

    def generate(
        self,
        prompt: str,
        *,
        sleep_fn: Callable[[float], None] = None,
    ) -> str:
        """Return Gemini's text response or the soft-fail placeholder.

        Cache → rate-limit acquire → SDK call → exponential backoff on
        failure (max self.max_retries). Soft-fails on terminal error.
        """
        cached = self.cache.get(self.model, prompt)
        if cached is not None:
            return cached

        import time
        sleep = sleep_fn or time.sleep

        last_exc: Optional[BaseException] = None
        delays = list(backoff_delays(self.retry_delays))
        attempt = 0
        while attempt <= self.max_retries:
            self.rate_limiter.acquire(sleep_fn=sleep)
            try:
                response = self._call_sdk(prompt)
                self.cache.put(self.model, prompt, response)
                return response
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries + 1, exc,
                )
                if attempt >= self.max_retries:
                    break
                delay = delays[attempt] if attempt < len(delays) else delays[-1]
                sleep(delay)
                attempt += 1

        logger.warning(
            "Gemini call soft-failed after %d retries; returning placeholder. Last error: %s",
            self.max_retries, last_exc,
        )
        return SOFT_FAIL_PLACEHOLDER


_default_client: Optional[GeminiClient] = None
_default_client_lock = threading.Lock()


def get_default_client() -> GeminiClient:
    """Lazy singleton for production code. Tests should construct GeminiClient
    directly with an injected `sdk=` mock instead of calling this."""
    global _default_client
    if _default_client is None:
        with _default_client_lock:
            if _default_client is None:
                _default_client = GeminiClient()
    return _default_client
