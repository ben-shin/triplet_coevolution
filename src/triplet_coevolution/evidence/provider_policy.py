"""Bound identifier-only HTTP lookups. Each analysis has independent limits and counters."""

import math
import random
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProviderPolicy:
    uniprot_attempts: int = 4
    nucleotide_attempts: int = 2
    timeout_seconds: float | None = None
    minimum_interval_seconds: float = 0.0
    max_requests: int = 100_000
    max_retry_wait: float = 60.0
    jitter_seconds: float = 0.0
    cache_generation: str = "default"
    progress_every: int = 0

    def __post_init__(self):
        for key, lo, hi in (
            ("uniprot_attempts", 1, 8),
            ("nucleotide_attempts", 1, 8),
            ("max_requests", 1, 10_000_000),
            ("progress_every", 0, 10_000_000),
        ):
            val = getattr(self, key)
            if type(val) is not int or not lo <= val <= hi:
                raise ValueError(f"providers.{key} requires an integer from {lo} to {hi}")
        for key, lo, hi in (
            ("timeout_seconds", 0.1, 120.0),
            ("minimum_interval_seconds", 0.0, 60.0),
            ("max_retry_wait", 0.0, 120.0),
            ("jitter_seconds", 0.0, 5.0),
        ):
            val = getattr(self, key)
            if val is None and key == "timeout_seconds":
                continue
            if type(val) not in (float, int) or not math.isfinite(val) or not lo <= val <= hi:
                raise ValueError(f"providers.{key} requires a finite number from {lo} to {hi}")
        if not isinstance(self.cache_generation, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.cache_generation
        ):
            raise ValueError(
                "providers.cache_generation requires a portable name of 1–64 letters, digits, underscores or hyphens"
            )

    @classmethod
    def parse(cls, options):
        if isinstance(options, cls):
            return options
        if options is not None and not isinstance(options, dict):
            raise TypeError("providers must be a JSON object")
        return cls(**(options or {}))


class RunRequests:
    def __init__(self, policy, progress=None, checkpoint=None, total=None):
        self.checkpoint = checkpoint
        self.expected_rows = total
        self.policy = policy
        self.count = 0
        self.completed = 0
        self.started = time.monotonic()
        self.last = {}
        self.progress = progress
        self.lock = threading.Lock()

    def request(self, service):
        with self.lock:
            if self.count >= self.policy.max_requests:
                raise ValueError(
                    "identifier lookup request budget exhausted; completed records remain reusable"
                )
            delay = (
                self.last.get(service, 0.0)
                + self.policy.minimum_interval_seconds
                - time.monotonic()
            )
            if delay > 0:
                time.sleep(delay)
            self.last[service] = time.monotonic()
            self.count += 1

    def advance(self, n):
        previous = self.completed
        self.completed += n
        every = self.policy.progress_every
        if every and self.progress and previous // every < self.completed // every:
            elapsed = time.monotonic() - self.started
            eta = (
                elapsed * max(0, self.expected_rows / self.completed - 1)
                if self.expected_rows is not None
                else None
            )
            self.progress(
                {
                    "event": "evidence_batch",
                    "examined_rows": self.completed,
                    "http_attempts": self.count,
                    "elapsed_seconds": round(elapsed, 2),
                    "total_rows": self.expected_rows,
                    "eta_seconds": round(eta, 1) if eta is not None else None,
                }
            )

    def metadata(self):
        return {
            **asdict(self.policy),
            "http_attempts": self.count,
            "max_concurrent_requests": 1,
            "payload": "public identifiers only; no sequences",
            "access_method": "direct read-only queries with verified pagination",
        }


_current = ContextVar("evidence_requests", default=None)


def current():
    return _current.get()


def policy():
    ctx = current()
    return ctx.policy if ctx else ProviderPolicy()


@contextmanager
def request_context(options=None, progress=None, *, checkpoint=None, total=None):
    ctx = RunRequests(ProviderPolicy.parse(options), progress, checkpoint, total)
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def checkpoint_keys(filename, keys):
    ctx = current()
    if keys and ctx and ctx.checkpoint:
        ctx.checkpoint(filename, keys)


def allow_request(service):
    if ctx := current():
        ctx.request(service)


def backoff(delay):
    opts = policy()
    if delay > opts.max_retry_wait:
        raise ValueError(f"provider retry wait {delay:g}s exceeds configured limit; resume later")
    return min(opts.max_retry_wait, delay + random.uniform(0.0, opts.jitter_seconds))
