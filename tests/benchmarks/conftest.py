"""Shared fixtures for Pipecat performance benchmarks.

Provides latency recording, baseline management, and experiment logging
for the auto-research optimization loop.
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

BASELINES_DIR = Path(__file__).parent / "baselines"
EXPERIMENT_LOG = Path(__file__).parent / "experiment_log.tsv"
CURRENT_BASELINE = BASELINES_DIR / "current.json"


# ---------------------------------------------------------------------------
# Latency recording
# ---------------------------------------------------------------------------


@dataclass
class LatencyRecorder:
    """Records timing samples and computes percentile statistics."""

    samples: List[float] = field(default_factory=list)

    def record(self, value_ns: float):
        """Record a latency sample in nanoseconds."""
        self.samples.append(value_ns)

    def record_us(self, value_us: float):
        """Record a latency sample in microseconds (stored as ns)."""
        self.samples.append(value_us * 1000)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_ns(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0

    @property
    def mean_us(self) -> float:
        return self.mean_ns / 1000

    def percentile_ns(self, p: float) -> float:
        if not self.samples:
            return 0
        sorted_s = sorted(self.samples)
        # Nearest-rank percentile: ceil(N * p / 100) - 1, clamped to [0, N-1]
        from math import ceil
        idx = max(0, ceil(len(sorted_s) * p / 100) - 1)
        idx = min(idx, len(sorted_s) - 1)
        return sorted_s[idx]

    def percentile_us(self, p: float) -> float:
        return self.percentile_ns(p) / 1000

    @property
    def p50_ns(self) -> float:
        return self.percentile_ns(50)

    @property
    def p95_ns(self) -> float:
        return self.percentile_ns(95)

    @property
    def p99_ns(self) -> float:
        return self.percentile_ns(99)

    @property
    def p50_us(self) -> float:
        return self.p50_ns / 1000

    @property
    def p95_us(self) -> float:
        return self.p95_ns / 1000

    @property
    def p99_us(self) -> float:
        return self.p99_ns / 1000

    def summary(self, unit: str = "ns") -> Dict[str, float]:
        divisor = 1000 if unit == "us" else 1
        return {
            "p50": self.p50_ns / divisor,
            "p95": self.p95_ns / divisor,
            "p99": self.p99_ns / divisor,
            "mean": self.mean_ns / divisor,
            "count": self.count,
        }


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------


def load_baseline() -> Optional[Dict[str, Any]]:
    """Load the current baseline from JSON file."""
    if CURRENT_BASELINE.exists():
        with open(CURRENT_BASELINE) as f:
            return json.load(f)
    return None


def save_baseline(metrics: Dict[str, Any]):
    """Save metrics as the current baseline."""
    import subprocess

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)

    sha = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
    except Exception:
        pass

    baseline = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "metrics": metrics,
    }

    with open(CURRENT_BASELINE, "w") as f:
        json.dump(baseline, f, indent=2)

    # Also save a timestamped copy
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(BASELINES_DIR / f"baseline_{ts}.json", "w") as f:
        json.dump(baseline, f, indent=2)

    return baseline


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------


def log_experiment(
    experiment_id: int,
    git_sha: str,
    description: str,
    metric_name: str,
    baseline_value: float,
    result_value: float,
    status: str,
    duration_sec: float,
):
    """Append an experiment result to the TSV log."""
    delta_pct = ((result_value - baseline_value) / baseline_value * 100) if baseline_value else 0

    header = "timestamp\texperiment_id\tgit_sha\tdescription\tmetric_name\tbaseline\tresult\tdelta_pct\tstatus\tduration_sec\n"

    if not EXPERIMENT_LOG.exists():
        with open(EXPERIMENT_LOG, "w") as f:
            f.write(header)

    with open(EXPERIMENT_LOG, "a") as f:
        row = (
            f"{datetime.now(timezone.utc).isoformat()}\t"
            f"{experiment_id}\t"
            f"{git_sha}\t"
            f"{description}\t"
            f"{metric_name}\t"
            f"{baseline_value:.2f}\t"
            f"{result_value:.2f}\t"
            f"{delta_pct:.2f}\t"
            f"{status}\t"
            f"{duration_sec:.1f}\n"
        )
        f.write(row)


def load_experiment_log() -> List[Dict[str, str]]:
    """Load the experiment log as a list of dicts."""
    if not EXPERIMENT_LOG.exists():
        return []

    rows = []
    with open(EXPERIMENT_LOG) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return []
        headers = lines[0].strip().split("\t")
        for line in lines[1:]:
            values = line.strip().split("\t")
            if len(values) == len(headers):
                rows.append(dict(zip(headers, values)))
    return rows


def get_next_experiment_id() -> int:
    """Get the next experiment ID from the log."""
    rows = load_experiment_log()
    if not rows:
        return 1
    return max(int(r.get("experiment_id", 0)) for r in rows) + 1


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self):
        self.elapsed_ns = 0
        self.elapsed_us = 0
        self.elapsed_ms = 0
        self.elapsed_s = 0

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.elapsed_ns = time.perf_counter_ns() - self._start
        self.elapsed_us = self.elapsed_ns / 1000
        self.elapsed_ms = self.elapsed_ns / 1_000_000
        self.elapsed_s = self.elapsed_ns / 1_000_000_000


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def latency_recorder():
    """Provide a fresh LatencyRecorder."""
    return LatencyRecorder()


@pytest.fixture
def timer():
    """Provide a Timer instance."""
    return Timer()


@pytest.fixture
def baseline():
    """Load the current baseline, or None if not established."""
    return load_baseline()
