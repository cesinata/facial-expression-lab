import pytest

from fel.latency import measure_latency


def test_latency_reports_median_and_p95():
    r = measure_latency(lambda x: sum(range(1000)), None, n_runs=30, n_warmup=2)
    assert r["median_ms"] > 0 and r["p95_ms"] >= r["median_ms"] and r["n_runs"] == 30


def test_latency_calls_sync_around_each_run():
    calls = []
    measure_latency(lambda x: None, None, n_runs=3, n_warmup=0, sync=lambda: calls.append(1))
    assert len(calls) == 6


def test_latency_needs_at_least_one_run():
    with pytest.raises(ValueError):
        measure_latency(lambda x: None, None, n_runs=0)
