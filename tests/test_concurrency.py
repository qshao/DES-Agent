import threading
import time

from des_multi_agent.concurrency import CallResult, run_concurrent


def test_run_concurrent_empty_list_returns_empty_list():
    assert run_concurrent([], lambda x: x) == []


def test_run_concurrent_preserves_input_order():
    items = [5, 1, 4, 2, 3]

    def _slow_identity(x):
        time.sleep(0.01 * (5 - x))
        return x

    results = run_concurrent(items, _slow_identity)
    assert [r.value for r in results] == items


def test_run_concurrent_captures_single_item_failure_without_aborting_others():
    def _call(x):
        if x == 2:
            raise ValueError("boom")
        return x * 10

    results = run_concurrent([1, 2, 3], _call)

    assert results[0] == CallResult(value=10, error=None)
    assert results[1].value is None
    assert isinstance(results[1].error, ValueError)
    assert str(results[1].error) == "boom"
    assert results[2] == CallResult(value=30, error=None)


def test_run_concurrent_caps_workers_at_item_count():
    max_seen_concurrent = []
    lock = threading.Lock()
    active = {"count": 0}

    def _call(x):
        with lock:
            active["count"] += 1
            max_seen_concurrent.append(active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return x

    run_concurrent([1, 2, 3], _call, max_workers=8)

    assert max(max_seen_concurrent) <= 3
