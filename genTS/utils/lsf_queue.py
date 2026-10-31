"""
LSF queue selection utility.
"""

import subprocess


def _get_queue_stats(queue_name):
    """Return (pending, running) job counts for the given queue."""
    try:
        result = subprocess.run(
            ["bqueues", "-noheader", "-o", "pend run", queue_name],
            capture_output=True, text=True, timeout=15,
        )
        parts = result.stdout.strip().split()
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
        if len(parts) == 1:
            return int(parts[0]), 0
        return 0, 0
    except Exception:
        return 0, 0


def pick_best_queue(queue_configs, fallback=None, verbose=True):
    """Pick the queue with the lowest congestion ratio (PEND / RUN)."""
    if fallback is None:
        fallback = queue_configs[0]["queue"]

    entries = []
    for cfg in queue_configs:
        pending, running = _get_queue_stats(cfg["queue"])
        congestion = pending / max(running, 1)
        entry = {**cfg, "pending": pending, "running": running,
                 "congestion": congestion}
        entries.append(entry)
        if verbose:
            print(f"  [queue] {cfg['queue']:20s}  "
                  f"pend: {pending:4d}  run: {running:4d}  "
                  f"pend/run: {congestion:.2f}")

    if not entries:
        for cfg in queue_configs:
            if cfg["queue"] == fallback:
                return {**cfg, "pending": 0, "running": 0,
                        "congestion": float("inf")}
        return {"queue": fallback, "pending": 0, "running": 0,
                "congestion": float("inf")}

    best = min(entries, key=lambda e: e["congestion"])
    if verbose:
        print(f"  \u2192 {best['queue']} (pend/run={best['congestion']:.2f})")
    return best
