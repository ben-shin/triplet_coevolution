"""Measure cold and verified-reuse local stages on a supplied configuration. No network."""

import argparse
import json
import tempfile
import time
from pathlib import Path

from triplet_coevolution.local_pipeline import run_pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--until", choices=["prepare", "plan", "score", "report"], default="plan")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    results = []
    with tempfile.TemporaryDirectory(prefix="triplet-job-benchmark-") as tmp:
        for i in range(args.repeats):
            root = Path(tmp) / str(i)
            for state in ("cold", "reuse"):
                start = time.perf_counter()
                result = run_pipeline(args.config, root, until=args.until)
                elapsed = time.perf_counter() - start
                row = {
                    "repeat": i,
                    "state": state,
                    "elapsed_seconds": elapsed,
                    "status": result["status"],
                    "stages": {
                        s["stage"]: {
                            k: s[k] for k in ("action", "artifact_bytes", "measured") if k in s
                        }
                        for s in result["stages"]
                    },
                }
                plan = next(
                    (
                        Path(s["output"]) / "plan.json"
                        for s in result["stages"]
                        if s["stage"] == "plan"
                    ),
                    None,
                )
                if plan:
                    p = json.loads(plan.read_text(encoding="utf-8"))
                    row.update(observations=p["observations"], estimates=p["estimates"])
                results.append(row)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
