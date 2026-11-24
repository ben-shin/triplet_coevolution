"""Report actual branch and statement coverage for core and validation separately."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    options = parser.parse_args()
    obj = json.loads(options.report.read_text("utf-8"))
    if not obj["meta"]["branch_coverage"]:
        parser.error("branch measurements are required")
    below = []
    for scope, floor in (("core", 90), ("validation", 82)):
        rows = [
            v["summary"]
            for k, v in obj["files"].items()
            if f"/triplet_coevolution/{scope}/" in ("/" + k.replace("\\", "/"))
        ]
        if not rows:
            parser.error(f"missing measured package {scope}")
        count = sum(row["covered_branches"] for row in rows)
        total = sum(row["num_branches"] for row in rows)
        statements = sum(row["num_statements"] for row in rows)
        covered = sum(row["covered_lines"] for row in rows)
        rate = 100 * count / total
        print(
            f"{scope}: {count}/{total} branches ({rate:.2f}%), {covered}/{statements} statements "
            f"({100 * covered / statements:.2f}%). Branch target 90%, current regression floor {floor}%."
        )
        if rate < floor:
            below.append(scope)
    if below:
        raise SystemExit("Branch regression floor missed for " + ", ".join(below))


if __name__ == "__main__":
    main()
