#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import itertools
import subprocess
import csv
import os
from datetime import datetime


LOCUSTFILE = Path(__file__).parent / "locustest.py"

def run_one(host, duration, spawn_rate, users, batch_size, sample_pool_size, output_prefix):
    env = { 
        **os.environ,
        "LOADTEST_BATCH_SIZE": str(batch_size),
        "LOADTEST_SAMPLE_POOL_SIZE" : str(sample_pool_size)
    }
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Launch cli for isolation
    subprocess.run([
        "locust", "-f", str(LOCUSTFILE),
        "--host", host,
        "--headless",
        "-u", str(users),
        "-r", str(spawn_rate),
        "-t", duration,
        "--csv", str(output_prefix)
    ], env=env, check=True)

def read_stats(output_prefix: Path) -> dict:
    stats_path = Path(f"{output_prefix}_stats.csv")
    with stats_path.open() as f:
        for row in csv.DictReader(f):
            if row["Name"] == "Aggregated":
                return { 
                    "requests": int(row["Request Count"]),
                    "failures": int(row["Failure Count"]),
                    "rps": float(row["Requests/s"]),
                    "median_ms": float(row["Median Response Time"]),
                    "p95_ms": float(row["95%"]),
                    "p99_ms": float(row["99%"])
                }
    raise ValueError(f"No 'Aggregated' row found in {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Load testing")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="Label for this test run")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    run_dir = Path("loadtest-results") / f"{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    summary_rows = []
    combos = list(itertools.product(
        config["concurrency_levels"],
        config["batch_sizes"]
    ))

    for users, batch_size in combos:
        print(f"--- users={users}, batch size={batch_size} --- ")
        output_prefix = run_dir / f"u{users}_b{batch_size}"
        run_one(
            host=config["host"],
            duration=config["duration"],
            spawn_rate=config["spawn_rate"],
            users=users,
            batch_size=batch_size,
            sample_pool_size=config["sample_pool_size"],
            output_prefix=output_prefix
        )

        stats = read_stats(output_prefix)
        summary_rows.append({
            "users": users,
            "batch_size": batch_size,
            **stats
        })

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"Summary written to {summary_path}")

if __name__ == "__main__":
    main()