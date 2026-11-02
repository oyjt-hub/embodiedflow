#!/usr/bin/env python3
"""Run the built-in MuJoCo policy comparison as a normal EmbodiedFlow Worker job."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("cartpole_balance", "arm_reach"), required=True)
    parser.add_argument("--seeds-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seeds = json.loads(args.seeds_json)
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty JSON array")
    seeds = [int(seed) for seed in seeds]

    if args.task == "cartpole_balance":
        from services import benchmark as suite

        policies = ["passive", "pd_balance"]
    else:
        from services import benchmark_arm as suite

        policies = ["proportional", "random"]

    replay_dir = Path(args.output).parent / "replay"
    result = suite.compare(seeds, policies, {"task": args.task}, replay_dir=str(replay_dir))
    result.setdefault("task", args.task)
    result.setdefault("seeds", seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"task": args.task, "seeds": len(seeds), "report": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
