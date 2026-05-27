#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Text-to-image latency benchmark sweep for the HunyuanImage-3 server.

Emits one CSV row per measurement (cold/warm-N) so results can be diffed
across commits. Aligns with `docs/superpowers/specs/...port-design.md`
§ Perf targets — median of ≥5 warm runs is the headline number; cold runs
are reported but never gate a release.

Example:
    python -m benchmarks.hunyuan_image3.bench_t2i \\
        --endpoint http://127.0.0.1:30010 \\
        --model HunyuanImage-3.0-Instruct-Distil \\
        --size 768x1024 \\
        --bot-task recaption think_recaption \\
        --steps 8 28 \\
        --seeds 42 43 44 45 46 \\
        --warmup 2 \\
        --measure 5 \\
        --commit "$(git rev-parse --short HEAD)" \\
        --out benchmarks/hunyuan_image3/results/run-$(date +%Y%m%d-%H%M%S).csv

The script does NOT modify the server; just curls it. Run it after a server
has been booted with the pipeline of interest. To compare two backends, run
twice with different `--commit` tags and concatenate the CSVs.

Output schema (one row per measurement, header always emitted):
    commit_sha,timestamp,model,bot_task,steps,size,seed,run_idx,kind,
    http_code,e2e_ms,server_ar_ms,server_dit_ms,server_kv_transfer_ms,
    ar_decoded_tokens,gpu_mem_mb_peak,error

Fields the server doesn't report come back as empty strings.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = (
    "A cinematic portrait of a young woman in a red silk dress standing in a "
    "rainy neon street, detailed face, soft rim light, shallow depth of field, "
    "high quality"
)

DEFAULT_CSV_COLUMNS = [
    "commit_sha",
    "timestamp",
    "model",
    "bot_task",
    "steps",
    "size",
    "seed",
    "run_idx",
    "kind",  # "cold" | "warm"
    "http_code",
    "e2e_ms",
    "server_ar_ms",
    "server_dit_ms",
    "server_kv_transfer_ms",
    "ar_decoded_tokens",
    "gpu_mem_mb_peak",
    "error",
]


@dataclass
class MeasurementRow:
    commit_sha: str
    timestamp: str
    model: str
    bot_task: str
    steps: int
    size: str
    seed: int
    run_idx: int
    kind: str
    http_code: int | None = None
    e2e_ms: float | None = None
    server_ar_ms: float | None = None
    server_dit_ms: float | None = None
    server_kv_transfer_ms: float | None = None
    ar_decoded_tokens: int | None = None
    gpu_mem_mb_peak: int | None = None
    error: str = ""

    def as_csv(self) -> list[str]:
        def fmt(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)

        return [
            self.commit_sha,
            self.timestamp,
            self.model,
            self.bot_task,
            fmt(self.steps),
            self.size,
            fmt(self.seed),
            fmt(self.run_idx),
            self.kind,
            fmt(self.http_code),
            fmt(self.e2e_ms),
            fmt(self.server_ar_ms),
            fmt(self.server_dit_ms),
            fmt(self.server_kv_transfer_ms),
            fmt(self.ar_decoded_tokens),
            fmt(self.gpu_mem_mb_peak),
            self.error,
        ]


def _post(
    endpoint: str,
    body: dict[str, Any],
    *,
    timeout: float = 600.0,
) -> tuple[int, dict[str, Any] | None, str]:
    """POST JSON to the endpoint. Returns (http_code, parsed_body|None, error)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            code = resp.status
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            return code, parsed, ""
    except urllib.error.HTTPError as e:
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            body_text = ""
        return e.code, None, body_text[:300]
    except Exception as e:
        return 0, None, f"{type(e).__name__}: {e}"


def _one_measurement(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    bot_task: str,
    steps: int,
    size: str,
    seed: int,
    guidance_scale: float,
    timeout: float,
) -> tuple[int, dict[str, Any] | None, str, float]:
    """Issue one request and return (http_code, body, error, e2e_ms)."""
    body = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "bot_task": bot_task,
    }
    t0 = time.perf_counter()
    code, parsed, err = _post(endpoint + "/v1/images/generations", body, timeout=timeout)
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    return code, parsed, err, e2e_ms


def _extract_server_metrics(body: dict[str, Any] | None) -> dict[str, Any]:
    """Pull optional per-stage timing fields from the server response.

    The server may return per-stage timings as `stage_durations` (matches
    sglang-omni's existing convention). Best-effort — if any field is missing
    we just leave the CSV cell empty.
    """
    out: dict[str, Any] = {}
    if not isinstance(body, dict):
        return out
    # Stage timings — flexible: support both "stage_durations" list and
    # explicit per-stage millisecond fields.
    durations = body.get("stage_durations")
    if isinstance(durations, dict):
        for src_key, dst_key in (
            ("ar", "server_ar_ms"),
            ("dit", "server_dit_ms"),
            ("kv_transfer", "server_kv_transfer_ms"),
        ):
            val = durations.get(src_key)
            if isinstance(val, (int, float)):
                out[dst_key] = float(val)
    elif isinstance(durations, list):
        for entry in durations:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("stage")
            ms = entry.get("ms") or entry.get("duration_ms")
            if name == "ar" and isinstance(ms, (int, float)):
                out["server_ar_ms"] = float(ms)
            elif name == "dit" and isinstance(ms, (int, float)):
                out["server_dit_ms"] = float(ms)
            elif name in ("kv_transfer", "kv") and isinstance(ms, (int, float)):
                out["server_kv_transfer_ms"] = float(ms)
    # Token-count and memory fields if exposed by the server.
    for src_key, dst_key in (
        ("ar_decoded_tokens", "ar_decoded_tokens"),
        ("ar_token_count", "ar_decoded_tokens"),
        ("peak_memory_mb", "gpu_mem_mb_peak"),
    ):
        val = body.get(src_key)
        if isinstance(val, (int, float)):
            out[dst_key] = int(val)
    return out


def run_sweep(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    sizes: list[str],
    bot_tasks: list[str],
    step_counts: list[int],
    seeds: list[int],
    guidance_scale: float,
    warmup: int,
    measure: int,
    commit_sha: str,
    out_path: Path,
    timeout: float,
) -> None:
    """Run the full sweep and write a CSV at `out_path`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DEFAULT_CSV_COLUMNS)

        total_cells = len(sizes) * len(bot_tasks) * len(step_counts) * len(seeds)
        cell_idx = 0
        for size in sizes:
            for bot_task in bot_tasks:
                for steps in step_counts:
                    for seed in seeds:
                        cell_idx += 1
                        print(
                            f"[{cell_idx}/{total_cells}] size={size} bot={bot_task} "
                            f"steps={steps} seed={seed}",
                            file=sys.stderr,
                            flush=True,
                        )
                        # Warmup (discarded).
                        for w in range(warmup):
                            code, body, err, e2e_ms = _one_measurement(
                                endpoint=endpoint,
                                model=model,
                                prompt=prompt,
                                bot_task=bot_task,
                                steps=steps,
                                size=size,
                                seed=seed,
                                guidance_scale=guidance_scale,
                                timeout=timeout,
                            )
                            row = MeasurementRow(
                                commit_sha=commit_sha,
                                timestamp=datetime.utcnow().isoformat(timespec="seconds"),
                                model=model,
                                bot_task=bot_task,
                                steps=steps,
                                size=size,
                                seed=seed,
                                run_idx=w,
                                kind="cold" if w == 0 else "warmup",
                                http_code=code,
                                e2e_ms=e2e_ms,
                                error=err[:120],
                            )
                            metrics = _extract_server_metrics(body)
                            for k, v in metrics.items():
                                setattr(row, k, v)
                            writer.writerow(row.as_csv())
                            f.flush()
                        # Measured runs.
                        for m in range(measure):
                            code, body, err, e2e_ms = _one_measurement(
                                endpoint=endpoint,
                                model=model,
                                prompt=prompt,
                                bot_task=bot_task,
                                steps=steps,
                                size=size,
                                seed=seed,
                                guidance_scale=guidance_scale,
                                timeout=timeout,
                            )
                            row = MeasurementRow(
                                commit_sha=commit_sha,
                                timestamp=datetime.utcnow().isoformat(timespec="seconds"),
                                model=model,
                                bot_task=bot_task,
                                steps=steps,
                                size=size,
                                seed=seed,
                                run_idx=m,
                                kind="warm",
                                http_code=code,
                                e2e_ms=e2e_ms,
                                error=err[:120],
                            )
                            metrics = _extract_server_metrics(body)
                            for k, v in metrics.items():
                                setattr(row, k, v)
                            writer.writerow(row.as_csv())
                            f.flush()

    print(f"\nWrote {out_path}", file=sys.stderr)
    print_summary(out_path)


def print_summary(csv_path: Path) -> None:
    """Print a quick {bot_task, steps} → median(warm e2e_ms) table."""
    by_cell: dict[tuple[str, int], list[float]] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("kind") != "warm":
                continue
            try:
                e2e = float(row["e2e_ms"])
            except (KeyError, ValueError):
                continue
            key = (row["bot_task"], int(row["steps"]))
            by_cell.setdefault(key, []).append(e2e)

    print("\n=== Warm-run median latency (ms) ===", file=sys.stderr)
    print(f"{'bot_task':<18} {'steps':>6} {'n':>4} {'median':>10} {'p99':>10}", file=sys.stderr)
    for (bot_task, steps), values in sorted(by_cell.items()):
        if not values:
            continue
        med = statistics.median(values)
        p99 = max(values) if len(values) < 100 else sorted(values)[int(len(values) * 0.99)]
        print(
            f"{bot_task:<18} {steps:>6} {len(values):>4} {med:>10.0f} {p99:>10.0f}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HunyuanImage-3 t2i latency sweep")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30010")
    parser.add_argument("--model", default="HunyuanImage-3.0-Instruct-Distil")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--size", nargs="+", default=["768x1024"])
    parser.add_argument(
        "--bot-task",
        nargs="+",
        default=["recaption", "think_recaption"],
        choices=["vanilla", "recaption", "think", "think_recaption"],
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[8, 28],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Cold-discard runs per cell (1st is 'cold', rest are 'warmup' both ignored by the median).",
    )
    parser.add_argument(
        "--measure",
        type=int,
        default=5,
        help="Warm measurement runs per cell; the median is the headline number.",
    )
    parser.add_argument(
        "--commit",
        default="",
        help="Commit SHA tag for this sweep (recorded in every row).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path. Parent directory is created if missing.",
    )
    args = parser.parse_args(argv)

    run_sweep(
        endpoint=args.endpoint.rstrip("/"),
        model=args.model,
        prompt=args.prompt,
        sizes=args.size,
        bot_tasks=args.bot_task,
        step_counts=args.steps,
        seeds=args.seeds,
        guidance_scale=args.guidance_scale,
        warmup=args.warmup,
        measure=args.measure,
        commit_sha=args.commit,
        out_path=args.out,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
