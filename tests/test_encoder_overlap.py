# SPDX-License-Identifier: Apache-2.0
"""Encoder-only overlap benchmark.

Tests CPU/GPU overlap for encoder stages using thread pool + wait_result.
Compares blocking (no thread pool) vs concurrent (thread pool + wait_result) modes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logging.getLogger("sglang_omni").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    mode: str
    num_requests: int
    total_time: float
    latencies: list[float] = field(default_factory=list)

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def throughput(self) -> float:
        return self.num_requests / self.total_time if self.total_time > 0 else 0.0


def get_video_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(script_dir, "data/draw.mp4")
    return local if os.path.exists(local) else "tests/data/draw.mp4"


def create_request(
    video_path: str, prompt: str, idx: int, fps: float = 2.0
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"{prompt} [req{idx}-{uuid.uuid4().hex[:6]}]"}
        ],
        "videos": [video_path],
        "video_fps": fps,
    }


async def run_encoder_benchmark(
    args: argparse.Namespace, use_thread_pool: bool
) -> BenchmarkResult:
    """Run encoder-only benchmark with concurrent requests.

    Args:
        args: Command line arguments.
        use_thread_pool: If True, use thread pool for CPU work enabling true overlap.
                        If False, CPU work blocks event loop (baseline).
    """
    from sglang_omni.engines.omni import create_encoder_engine
    from sglang_omni.executors import EngineExecutor, FrontendExecutor
    from sglang_omni.models.qwen3_omni.components.frontend import Qwen3OmniFrontend
    from sglang_omni.models.qwen3_omni.components.image_encoder import (
        Qwen3OmniImageEncoder,
    )
    from sglang_omni.models.qwen3_omni.pipeline.engine_io import build_encoder_request
    from sglang_omni.models.qwen3_omni.pipeline.next_stage import IMAGE_STAGE
    from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state
    from sglang_omni.proto import OmniRequest, StagePayload

    video_path = get_video_path()
    mode = "Concurrent" if use_thread_pool else "Blocking"
    device = args.image_device
    dtype = args.dtype

    # Create frontend
    frontend = Qwen3OmniFrontend(model_id=args.model_id)

    def frontend_fn(payload: StagePayload) -> StagePayload:
        return frontend(payload)

    # Create image encoder
    image_model = Qwen3OmniImageEncoder(
        model_id=args.model_id, device=device, dtype=dtype
    )
    engine = create_encoder_engine(image_model, device=device)

    def request_builder(payload: StagePayload):
        state = load_state(payload)
        return build_encoder_request(state, stage_name=IMAGE_STAGE)

    def result_builder(payload: StagePayload, result: Any) -> StagePayload:
        # Just return the payload, we don't need to process the result
        return payload

    # Build executors
    cpu_executor = FrontendExecutor(
        frontend_fn,
        use_thread_pool=use_thread_pool,
        max_workers=4 if use_thread_pool else 1,
    )
    gpu_executor = EngineExecutor(
        engine=engine,
        request_builder=request_builder,
        result_builder=result_builder,
    )

    # Prepare requests
    requests = [
        create_request(video_path, args.prompt, i, args.video_fps)
        for i in range(args.num_requests)
    ]

    async def process_single(idx: int, req: dict) -> float:
        """Process single request with CPU/GPU overlap.

        Uses add_request return value for CPU and wait_result=True for GPU.
        This enables true parallel processing without result mix-up.
        """
        t0 = time.perf_counter()
        payload = StagePayload(
            request_id=f"req-{mode.lower()}-{idx}-{uuid.uuid4().hex[:6]}",
            request=OmniRequest(inputs=req, params={}),
            data={},
        )
        # CPU: use return value directly
        cpu_result = await cpu_executor.add_request(payload)
        assert cpu_result is not None
        # GPU: wait_result=True ensures each coroutine gets its own result
        await gpu_executor.add_request(cpu_result, wait_result=True)
        return time.perf_counter() - t0

    # Warmup
    if args.warmup:
        logger.info(f"[{mode}] Warming up...")
        await cpu_executor.start()
        await gpu_executor.start()
        warmup_payload = StagePayload(
            request_id=f"warmup-{uuid.uuid4().hex[:6]}",
            request=OmniRequest(
                inputs=create_request(video_path, args.prompt, -1, args.video_fps),
                params={},
            ),
            data={},
        )
        cpu_result = await cpu_executor.add_request(warmup_payload)
        assert cpu_result is not None
        await gpu_executor.add_request(cpu_result, wait_result=True)
    else:
        await cpu_executor.start()
        await gpu_executor.start()

    # Benchmark - submit all requests concurrently
    logger.info(f"[{mode}] Running {args.num_requests} requests concurrently...")
    start = time.perf_counter()

    tasks = [process_single(i, req) for i, req in enumerate(requests)]
    latencies = await asyncio.gather(*tasks)
    total = time.perf_counter() - start

    logger.info(f"[{mode}] Complete: {total:.3f}s")

    # Cleanup
    await gpu_executor.stop()
    await cpu_executor.stop()

    return BenchmarkResult(
        mode=mode,
        num_requests=len(requests),
        total_time=total,
        latencies=list(latencies),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encoder-only overlap benchmark")
    p.add_argument("--model-id", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--prompt", default="What do you see in the video?")
    p.add_argument("--num-requests", type=int, default=10)
    p.add_argument("--video-fps", type=float, default=2.0)
    p.add_argument("--image-device", default="cuda:0")
    p.add_argument("--warmup", action="store_true", default=True)
    p.add_argument("--no-warmup", action="store_false", dest="warmup")
    p.add_argument("--blocking-only", action="store_true")
    p.add_argument("--concurrent-only", action="store_true")
    return p.parse_args()


def print_results(
    baseline: BenchmarkResult | None, optimized: BenchmarkResult | None
) -> None:
    print("\n" + "=" * 80)
    print("ENCODER-ONLY BENCHMARK RESULTS".center(80))
    print("=" * 80)

    if baseline and optimized:
        speedup = baseline.total_time / optimized.total_time
        saved = baseline.total_time - optimized.total_time

        print(f"{'Mode':<15} {'Total':>10} {'Avg':>10} {'Throughput':>12}")
        print("-" * 80)
        print(
            f"{baseline.mode:<15} {baseline.total_time:9.2f}s {baseline.avg_latency:9.3f}s {baseline.throughput:11.2f}/s"
        )
        print(
            f"{optimized.mode:<15} {optimized.total_time:9.2f}s {optimized.avg_latency:9.3f}s {optimized.throughput:11.2f}/s"
        )
        print("-" * 80)
        print(
            f"Speedup: {speedup:.2f}x | Time saved: {saved:.2f}s ({saved/baseline.total_time*100:.1f}%)"
        )

        if speedup >= 1.5:
            print(f"✓ Excellent! Concurrent mode achieves {speedup:.2f}x speedup")
        elif speedup >= 1.2:
            print(f"✓ Good improvement ({speedup:.2f}x speedup)")
        else:
            print(f"⚠ Modest improvement ({speedup:.2f}x speedup)")
    elif baseline:
        print(
            f"Blocking: {baseline.total_time:.2f}s total, {baseline.avg_latency:.3f}s avg, {baseline.throughput:.2f} req/s"
        )
    elif optimized:
        print(
            f"Concurrent: {optimized.total_time:.2f}s total, {optimized.avg_latency:.3f}s avg, {optimized.throughput:.2f} req/s"
        )

    print("=" * 80)


async def main() -> None:
    args = parse_args()

    print("\n" + "=" * 80)
    print("ENCODER-ONLY OVERLAP BENCHMARK".center(80))
    print("=" * 80)
    print(f"Requests: {args.num_requests} | Device: {args.image_device}")
    print(
        "Comparing: Blocking (no thread pool) vs Concurrent (thread pool + wait_result)"
    )
    print("=" * 80)

    baseline = (
        None
        if args.concurrent_only
        else await run_encoder_benchmark(args, use_thread_pool=False)
    )
    optimized = (
        None
        if args.blocking_only
        else await run_encoder_benchmark(args, use_thread_pool=True)
    )

    print_results(baseline, optimized)


if __name__ == "__main__":
    asyncio.run(main())
