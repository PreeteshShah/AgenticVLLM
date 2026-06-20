#!/usr/bin/env python3
"""Drive many HyperAgent traces as CONCURRENT continuum jobs against vLLM.

The stock `ha-trace-batch-replay` runs one trace at a time, so only one job is
ever in flight and the continuum scheduler never has to choose between jobs --
which means binding-grouped admission (CONTINUUM_RG_MODE=binding) has nothing
to group. This driver launches each trace as its own concurrent job at a
controllable arrival rate, so the scheduler is actually under contention, and
reports end-to-end latency / JCT so baseline vs RG modes can be compared.

Run the SAME command twice against two server configs to A/B:
  baseline: vllm serve ... --scheduling-policy continuum   (CONTINUUM_RG_MODE=job)
  rg:       CONTINUUM_RG_MODE=binding vllm serve ... --scheduling-policy continuum

Example:
  python tools/concurrent_replay.py "trajectories/*_human.json" \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --base-url http://127.0.0.1:8000/v1 \
    --arrival-rate 0.2 --max-model-len 32768 \
    --output runs/rg_binding.json
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import subprocess
import threading
import time
from pathlib import Path

from openai import OpenAI

from hyperagent_replay.replay import (
    DEFAULT_CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_CONTEXT_SAFETY_MARGIN,
    DEFAULT_MIN_REFERENCE_CHARS,
    ENGINE_MODE_CONTINUUM,
    percentile,
)
from hyperagent_replay.replay_reuse import (
    load_trace_with_subgoals,
    replay_trace_with_reuse,
)


class GpuSampler:
    """Background sampler of per-GPU utilization via `nvidia-smi`.

    Tolerates a missing/failing nvidia-smi (returns no samples) so the driver
    still runs on CPU-only boxes. Captures SM utilization (%) and memory used
    (MiB) for every visible GPU at a fixed interval.
    """

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.util_samples: list[float] = []   # mean across GPUs per tick
        self.util_per_gpu: dict[int, list[float]] = {}
        self.mem_used_mib: list[float] = []    # max across GPUs per tick
        self.available = interval_s > 0 and self._probe()

    @staticmethod
    def _probe() -> bool:
        try:
            subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                           timeout=5, check=True)
            return True
        except Exception:
            return False

    def _read_once(self) -> None:
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True).stdout
        except Exception:
            return
        utils = []
        mems = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx, util, mem = int(parts[0]), float(parts[1]), float(parts[2])
            utils.append(util)
            mems.append(mem)
            self.util_per_gpu.setdefault(idx, []).append(util)
        if utils:
            self.util_samples.append(sum(utils) / len(utils))
            self.mem_used_mib.append(max(mems))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._read_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2)

    def summary(self) -> dict:
        if not self.util_samples:
            return {"gpu_sampling": "unavailable"}
        s = sorted(self.util_samples)
        return {
            "gpu_sampling": "ok",
            "num_gpu_samples": len(s),
            "num_gpus": len(self.util_per_gpu),
            "gpu_util_mean_pct": sum(s) / len(s),
            "gpu_util_p50_pct": percentile(s, 50),
            "gpu_util_p95_pct": percentile(s, 95),
            "gpu_util_max_pct": max(s),
            "gpu_mem_used_max_mib": max(self.mem_used_mib),
        }


def run_one_job(
    *,
    input_path: Path,
    args: argparse.Namespace,
    results: dict,
    lock: threading.Lock,
) -> None:
    try:
        trace, subgoals = load_trace_with_subgoals(input_path)
    except Exception as exc:  # corrupt trace (e.g. sphinx-8435) -> skip
        with lock:
            results[input_path.name] = {"error": f"load_failed: {exc}"}
        print(f"[skip] {input_path.name}: {exc}", flush=True)
        return

    client = OpenAI(base_url=args.base_url, api_key=args.api_key,
                    timeout=max(args.timeout_s, 30.0))
    t0 = time.time()
    try:
        replay = replay_trace_with_reuse(
            trace=trace,
            subgoals=subgoals,
            client=client,
            model_name=args.model,
            context_mode=args.context_mode,
            slo_class=args.slo_class,
            max_reference_chars=args.max_reference_chars,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            seed=args.seed,
            delay_policy=args.delay_policy,
            constant_delay=args.constant_delay,
            max_model_len=args.max_model_len,
            context_safety_margin=DEFAULT_CONTEXT_SAFETY_MARGIN,
            min_reference_chars=DEFAULT_MIN_REFERENCE_CHARS,
            chars_per_token_estimate=DEFAULT_CHARS_PER_TOKEN_ESTIMATE,
            show_top=0,
            engine_mode=ENGINE_MODE_CONTINUUM,
            job_id_override=trace["instance_id"],
        )
    except Exception as exc:
        with lock:
            results[input_path.name] = {"error": f"replay_failed: {exc}"}
        print(f"[fail] {input_path.name}: {exc}", flush=True)
        return

    timing = replay["timing"]
    per_turn_latencies = [
        t["request_latency_s"] for t in replay["turn_metrics"]
        if t.get("executed_on_vllm")
    ]
    with lock:
        results[input_path.name] = {
            "instance_id": replay["instance_id"],
            "wall_solve_time_s": timing["wall_solve_time_s"],
            "lm_only_solve_time_s": timing["lm_only_solve_time_s"],
            "num_replayed_turns": timing["num_replayed_turns"],
            "num_vllm_requests_executed": timing["num_vllm_requests_executed"],
            "total_prompt_tokens": timing.get("total_prompt_tokens", 0),
            "total_completion_tokens": timing.get("total_completion_tokens", 0),
            "request_latencies_s": per_turn_latencies,
            "job_started_wall": t0,
            "job_finished_wall": time.time(),
        }
    print(f"[done] {input_path.name}: JCT={timing['wall_solve_time_s']:.1f}s "
          f"turns={timing['num_replayed_turns']}", flush=True)


def aggregate(results: dict, experiment_t0: float, experiment_t1: float) -> dict:
    ok = [r for r in results.values() if "error" not in r]
    jcts = [r["wall_solve_time_s"] for r in ok]
    all_lat = [x for r in ok for x in r["request_latencies_s"]]
    makespan = experiment_t1 - experiment_t0
    total_completion = sum(r.get("total_completion_tokens", 0) for r in ok)
    total_prompt = sum(r.get("total_prompt_tokens", 0) for r in ok)
    return {
        "num_jobs_ok": len(ok),
        "num_jobs_failed": len(results) - len(ok),
        # System end-to-end latency: wall time to drain all concurrent jobs.
        "system_e2e_latency_s": makespan,
        "throughput_jobs_per_s": (len(ok) / makespan) if makespan > 0 else 0.0,
        "throughput_completion_tokens_per_s":
            (total_completion / makespan) if makespan > 0 else 0.0,
        # Per-job completion time (JCT).
        "jct_mean_s": (sum(jcts) / len(jcts)) if jcts else 0.0,
        "jct_p50_s": percentile(jcts, 50) if jcts else 0.0,
        "jct_p95_s": percentile(jcts, 95) if jcts else 0.0,
        "jct_p99_s": percentile(jcts, 99) if jcts else 0.0,
        # Per-request (per-turn) latency.
        "request_latency_p50_s": percentile(all_lat, 50) if all_lat else 0.0,
        "request_latency_p95_s": percentile(all_lat, 95) if all_lat else 0.0,
        "request_latency_p99_s": percentile(all_lat, 99) if all_lat else 0.0,
        "num_requests": len(all_lat),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", help="Glob of raw/extracted trace JSON files")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--arrival-rate", type=float, default=0.2,
                    help="Mean job arrivals per second (Poisson)")
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--slo-class", default="interactive",
                    choices=["interactive", "batch"])
    ap.add_argument("--context-mode", default="per_agent",
                    choices=["flattened", "per_agent"])
    ap.add_argument("--max-reference-chars", type=int, default=4000)
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-completion-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--delay-policy", default="heuristic",
                    choices=["none", "constant", "heuristic"])
    ap.add_argument("--constant-delay", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--gpu-sample-interval-s", type=float, default=1.0,
                    help="Poll nvidia-smi every N seconds (0 disables)")
    ap.add_argument("--output", type=Path, default=Path("concurrent_replay.json"))
    args = ap.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.inputs))
    if args.max_jobs is not None:
        files = files[:args.max_jobs]
    if not files:
        raise SystemExit(f"No files matched {args.inputs}")

    print(f"Launching {len(files)} concurrent jobs at "
          f"~{args.arrival_rate} jobs/s against {args.base_url}", flush=True)

    rng = random.Random(args.seed)
    results: dict = {}
    lock = threading.Lock()
    threads: list[threading.Thread] = []

    gpu = GpuSampler(args.gpu_sample_interval_s)
    if args.gpu_sample_interval_s > 0 and not gpu.available:
        print("[warn] nvidia-smi unavailable; GPU utilization will not be "
              "reported.", flush=True)
    gpu.start()
    experiment_t0 = time.time()
    for input_path in files:
        th = threading.Thread(
            target=run_one_job,
            kwargs=dict(input_path=input_path, args=args,
                        results=results, lock=lock),
            daemon=True,
        )
        th.start()
        threads.append(th)
        # Poisson arrivals: exponential inter-arrival gap before next launch.
        if args.arrival_rate > 0:
            time.sleep(rng.expovariate(args.arrival_rate))

    for th in threads:
        th.join()
    experiment_t1 = time.time()
    gpu.stop()

    summary = aggregate(results, experiment_t0, experiment_t1)
    summary.update(gpu.summary())
    out = {
        "settings": {
            "model": args.model,
            "base_url": args.base_url,
            "arrival_rate": args.arrival_rate,
            "num_jobs": len(files),
            "slo_class": args.slo_class,
            "max_completion_tokens": args.max_completion_tokens,
            "delay_policy": args.delay_policy,
            "note": "Set CONTINUUM_RG_MODE on the server (job|binding) to A/B.",
        },
        "summary": summary,
        "per_job": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print("\n=== Concurrent replay summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
