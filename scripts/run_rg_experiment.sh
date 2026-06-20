#!/usr/bin/env bash
# Run the binding-grouped continuum RG experiment on a RunPod GPU pod.
#
# Prereqs on the pod:
#   - PyTorch + CUDA 12.x -devel image (toolkit needed to install the vLLM fork)
#   - export HF_TOKEN=...        (gated Llama repo)
#   - run this on the SAME host as vllm serve (so nvidia-smi sees the GPUs)
#
# Usage:
#   cd /workspace/AgenticVLLM
#   export HF_TOKEN=hf_xxx
#   bash scripts/run_rg_experiment.sh
#
# Override defaults via env, e.g.:
#   MODEL=Qwen/Qwen2.5-Coder-14B-Instruct TP=1 MAXLEN=32768 \
#   ARRIVAL_RATES="0.5 1.0 2.0" bash scripts/run_rg_experiment.sh
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
TP="${TP:-4}"                       # tensor-parallel size (= #GPUs)
MAXLEN="${MAXLEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT="${PORT:-8000}"
ARRIVAL_RATES="${ARRIVAL_RATES:-1.0}"   # space-separated sweep
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"  # e.g. "--quantization fp8" for 2xGPU
TRACES="${TRACES:-hyperagent-replay/trajectories/*_human.json}"
OUTDIR="${OUTDIR:-runs}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

mkdir -p "$OUTDIR"

echo "=== Installing (one-time; skip if already installed) ==="
if ! python -c "import vllm" 2>/dev/null; then
  # VLLM_USE_PRECOMPILED reuses prebuilt CUDA kernels -> only Python is rebuilt.
  VLLM_USE_PRECOMPILED=1 pip install -e vllm-continuum
fi
pip install -e hyperagent-replay

wait_ready() {
  echo "Waiting for vLLM at $BASE_URL ..."
  for _ in $(seq 1 120); do
    if curl -sf "$BASE_URL/models" >/dev/null 2>&1; then echo "ready"; return 0; fi
    sleep 5
  done
  echo "ERROR: server did not become ready" >&2; return 1
}

run_mode() {  # $1 = job|binding
  local mode="$1"
  echo ""
  echo "############################################################"
  echo "### CONTINUUM_RG_MODE=$mode"
  echo "############################################################"
  CONTINUUM_RG_MODE="$mode" vllm serve "$MODEL" \
    --scheduling-policy continuum \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAXLEN" \
    $EXTRA_SERVE_ARGS > "$OUTDIR/server_${mode}.log" 2>&1 &
  local server_pid=$!
  trap 'kill $server_pid 2>/dev/null || true' RETURN
  wait_ready

  for rate in $ARRIVAL_RATES; do
    local out="$OUTDIR/${mode}_rate${rate}.json"
    echo "--- driver: mode=$mode arrival_rate=$rate -> $out ---"
    python hyperagent-replay/tools/concurrent_replay.py "$TRACES" \
      --model "$MODEL" \
      --base-url "$BASE_URL" \
      --arrival-rate "$rate" \
      --max-model-len "$MAXLEN" \
      --output "$out"
  done

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
  sleep 10   # let the port/GPU free before next mode
}

run_mode job
run_mode binding

echo ""
echo "=== COMPARISON (system E2E latency / JCT / GPU util) ==="
python - "$OUTDIR" "$ARRIVAL_RATES" <<'PY'
import glob, json, os, sys
outdir, rates = sys.argv[1], sys.argv[2].split()
keys = ["system_e2e_latency_s", "throughput_jobs_per_s",
        "jct_mean_s", "jct_p95_s", "jct_p99_s",
        "request_latency_p95_s", "gpu_util_mean_pct", "gpu_util_max_pct"]
for rate in rates:
    print(f"\n# arrival_rate = {rate}")
    print(f"{'metric':<32}{'job':>14}{'binding':>14}{'delta%':>10}")
    data = {}
    for mode in ("job", "binding"):
        fp = os.path.join(outdir, f"{mode}_rate{rate}.json")
        data[mode] = json.load(open(fp))["summary"] if os.path.exists(fp) else {}
    for k in keys:
        a = data["job"].get(k); b = data["binding"].get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
            d = 100.0 * (b - a) / a
            print(f"{k:<32}{a:>14.2f}{b:>14.2f}{d:>9.1f}%")
        else:
            print(f"{k:<32}{str(a):>14}{str(b):>14}{'':>10}")
PY
echo ""
echo "Raw per-run JSON in $OUTDIR/. For 'binding', PREFILL_BOUND vs DECODE_BOUND"
echo "is logged per request; lower JCT/E2E + higher GPU util vs 'job' = RG helped."
