#!/usr/bin/env python3
"""Build CSVs + plots for the binding-grouped RG experiment.

Data captured from the live 4x H100 / Llama-3.1-70B run (job vs binding),
27 MAST/HyperAgent traces, 30 turns/job, 256 completion tokens, arrival 3.0/s.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# (key, label, job, binding)  -- summary metrics from both runs
METRICS = [
    ("system_e2e_latency_s",            "System E2E latency (s)",      147.06, 146.29),
    ("throughput_jobs_per_s",           "Throughput (jobs/s)",           0.18,   0.18),
    ("throughput_completion_tokens_per_s","Throughput (tok/s)",        740.22, 737.91),
    ("jct_mean_s",                      "JCT mean (s)",                102.42, 101.71),
    ("jct_p50_s",                       "JCT p50 (s)",                 107.61, 105.77),
    ("jct_p95_s",                       "JCT p95 (s)",                 133.36, 132.73),
    ("jct_p99_s",                       "JCT p99 (s)",                 142.82, 142.13),
    ("request_latency_p50_s",           "Req latency p50 (s)",           3.05,   3.07),
    ("request_latency_p95_s",           "Req latency p95 (s)",           6.22,   6.22),
    ("request_latency_p99_s",           "Req latency p99 (s)",           6.45,   6.41),
    ("gpu_util_mean_pct",               "GPU util mean (%)",            91.01,  90.89),
    ("gpu_util_p95_pct",                "GPU util p95 (%)",             94.80,  95.00),
    ("gpu_util_max_pct",                "GPU util max (%)",             95.25,  95.50),
]

CONFIG = [
    ("model", "meta-llama/Llama-3.1-70B-Instruct"),
    ("hardware", "4x NVIDIA H100 80GB (TP=4)"),
    ("vllm_version", "0.10.2 (continuum fork)"),
    ("scheduling_policy", "continuum"),
    ("rg_modes_compared", "job (baseline) vs binding (CONTINUUM_RG_MODE)"),
    ("traces", "27 MAST/HyperAgent SWE-bench (1 corrupt skipped)"),
    ("max_turns_per_job", 30),
    ("max_completion_tokens", 256),
    ("arrival_rate_jobs_per_s", 3.0),
    ("max_model_len", 32768),
    ("gpu_memory_utilization", 0.90),
    ("jobs_ok", "27 / 27"),
    ("jobs_failed", "1 (corrupt sphinx-8435 trace)"),
    ("requests_executed", "702 / 702"),
]

# ---- summary.csv ----
with open(os.path.join(HERE, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "job_baseline", "binding_rg", "delta_pct"])
    for _, label, j, b in METRICS:
        delta = round(100.0 * (b - j) / j, 2) if j else ""
        w.writerow([label, j, b, delta])

# ---- run_config.csv ----
with open(os.path.join(HERE, "run_config.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["parameter", "value"])
    for k, v in CONFIG:
        w.writerow([k, v])

# ---- plot helper ----
def grouped_bar(keys, fname, title, ylabel):
    sel = [m for m in METRICS if m[0] in keys]
    labels = [m[1] for m in sel]
    job = [m[2] for m in sel]
    binding = [m[3] for m in sel]
    x = range(len(sel))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.7 * len(sel)), 4.5))
    b1 = ax.bar([i - w / 2 for i in x], job, w, label="job (baseline)", color="#4C72B0")
    b2 = ax.bar([i + w / 2 for i in x], binding, w, label="binding (RG)", color="#DD8452")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, fname), dpi=130); plt.close(fig)

grouped_bar(["jct_mean_s", "jct_p50_s", "jct_p95_s", "jct_p99_s"],
            "jct_comparison.png", "Job Completion Time: job vs binding (70B, 4xH100)", "seconds")
grouped_bar(["request_latency_p50_s", "request_latency_p95_s", "request_latency_p99_s"],
            "request_latency.png", "Per-request latency: job vs binding", "seconds")
grouped_bar(["system_e2e_latency_s"],
            "system_e2e_latency.png", "System end-to-end latency (makespan)", "seconds")
grouped_bar(["gpu_util_mean_pct", "gpu_util_p95_pct", "gpu_util_max_pct"],
            "gpu_utilization.png", "GPU utilization: job vs binding", "percent")

# ---- delta plot (the headline: everything is ~noise) ----
labels = [m[1] for m in METRICS]
deltas = [100.0 * (m[3] - m[2]) / m[2] if m[2] else 0 for m in METRICS]
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#55A868" if d <= 0 else "#C44E52" for d in deltas]
ax.barh(labels, deltas, color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.axvspan(-2, 2, color="gray", alpha=0.15, label="+/-2% noise band")
ax.set_xlabel("binding vs job  (delta %, negative = binding better)")
ax.set_title("RG binding effect on all metrics (within noise)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "delta_overview.png"), dpi=130); plt.close(fig)

print("wrote summary.csv, run_config.csv, and", len(os.listdir(PLOTS)), "plots to", PLOTS)
