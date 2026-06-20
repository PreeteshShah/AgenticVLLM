#!/usr/bin/env python3
"""Offline (GPU-free) preview of the prefill/decode binding split.

Estimates each turn's binding_type from the recorded trajectory using a
char-based token proxy, so we can see the PREFILL_BOUND vs DECODE_BOUND split
across the MAST/HyperAgent traces *before* spending RunPod time. The real
classification at replay uses identical logic (`binding.classify_binding`) fed
vLLM's actual prompt/cached/completion token counts.

Usage:
  python tools/binding_split_preview.py --glob "trajectories/*_human.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter

# Make `hyperagent_replay` importable from a source checkout without install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from hyperagent_replay.trace import parse_events  # noqa: E402
from hyperagent_replay import binding  # noqa: E402

CHARS_PER_TOKEN = 3.0


def est_tokens(text):
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def tool_io_seconds(action):
    """Mirror of replay.tool_delay_for_turn (heuristic policy)."""
    if not action:
        return 0.0
    tool_name = (action.get("tool_name") or "").lower()
    language = action.get("language")
    if language == "bash":
        if tool_name in {"pytest", "python"}:
            return 4.0
        if tool_name in {"sed", "cat", "grep", "ls"}:
            return 0.2
        return 0.5
    if language == "python":
        if tool_name.endswith("_run"):
            if any(n in tool_name for n in ("open_file", "search", "symbol", "folder")):
                return 0.15
            return 0.35
        return 1.0
    return 0.2


def turns_with_prefill(events, problem_statement_chars):
    """Yield (agent, action, new_prefill_tokens, decode_tokens, io_seconds).

    new_prefill_tokens approximates the *uncached* context the model reads on
    this turn: content from non-response events since the previous response
    (tool observations, subgoal/intern dispatch, logs). The first turn also
    carries the problem statement + seed prompt.
    """
    pending_prefill_chars = problem_statement_chars
    # First, attach actions to their preceding response (like build_turns).
    responses = []  # list of dicts {agent, content, action, prefill_chars}
    for ev in events:
        et = ev.get("type")
        if et == "response":
            responses.append({
                "agent": ev.get("agent", ""),
                "content": ev.get("content", ""),
                "action": None,
                "prefill_chars": pending_prefill_chars,
            })
            pending_prefill_chars = 0
        elif et == "action":
            if responses and responses[-1]["action"] is None:
                responses[-1]["action"] = {
                    "language": ev.get("language"),
                    "tool_name": ev.get("tool_name"),
                    "code": ev.get("code"),
                }
            # action code is also new context the next turn must read
            pending_prefill_chars += len(ev.get("code") or "")
        else:
            # observation / subgoal / intern_name / handoff / log
            pending_prefill_chars += len(ev.get("content") or "")

    for r in responses:
        yield (
            r["agent"],
            r["action"],
            est_tokens_from_chars(r["prefill_chars"]),
            est_tokens(r["content"]),
            tool_io_seconds(r["action"]),
        )


def est_tokens_from_chars(n_chars):
    if n_chars <= 0:
        return 0
    return max(1, math.ceil(n_chars / CHARS_PER_TOKEN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="trajectories/*_human.json")
    ap.add_argument("--decode-tps", type=float, default=binding.DEFAULT_DECODE_TPS)
    ap.add_argument("--prefill-tps", type=float, default=binding.DEFAULT_PREFILL_TPS)
    ap.add_argument("--per-trace", action="store_true",
                    help="Print a per-trace binding breakdown too")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit("No files matched %s" % args.glob)

    overall = Counter()
    overall_by_workload = Counter()
    binding_by_workload = {}  # workload -> Counter(binding)
    n_ok = 0
    n_bad = 0
    per_trace_rows = []

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            n_bad += 1
            print("  [skip] %s (%s)" % (os.path.basename(fp), str(e)[:50]))
            continue
        n_ok += 1
        trajectory = d.get("trajectory", [])
        problem_chars = len(json.dumps(d.get("problem_statement", "")))
        events = parse_events(trajectory)

        trace_counter = Counter()
        for agent, action, prefill_tok, decode_tok, io_s in turns_with_prefill(
                events, problem_chars):
            tool = action.get("tool_name") if action else None
            b = binding.classify_binding(
                new_prefill_tokens=prefill_tok,
                decode_tokens=decode_tok,
                io_seconds=io_s,
                has_action=action is not None,
                decode_tps=args.decode_tps,
                prefill_tps=args.prefill_tps,
            )
            wl = binding.workload_type(agent, tool)
            overall[b] += 1
            trace_counter[b] += 1
            overall_by_workload[wl] += 1
            binding_by_workload.setdefault(wl, Counter())[b] += 1

        per_trace_rows.append((os.path.basename(fp), trace_counter))

    total_turns = sum(overall.values())
    print("\n=== Binding cost model: prefill_tps=%.0f decode_tps=%.0f (ratio %.0fx) ==="
          % (args.prefill_tps, args.decode_tps, args.prefill_tps / args.decode_tps))
    print("Traces parsed: %d ok, %d skipped | total LLM turns: %d\n"
          % (n_ok, n_bad, total_turns))

    print("=== Overall binding_type split ===")
    order = [binding.PREFILL_BOUND, binding.DECODE_BOUND, binding.IO_BOUND,
             binding.MIXED, binding.LLM_ONLY]
    for b in order:
        c = overall.get(b, 0)
        pct = (100.0 * c / total_turns) if total_turns else 0.0
        print("  %-14s %6d  (%5.1f%%)" % (b, c, pct))

    print("\n=== binding_type x workload_type (turn counts) ===")
    wl_order = sorted(overall_by_workload, key=lambda w: -overall_by_workload[w])
    header = "%-12s" % "workload" + "".join("%14s" % b for b in order) + "%10s" % "total"
    print(header)
    for wl in wl_order:
        row = "%-12s" % wl
        bc = binding_by_workload.get(wl, Counter())
        for b in order:
            row += "%14d" % bc.get(b, 0)
        row += "%10d" % overall_by_workload[wl]
        print(row)

    if args.per_trace:
        print("\n=== Per-trace binding split ===")
        for name, c in per_trace_rows:
            tot = sum(c.values())
            parts = " ".join("%s=%d" % (b, c.get(b, 0)) for b in order if c.get(b, 0))
            print("  %-45s n=%-4d %s" % (name[:45], tot, parts))


if __name__ == "__main__":
    main()