#!/usr/bin/env python3
"""
Derive estimated SLOs + resource groups from HyperAgent trajectory JSONs (no timestamps).

Resource groups are primarily by (sub_agent, tool_signature) — sub-agent grouping as requested,
with tool calls inside sub-agent used to refine performance homogeneity (QLM-style).

Outputs:
- Stage-level SLOs per resource group (p50/p95/p99 of EstimatedStageCost)
- Episode-level SLOs (p50/p95/p99 of sum(stage_cost))
- Similarity metrics:
  - repeated tool+args within episodes
  - repeated stages (same sub_agent+subgoal+tool_signature) within episodes
  - top repeated tool calls per resource group
  - AvoidableCost (proxy): redundant_calls * assumed tool latency
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from hyperagent_replay.resource_groups import normalize_space

# ----------------------------
# Parsing patterns (trace format)
# ----------------------------
INTERN_RE = re.compile(r"^Intern Name:\s*(.+?)\s*$")
SUBGOAL_RE = re.compile(r"^Subgoal:\s*(.+?)\s*$")
CODE_FENCE_RE = re.compile(r"^\s*```(\w+)?\s*$")          # ```python / ```bash / ```
TOOL_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\._run\s*\(") # open_file._run( ... )
# Capture tool name + argument string (best-effort) for redundancy metrics
TOOL_CALL_WITH_ARGS_RE = re.compile(r"\b([A-Za-z_]\w*)\._run\s*\((.*)\)\s*$")


# ----------------------------
# Defaults for proxy cost model
# (not real seconds; consistent units)
# ----------------------------
DEFAULT_DECODE_TPS = 60.0     # "output token throughput" proxy


DEFAULT_PREFILL_TPS = 400.0   # we dont actually use this 

DEFAULT_TOOL_LAT = {
    "open_file": 0.20,
    "open_file_gen": 0.35,
    "code_search": 0.45,
    "get_folder_structure": 0.25,
    "get_all_symbols": 0.40,
    "editor": 0.60,
    "bash_exec": 2.0,      
    "__default__": 0.40,
}


def build_token_counter(tokenizer: str, tokenizer_model: Optional[str]):
    """
    Returns a function f(text) -> int.

    tokenizer:
      - whitespace: current behavior (fast, no deps)
      - tiktoken: OpenAI BPE tokenizer (fast, local; pip install tiktoken)
      - hf: Hugging Face tokenizer (pip install transformers; may download model files)

    tokenizer_model:
      - tiktoken: encoding name or model name (e.g., "cl100k_base" or "gpt-4o-mini")
      - hf: model id / path (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    """
    if tokenizer == "whitespace":
        ws_re = re.compile(r"\S+")
        return lambda text: len(ws_re.findall(text))

    if tokenizer == "tiktoken":
        try:
            import tiktoken  # type: ignore
        except Exception as e:
            raise SystemExit(
                "Tokenizer 'tiktoken' requested but tiktoken is not installed.\n"
                "Install with: pip install tiktoken"
            ) from e

        if tokenizer_model:
            # If user gave a model name, try encoding_for_model; else treat as encoding name.
            try:
                enc = tiktoken.encoding_for_model(tokenizer_model)
            except Exception:
                enc = tiktoken.get_encoding(tokenizer_model)
        else:
            enc = tiktoken.get_encoding("cl100k_base")

        return lambda text: len(enc.encode(text))

    if tokenizer == "hf":
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:
            raise SystemExit(
                "Tokenizer 'hf' requested but transformers is not installed.\n"
                "Install with: pip install transformers"
            ) from e

        if not tokenizer_model:
            raise SystemExit("Tokenizer 'hf' requires --tokenizer-model <hf_model_id_or_path>.")

        tok = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
        return lambda text: len(tok.encode(text, add_special_tokens=False))

    raise SystemExit(f"Unknown tokenizer: {tokenizer}")


def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


@dataclass
class Stage:
    file: str
    instance_id: str
    sub_agent: str
    subgoal: str
    # aggregated text outside code blocks (LLM output + narrative)
    text: str
    # code blocks (we keep separately for args parsing)
    code_blocks: List[Tuple[str, str]]  # (lang, block_text)
    tool_counts: Counter
    tool_calls_with_args: Counter
    text_tokens: int = 0
    obs_tokens: int = 0
    est_cost: float = 0.0
    tool_signature: Tuple[str, ...] = ()


def _ingest_tool_line(stage: Stage, line: str) -> None:
    """Fallback tool detector for lines outside fenced code blocks."""
    # Count tool names
    for tool in TOOL_CALL_RE.findall(line):
        stage.tool_counts[tool] += 1

    # Best-effort tool+args redundancy key
    m = TOOL_CALL_WITH_ARGS_RE.search(line.strip())
    if m:
        tool = m.group(1)
        args = normalize_space(m.group(2))
        stage.tool_calls_with_args[f"{tool}({args})"] += 1


def parse_stages_from_trajectory(
    trajectory: List[str],
    file: str,
    instance_id: str,
) -> List[Stage]:
    stages: List[Stage] = []
    cur: Optional[Stage] = None
    in_code = False
    code_lang = ""
    code_lines: List[str] = []

    def flush_code_block():
        nonlocal code_lang, code_lines
        if cur is None:
            code_lang = ""
            code_lines = []
            return
        block = "\n".join(code_lines).strip()
        if block:
            cur.code_blocks.append((code_lang, block))
        code_lang = ""
        code_lines = []

    def flush_stage():
        nonlocal cur
        if cur is not None:
            stages.append(cur)
        cur = None

    for line in trajectory:
        mi = INTERN_RE.match(line)
        if mi:
            # close previous stage
            if in_code:
                flush_code_block()
                in_code = False
            flush_stage()
            cur = Stage(
                file=file,
                instance_id=instance_id,
                sub_agent=mi.group(1).strip(),
                subgoal="",
                text="",
                code_blocks=[],
                tool_counts=Counter(),
                tool_calls_with_args=Counter(),
            )
            continue

        if cur is None:
            continue

        ms = SUBGOAL_RE.match(line)
        if ms and not cur.subgoal:
            cur.subgoal = ms.group(1).strip()
            continue

        mf = CODE_FENCE_RE.match(line)
        if mf:
            if not in_code:
                in_code = True
                code_lang = (mf.group(1) or "").lower()
                code_lines = []
            else:
                in_code = False
                flush_code_block()
            continue

        if in_code:
            code_lines.append(line)
        else:
            # Everything outside code fences we treat as "text/obs" proxy.
            cur.text += line + "\n"

            # Fallback: some traces include tool calls as plain lines outside fenced code blocks.
            if "._run" in line:
                _ingest_tool_line(cur, line)

    # flush tail
    if in_code:
        flush_code_block()
    flush_stage()
    return stages


def extract_tools(stage: Stage) -> None:
    """
    Populate stage.tool_counts and stage.tool_calls_with_args from code blocks.
    NOTE: parse_stages_from_trajectory also ingests tool calls outside code fences.
    """
    for lang, block in stage.code_blocks:
        # Bash execution is its own "tool" bucket.
        if lang in ("bash", "sh", "zsh", "shell"):
            stage.tool_counts["bash_exec"] += 1
            stage.tool_calls_with_args["bash_exec()"] += 1
            continue

        # Otherwise look for tool._run(...) invocations.
        for raw_line in block.splitlines():
            raw_line = raw_line.strip()

            for tool in TOOL_CALL_RE.findall(raw_line):
                stage.tool_counts[tool] += 1

            m = TOOL_CALL_WITH_ARGS_RE.search(raw_line)
            if m:
                tool = m.group(1)
                args = normalize_space(m.group(2))
                stage.tool_calls_with_args[f"{tool}({args})"] += 1


def estimate_cost(
    stage: Stage,
    decode_tps: float,
    prefill_tps: float,
    tool_lat: Dict[str, float],
    token_count,
) -> float:
    """
    QLM-style decomposition (proxy):
      completion ~= decode_cost + prefill/obs_cost + tool_cost

    Without timestamps:
      - treat stage.text as "output-like" (decode proxy)
      - obs_tokens remains 0 unless you later extract tool outputs
      - tool cost from detected tool counts
    """
    stage.text_tokens = token_count(stage.text)
    stage.obs_tokens = 0

    cost = (stage.text_tokens / decode_tps) + (stage.obs_tokens / prefill_tps)

    for tool, cnt in stage.tool_counts.items():
        cost += cnt * tool_lat.get(tool, tool_lat.get("__default__", 0.40))

    return cost


def stage_rg_key(stage: Stage, slo_class: str, rg_version: str = "legacy") -> Tuple[Any, Any, str]:
    """
    Resource group key.

    legacy: RG = (sub_agent, tool_signature, slo_class)
    v2:     RG = (workload_type, binding_type, slo_class)

    NOTE: the v2 binding here is an OFFLINE estimate. This script does not
    reconstruct per-turn prompts, so new-prefill tokens are unmodeled (treated
    as 0) and binding is decode/io-only -- it will under-report PREFILL_BOUND.
    The faithful prefill/decode split comes from tools/binding_split_preview.py
    (context-delta estimate) and from the live replay (real vLLM usage tokens).
    The 3-arity is preserved so the rest of the report plumbing is unchanged.
    """
    if rg_version == "v2":
        from hyperagent_replay import binding
        primary_tool = None
        if stage.tool_counts:
            primary_tool = max(
                (t for t in stage.tool_counts if t != "__default__"),
                key=lambda t: stage.tool_counts[t],
                default=None,
            )
        io_seconds = sum(
            cnt * DEFAULT_TOOL_LAT.get(t, DEFAULT_TOOL_LAT["__default__"])
            for t, cnt in stage.tool_counts.items())
        binding_type = binding.classify_binding(
            new_prefill_tokens=0,
            decode_tokens=stage.text_tokens,
            io_seconds=io_seconds,
            has_action=bool(stage.tool_counts),
        )
        workload = binding.workload_type(stage.sub_agent, primary_tool)
        return (workload, binding_type, slo_class)
    return (stage.sub_agent, stage.tool_signature, slo_class)


def stage_signature_for_repetition(stage: Stage) -> Tuple[str, str, Tuple[str, ...]]:
    # Robust: use actual detected tools, not the precomputed signature alone.
    tool_sig = tuple(sorted(t for t in stage.tool_counts.keys() if t != "__default__"))
    return (stage.sub_agent, stage.subgoal or "", tool_sig)


def print_table(rows: List[List[Any]], headers: List[str], out=sys.stdout) -> None:
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(v)))
    fmt = " | ".join("{:" + str(w) + "}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)
    print(fmt.format(*headers), file=out)
    print(sep, file=out)
    for r in rows:
        print(fmt.format(*[str(v) for v in r]), file=out)


def build_single_file_report_paths(files: List[str], output_dir: str) -> Dict[str, str]:
    stem_counts = Counter(os.path.splitext(os.path.basename(fp))[0] for fp in files)
    common_root = os.path.commonpath(files)
    report_paths: Dict[str, str] = {}

    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        if stem_counts[stem] > 1:
            stem = os.path.splitext(os.path.relpath(fp, common_root))[0].replace(os.sep, "__")
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
        report_paths[fp] = os.path.join(output_dir, f"{safe_stem}_single_file_slos.txt")

    return report_paths


def print_file_similarity_section(
    file: str,
    episode_total: float,
    file_redundancy_by_rg: Dict[Tuple[str, Tuple[str, ...], str], Counter],
    file_avoidable_cost_by_rg: Dict[Tuple[str, Tuple[str, ...], str], float],
    repeated_stage_counts: Counter,
    show_top: int,
    include_file_header: bool = True,
    out=sys.stdout,
) -> None:
    if include_file_header:
        print(f"\n=== File: {os.path.basename(file)} ===", file=out)
    print(f"Episode total: {episode_total:.2f}", file=out)

    print("\n=== Similarity metric 1: repeated tool calls ===", file=out)
    red_rows = []
    for rg, cnts in file_redundancy_by_rg.items():
        if not cnts:
            continue
        sa, tool_sig, slo_class = rg
        total_redundant = sum(cnts.values())
        red_rows.append([
            sa,
            "+".join(tool_sig) if tool_sig else "LLM_ONLY",
            slo_class,
            total_redundant,
            f"{file_avoidable_cost_by_rg[rg]:.2f}",
        ])

    red_rows.sort(key=lambda r: r[3], reverse=True)
    if red_rows:
        print_table(
            red_rows[:show_top],
            headers=["SubAgent", "ToolSignature", "SLOClass", "RedundantCalls", "AvoidableCost"],
            out=out,
        )
    else:
        print("No repeated tool+args patterns detected (or tool args not captured in this trace).", file=out)

    print("\nTop redundant call patterns (by RG):", file=out)
    top_redundancy = [
        (rg, cnts)
        for rg, cnts in sorted(
            file_redundancy_by_rg.items(),
            key=lambda kv: sum(kv[1].values()),
            reverse=True,
        )
        if cnts
    ][:5]
    if top_redundancy:
        for rg, _ in top_redundancy:
            sa, tool_sig, slo_class = rg
            top = file_redundancy_by_rg[rg].most_common(5)
            print(
                f"\nRG=({sa}, {('+'.join(tool_sig) if tool_sig else 'LLM_ONLY')}, {slo_class})"
                f"  avoidable_cost={file_avoidable_cost_by_rg[rg]:.2f}",
                file=out,
            )
            for k, v in top:
                print(f"  +{v}  {k}", file=out)
    else:
        print("None.", file=out)

    print("\nTop repeated stage signatures:", file=out)
    if repeated_stage_counts:
        for (sa, subgoal, tool_sig), cnt in repeated_stage_counts.most_common(10):
            print(
                f"  +{cnt}  sub_agent={sa}  tool_sig={('+'.join(tool_sig) if tool_sig else 'LLM_ONLY')}"
                f"  subgoal={subgoal[:120]}",
                file=out,
            )
    else:
        print("No repeated stage signatures detected.", file=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/mnt/data/*.json", help="Glob for trace JSON files")
    ap.add_argument("--slo-class", default="interactive", choices=["interactive", "batch"], help="SLO class label")
    ap.add_argument("--rg-version", default="legacy", choices=["legacy", "v2"],
                    help="Resource-group key: legacy=(sub_agent,tool_signature,slo_class), "
                         "v2=(workload_type,binding_type,slo_class)")
    ap.add_argument("--decode-tps", type=float, default=DEFAULT_DECODE_TPS)
    ap.add_argument("--prefill-tps", type=float, default=DEFAULT_PREFILL_TPS)
    ap.add_argument("--show-top", type=int, default=20, help="How many top groups/patterns to display")

    # Tokenizer options
    ap.add_argument(
        "--tokenizer",
        default="whitespace",
        choices=["whitespace", "tiktoken", "hf"],
        help="Token counting method",
    )
    ap.add_argument(
        "--tokenizer-model",
        default=None,
        help="Tokenizer model/encoding name. "
             "For tiktoken: encoding or model name (e.g., cl100k_base, gpt-4o-mini). "
             "For hf: HF model id/path (e.g., meta-llama/Llama-3.1-8B-Instruct).",
    )
    ap.add_argument(
        "--single-file-output-dir",
        default="single_file_reports",
        help="Directory to save per-file similarity reports when --glob matches multiple files.",
    )

    args = ap.parse_args()

    token_count = build_token_counter(args.tokenizer, args.tokenizer_model)

    tool_lat = dict(DEFAULT_TOOL_LAT)

    # Aggregates
    rg_costs: Dict[Tuple[str, Tuple[str, ...], str], List[float]] = defaultdict(list)
    episode_totals: List[float] = []
    episode_by_file: Dict[str, float] = {}

    # Similarity metrics
    redundancy_by_file_rg: Dict[str, Dict[Tuple[str, Tuple[str, ...], str], Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    repetitions_by_file: Dict[str, Counter] = defaultdict(Counter)
    avoidable_cost_by_file_rg: Dict[str, Dict[Tuple[str, Tuple[str, ...], str], float]] = defaultdict(
        lambda: defaultdict(float)
    )

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        raise SystemExit(f"No files matched {args.glob}")

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)

        instance_id = d.get("instance_id", os.path.basename(fp))
        trajectory = d.get("trajectory", [])

        stages = parse_stages_from_trajectory(trajectory, file=fp, instance_id=instance_id)

        ep_total = 0.0
        stage_sigs: List[Tuple[str, str, Tuple[str, ...]]] = []

        for st in stages:
            extract_tools(st)
            st.tool_signature = tuple(sorted([t for t in st.tool_counts.keys() if t != "__default__"]))

            st.est_cost = estimate_cost(st, args.decode_tps, args.prefill_tps, tool_lat, token_count)
            ep_total += st.est_cost

            rg = stage_rg_key(st, args.slo_class, args.rg_version)
            rg_costs[rg].append(st.est_cost)

            # redundancy: repeated tool+args within RG + avoidable cost
            for k, v in st.tool_calls_with_args.items():
                if v > 1:
                    redundancy_by_file_rg[fp][rg][k] += (v - 1)

                    tool_name = k.split("(", 1)[0].strip()
                    lat = tool_lat.get(tool_name, tool_lat.get("__default__", 0.40))
                    avoidable_cost_by_file_rg[fp][rg] += (v - 1) * lat

            # repetition signature within episode
            stage_sigs.append(stage_signature_for_repetition(st))

        # stage repetition within episode (same sig multiple times)
        c = Counter(stage_sigs)
        for sig, cnt in c.items():
            if cnt > 1:
                repetitions_by_file[fp][sig] += (cnt - 1)

        episode_totals.append(ep_total)
        episode_by_file[fp] = ep_total

    if len(files) > 1:
        os.makedirs(args.single_file_output_dir, exist_ok=True)
        single_file_report_paths = build_single_file_report_paths(files, args.single_file_output_dir)

        # ----------------------------
        # Stage-level SLOs per resource group
        # ----------------------------
        stage_rows = []
        for rg, xs in rg_costs.items():
            sa, tool_sig, slo_class = rg
            p50 = percentile(xs, 50) or 0.0
            p95 = percentile(xs, 95) or 0.0
            p99 = percentile(xs, 99) or 0.0
            stage_rows.append([
                sa,
                "+".join(tool_sig) if tool_sig else "LLM_ONLY",
                slo_class,
                len(xs),
                f"{p50:.2f}",
                f"{p95:.2f}",
                f"{p99:.2f}",
            ])

        stage_rows.sort(key=lambda r: (-int(r[3]), float(r[5])), reverse=False)  # primary: frequency

        print("\n=== Stage-level SLOs by Resource Group (sub_agent + tool_signature) ===")
        print_table(
            stage_rows[:args.show_top],
            headers=["SubAgent", "ToolSignature", "SLOClass", "n", "p50", "p95", "p99"],
        )
        print("\nSuggested stage SLO for interactive groups: use p95; for batch groups: use p99.\n")

        # ----------------------------
        # Episode-level SLOs
        # ----------------------------
        ep_p50 = percentile(episode_totals, 50) or 0.0
        ep_p95 = percentile(episode_totals, 95) or 0.0
        ep_p99 = percentile(episode_totals, 99) or 0.0

        print("=== Episode-level SLOs (sum of stage EstimatedCost per file) ===")
        print(f"n={len(episode_totals)}  p50={ep_p50:.2f}  p95={ep_p95:.2f}  p99={ep_p99:.2f}")
        print("\nPer-episode totals:")
        for fp, tot in sorted(episode_by_file.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {os.path.basename(fp)}  total={tot:.2f}")

        for fp in files:
            with open(single_file_report_paths[fp], "w", encoding="utf-8") as out:
                print_file_similarity_section(
                    file=fp,
                    episode_total=episode_by_file[fp],
                    file_redundancy_by_rg=redundancy_by_file_rg[fp],
                    file_avoidable_cost_by_rg=avoidable_cost_by_file_rg[fp],
                    repeated_stage_counts=repetitions_by_file[fp],
                    show_top=args.show_top,
                    include_file_header=False,
                    out=out,
                )
    else:
        fp = files[0]
        print_file_similarity_section(
            file=fp,
            episode_total=episode_by_file[fp],
            file_redundancy_by_rg=redundancy_by_file_rg[fp],
            file_avoidable_cost_by_rg=avoidable_cost_by_file_rg[fp],
            repeated_stage_counts=repetitions_by_file[fp],
            show_top=args.show_top,
            include_file_header=False,
            out=sys.stdout,
        )

if __name__ == "__main__":
    main()
