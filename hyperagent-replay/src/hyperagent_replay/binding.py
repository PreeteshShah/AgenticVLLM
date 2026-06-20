"""Pure, dependency-free helpers for the v2 resource-group fields.

This module is intentionally free of third-party imports (and of any other
`hyperagent_replay` module that pulls in `openai`) so it can be imported both:

- inside the replay/scheduler path, fed *real* vLLM `usage` token counts, and
- by the offline split-preview tooling on a bare Python 3.8 interpreter, fed
  *estimated* token counts.

The single source of truth for how a turn maps to a `binding_type` /
`workload_type` / `dag_layer` lives here so the offline preview and the live
replay can never drift.
"""

from __future__ import annotations

import hashlib
from typing import Optional

# ---------------------------------------------------------------------------
# Cost model (proxy units; calibrate the *ratio* on RunPod later).
#
# A turn's wall cost decomposes into:
#   prefill_cost = new_prefill_tokens / prefill_tps   (compute-bound work)
#   decode_cost  = decode_tokens      / decode_tps    (memory-bandwidth work)
#   io_cost      = tool/io seconds                    (off-GPU wait)
#
# For a 70B model, decode is ~1 token at a time (memory bound, slow) while
# prefill processes the whole new prompt in parallel (compute bound, fast per
# token). So prefill_tps >> decode_tps. The defaults below encode that gap;
# the absolute numbers do not matter for classification, only the ratio does.
# ---------------------------------------------------------------------------
DEFAULT_DECODE_TPS = 40.0
DEFAULT_PREFILL_TPS = 2000.0

# Share of total cost a single resource must exceed to "bind" the turn.
BINDING_DOMINANCE_THRESHOLD = 0.5

PREFILL_BOUND = "PREFILL_BOUND"
DECODE_BOUND = "DECODE_BOUND"
IO_BOUND = "IO_BOUND"
MIXED = "MIXED"
LLM_ONLY = "LLM_ONLY"

# workload_type values
WL_PLANNING = "PLANNING"
WL_NAVIGATION = "NAVIGATION"
WL_SEARCH = "SEARCH"
WL_CODE_EXEC = "CODE_EXEC"
WL_EDITING = "EDITING"
WL_LLM_ONLY = "LLM_ONLY"

# Tool-name (lowercased, sans trailing ``_run``) -> workload_type.
_NAVIGATION_TOOLS = {
    "open_file",
    "open_file_gen",
    "get_folder_structure",
    "find_file",
    "find_all_refs",
    "go_to_def",
}
_SEARCH_TOOLS = {
    "code_search",
    "get_all_symbols",
}
_EDIT_TOOLS = {
    "editor",
}
_CODE_EXEC_TOOLS = {
    "executor",
    "run_pytest",
    "run_test",
    "python_exec",
    "git_log",
    "pytest",
    "python",
    "bash_exec",
}


def normalize_tool(tool_name: Optional[str]) -> str:
    if not tool_name:
        return ""
    name = tool_name.strip().lower()
    if name.endswith("._run"):
        name = name[: -len("._run")]
    if name.endswith("_run") and not name.endswith("._run"):
        # e.g. "open_file_run" defensive; real names are "open_file"
        pass
    return name


def classify_binding(
    new_prefill_tokens: float,
    decode_tokens: float,
    io_seconds: float,
    has_action: bool,
    decode_tps: float = DEFAULT_DECODE_TPS,
    prefill_tps: float = DEFAULT_PREFILL_TPS,
) -> str:
    """Return the binding_type for a single turn from its cost shares.

    `has_action` distinguishes a pure-reasoning turn (no tool call) so we can
    honor the LLM_ONLY binding value: a turn with no tool and no off-GPU wait
    is reported as LLM_ONLY rather than being forced into PREFILL/DECODE.
    """
    prefill_cost = max(0.0, new_prefill_tokens) / prefill_tps
    decode_cost = max(0.0, decode_tokens) / decode_tps
    io_cost = max(0.0, io_seconds)

    total = prefill_cost + decode_cost + io_cost
    if total <= 0.0:
        return LLM_ONLY

    shares = {
        PREFILL_BOUND: prefill_cost / total,
        DECODE_BOUND: decode_cost / total,
        IO_BOUND: io_cost / total,
    }
    leader, leader_share = max(shares.items(), key=lambda kv: kv[1])

    if leader_share > BINDING_DOMINANCE_THRESHOLD:
        # A turn with no tool action that is dominated by generation work is a
        # pure-LLM turn; keep the dedicated LLM_ONLY label for it.
        if leader == DECODE_BOUND and not has_action and io_cost == 0.0:
            return LLM_ONLY
        return leader
    return MIXED


def workload_type(agent: Optional[str], tool_name: Optional[str]) -> str:
    """Map (sub_agent, tool) -> what the turn is *doing*."""
    tool = normalize_tool(tool_name)
    if tool:
        if tool in _NAVIGATION_TOOLS:
            return WL_NAVIGATION
        if tool in _SEARCH_TOOLS:
            return WL_SEARCH
        if tool in _EDIT_TOOLS:
            return WL_EDITING
        if tool in _CODE_EXEC_TOOLS:
            return WL_CODE_EXEC
        # Unknown tool: fall through to agent-based guess below.

    agent_l = (agent or "").lower()
    if "planner" in agent_l:
        return WL_PLANNING
    if "editor" in agent_l:
        return WL_EDITING
    if "executor" in agent_l:
        return WL_CODE_EXEC
    if "navigator" in agent_l:
        return WL_NAVIGATION
    return WL_LLM_ONLY


def dag_layer(agent: Optional[str]) -> str:
    """Bucket the turn's depth in the HyperAgent planner->intern call tree."""
    agent_l = (agent or "").lower()
    if "planner" in agent_l:
        return "L0"
    if agent_l.startswith("inner-") or "assistant" in agent_l:
        return "L2"
    return "L1"


def subgoal_prefix(subgoal: Optional[str], length: int = 12) -> str:
    """Stable short id for the active subgoal (proxy for shared-context reuse)."""
    if not subgoal:
        return ""
    norm = " ".join(subgoal.split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:length]