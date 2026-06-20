"""Shared helpers for resource-group classification and reuse keys."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from hyperagent_replay import binding
from hyperagent_replay.replay import estimate_text_tokens
from hyperagent_replay.trace import parse_events

TOOL_CALL_WITH_ARGS_RE = re.compile(r"\b([A-Za-z_]\w*)\._run\s*\((.*)\)\s*$")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def stable_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def normalize_subgoal(subgoal: str | None) -> str:
    if not subgoal:
        return ""
    return normalize_space(subgoal)


def normalize_tool_name(tool_name: str | None) -> str:
    if not tool_name:
        return ""
    return normalize_space(tool_name).lower()


def normalize_action_code(code: str | None) -> str:
    if not code:
        return ""
    lines = [normalize_space(line) for line in code.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_action_signature(
    action: dict[str, Any] | None,
) -> tuple[str, str]:
    if not action:
        return "", ""

    tool_name = normalize_tool_name(action.get("tool_name"))
    language = normalize_tool_name(action.get("language"))
    code = action.get("code") or ""

    if language == "python":
        for raw_line in code.splitlines():
            normalized_line = normalize_space(raw_line)
            if not normalized_line or normalized_line.startswith("#"):
                continue
            match = TOOL_CALL_WITH_ARGS_RE.search(normalized_line)
            if match:
                name = normalize_tool_name(match.group(1))
                args = normalize_space(match.group(2))
                return f"{name}({args})", name or tool_name

    normalized_code = normalize_action_code(code)
    if normalized_code:
        return normalized_code, tool_name
    return tool_name, tool_name


def turn_tool_signature(turn: dict[str, Any]) -> str:
    action = turn.get("action") or {}
    action_signature, tool_name = extract_action_signature(action)
    if tool_name:
        return tool_name
    if action_signature:
        return "ACTION"
    return "LLM_ONLY"


def token_bucket(n: int) -> str:
    if n < 256:
        return "0-255"
    if n < 1024:
        return "256-1023"
    if n < 4096:
        return "1024-4095"
    return "4096+"


def build_exact_repeat_key(turn: dict[str, Any], subgoal: str = "") -> str:
    action_signature, tool_name = extract_action_signature(turn.get("action"))
    payload = {
        "kind": "action" if action_signature else "llm",
        "agent": normalize_space(turn.get("agent", "")),
        "subgoal": normalize_subgoal(subgoal),
        "contains_final_answer": bool(turn.get("contains_final_answer")),
        "tool_name": tool_name,
        "action_signature_hash": stable_text_hash(action_signature),
        "content_hash": stable_text_hash(normalize_space(turn.get("content", ""))),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def build_resource_group(
    turn: dict[str, Any],
    slo_class: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    estimated_prompt_tokens: int | None = None,
    response_text: str = "",
    chars_per_token_estimate: float = 3.0,
    subgoal: str = "",
    rg_version: str = "legacy",
    new_prefill_tokens: int | None = None,
    io_seconds: float = 0.0,
) -> dict[str, str]:
    prompt_count = prompt_tokens
    if prompt_count is None:
        prompt_count = estimated_prompt_tokens
    if prompt_count is None:
        prompt_count = estimate_text_tokens(turn.get("content", ""),
                                            chars_per_token_estimate)

    completion_count = completion_tokens
    if completion_count is None and response_text:
        completion_count = estimate_text_tokens(response_text,
                                                chars_per_token_estimate)
    if completion_count is None:
        completion_count = 0

    if rg_version == "v2":
        action = turn.get("action") or None
        tool = action.get("tool_name") if action else None
        prefill_count = (new_prefill_tokens
                         if new_prefill_tokens is not None else prompt_count)
        binding_type = binding.classify_binding(
            new_prefill_tokens=prefill_count,
            decode_tokens=completion_count,
            io_seconds=io_seconds,
            has_action=action is not None,
        )
        group = {
            "workload_type": binding.workload_type(turn.get("agent"), tool),
            "binding_type": binding_type,
            "dag_layer_bucket": binding.dag_layer(turn.get("agent")),
            "subgoal_prefix": binding.subgoal_prefix(subgoal),
            "slo_class": slo_class,
        }
        group["key"] = json.dumps(group, sort_keys=True, separators=(",", ":"))
        return group

    group = {
        "slo_class": slo_class,
        "agent": turn.get("agent", ""),
        "tool_signature": turn_tool_signature(turn),
        "subgoal": normalize_subgoal(subgoal),
        "prompt_bucket": token_bucket(prompt_count),
        "completion_bucket": token_bucket(completion_count),
    }
    group["key"] = json.dumps(group, sort_keys=True, separators=(",", ":"))
    return group


def extract_response_subgoals(raw_payload: dict[str, Any]) -> list[str]:
    subgoals: list[str] = []
    current_subgoal = ""
    for event in parse_events(raw_payload.get("trajectory", [])):
        if event["type"] == "subgoal":
            current_subgoal = normalize_subgoal(event.get("content", ""))
        elif event["type"] == "response":
            subgoals.append(current_subgoal)
    return subgoals
