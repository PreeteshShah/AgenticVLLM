"""Replay one HyperAgent trajectory against vLLM with exact-repeat reuse."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI

from hyperagent_replay.replay import (
    DEFAULT_BASE_URL,
    DEFAULT_CHARS_PER_TOKEN_ESTIMATE,
    DEFAULT_CONTEXT_SAFETY_MARGIN,
    DEFAULT_MIN_REFERENCE_CHARS,
    estimate_text_tokens,
    ENGINE_MODE_CHOICES,
    ENGINE_MODE_CONTINUUM,
    ENGINE_MODE_STOCK,
    build_continuum_extra_body,
    build_observation,
    build_scheduler_timestamps,
    compute_input_token_budget,
    compute_new_prefill_tokens,
    context_key,
    extract_cached_prompt_tokens,
    fit_request_to_budget,
    is_context_length_error,
    kill_process_tree,
    launch_server,
    make_seed_messages,
    percentile,
    reduce_context_for_retry,
    scheduler_timestamps_path_for,
    tool_delay_for_turn,
    wait_for_server,
)
from hyperagent_replay import binding
from hyperagent_replay.resource_groups import (
    build_exact_repeat_key,
    build_resource_group,
    extract_response_subgoals,
)
from hyperagent_replay.slo_report import build_slo_report, load_agent_targets
from hyperagent_replay.trace import load_trace_payload

DEFAULT_REQUEST_LATENCY_PRIOR_S = 10.0


def load_trace_with_subgoals(
    input_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(input_path.read_text())
    trace = load_trace_payload(input_path)
    subgoals: list[str] = []
    if isinstance(payload, dict) and "trajectory" in payload:
        subgoals = extract_response_subgoals(payload)

    turns = trace.get("llm_turns", [])
    if len(subgoals) < len(turns):
        subgoals = [*subgoals, *([""] * (len(turns) - len(subgoals)))]
    elif len(subgoals) > len(turns):
        subgoals = subgoals[:len(turns)]
    return trace, subgoals


def default_reuse_output_path(input_path: Path) -> Path:
    name = input_path.name
    for suffix in (".extracted.json", ".json"):
        if name.endswith(suffix):
            base = name[:-len(suffix)]
            return input_path.with_name(f"{base}.reuse.replay.json")
    return input_path.with_name(f"{name}.reuse.replay.json")


def compute_turn_binding_and_rg(
    turn: dict[str, Any],
    subgoal: str,
    estimated_prompt_tokens: int | None,
    slo_class: str,
    delay_policy: str,
    constant_delay: float,
    chars_per_token_estimate: float,
) -> tuple[str, str]:
    """Compute (binding_type, rg_key) for a turn *before* it is sent.

    Uses request-time-known quantities only: the estimated prompt size
    (prefill proxy), the recorded turn's output length (decode proxy), and the
    turn's tool delay (io). This is the v2 resource-group key the continuum
    scheduler groups on. Mirrors `binding.classify_binding` exactly.
    """
    action = turn.get("action") or None
    tool = action.get("tool_name") if action else None
    decode_tokens_est = estimate_text_tokens(
        turn.get("content", ""), chars_per_token_estimate)
    io_seconds = tool_delay_for_turn(turn, delay_policy, constant_delay)
    binding_type = binding.classify_binding(
        new_prefill_tokens=estimated_prompt_tokens or 0,
        decode_tokens=decode_tokens_est,
        io_seconds=io_seconds,
        has_action=action is not None,
    )
    workload = binding.workload_type(turn.get("agent"), tool)
    rg_key = "|".join([
        workload,
        binding_type,
        binding.dag_layer(turn.get("agent")),
        binding.subgoal_prefix(subgoal),
        slo_class,
    ])
    return binding_type, rg_key


def build_cache_hit_observation(turn: dict[str, Any],
                                source_turn_index: int) -> str:
    action = turn.get("action")
    if not action:
        return ""
    tool_name = action.get("tool_name") or "tool"
    language = action.get("language") or "text"
    lines = [
        "Observation: Cache hit reused a prior replay result "
        f"from turn {source_turn_index} for `{tool_name}`; "
        "synthetic tool execution was skipped.",
    ]
    if action.get("code"):
        lines.extend([
            "Recorded action code:",
            f"```{language}\n{action['code']}\n```",
        ])
    return "\n".join(lines)


def summarize_resource_groups(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    group_stats: dict[str, dict[str, Any]] = {}
    for item in results:
        group_key = item["resource_group_key"]
        if group_key not in group_stats:
            group_stats[group_key] = {
                "resource_group": item["resource_group"],
                "num_turns": 0,
                "num_cache_hits": 0,
                "num_executed_on_vllm": 0,
                "total_request_latency_s": 0.0,
                "total_synthetic_tool_sleep_s": 0.0,
            }
        stats = group_stats[group_key]
        stats["num_turns"] += 1
        stats["num_cache_hits"] += int(item["cache_hit"])
        stats["num_executed_on_vllm"] += int(item["executed_on_vllm"])
        stats["total_request_latency_s"] += item["request_latency_s"]
        stats["total_synthetic_tool_sleep_s"] += item["synthetic_tool_sleep_s"]

    groups = sorted(
        group_stats.values(),
        key=lambda entry: (
            -entry["num_turns"],
            -entry["num_cache_hits"],
            entry["resource_group"]["agent"],
            entry["resource_group"]["tool_signature"],
        ),
    )
    return {
        "num_groups": len(groups),
        "groups": groups,
    }


def summarize_repeated_work(
    results: list[dict[str, Any]],
    show_top: int,
) -> dict[str, Any]:
    by_key: dict[str, dict[str, Any]] = {}
    resource_group_counts: Counter[str] = Counter()
    resource_group_cache_hits: Counter[str] = Counter()

    for item in results:
        key = item["cache_key"]
        resource_group_counts[item["resource_group_key"]] += 1
        resource_group_cache_hits[item["resource_group_key"]] += int(
            item["cache_hit"])
        if key not in by_key:
            by_key[key] = {
                "cache_key": key,
                "agent": item["agent"],
                "resource_group_key": item["resource_group_key"],
                "first_turn_index": item["turn_index"],
                "turn_indices": [],
                "num_occurrences": 0,
                "num_cache_hits": 0,
            }
        entry = by_key[key]
        entry["turn_indices"].append(item["turn_index"])
        entry["num_occurrences"] += 1
        entry["num_cache_hits"] += int(item["cache_hit"])

    exact_repeats = [
        entry for entry in by_key.values() if entry["num_occurrences"] > 1
    ]
    exact_repeats.sort(
        key=lambda entry: (
            -entry["num_cache_hits"],
            -entry["num_occurrences"],
            entry["first_turn_index"],
        ),
    )

    repeated_resource_groups = sorted(
        (
            {
                "resource_group_key": key,
                "num_turns": count,
                "repeat_events": count - 1,
                "num_cache_hits": resource_group_cache_hits[key],
            }
            for key, count in resource_group_counts.items()
            if count > 1
        ),
        key=lambda entry: (
            -entry["repeat_events"],
            -entry["num_cache_hits"],
            entry["resource_group_key"],
        ),
    )

    return {
        "num_unique_exact_repeat_keys": len(exact_repeats),
        "exact_repeats": exact_repeats[:show_top],
        "repeated_resource_groups": repeated_resource_groups[:show_top],
    }


def precompute_cache_keys(
    turns: list[dict[str, Any]],
    subgoals: list[str],
) -> tuple[list[str], list[bool]]:
    keys: list[str] = []
    cache_hits: list[bool] = []
    seen: set[str] = set()
    for turn, subgoal in zip(turns, subgoals, strict=False):
        key = build_exact_repeat_key(turn, subgoal=subgoal)
        keys.append(key)
        cache_hits.append(key in seen)
        seen.add(key)
    return keys, cache_hits


def predicted_request_latency_for_agent(
    agent: str,
    latency_history_by_agent: dict[str, list[float]],
    global_latency_history: list[float],
    default_request_latency_prior_s: float,
) -> float:
    agent_history = latency_history_by_agent.get(agent, [])
    if agent_history:
        return percentile(agent_history, 50)
    if global_latency_history:
        return percentile(global_latency_history, 50)
    return default_request_latency_prior_s


def estimate_remaining_wall_time_s(
    *,
    turns: list[dict[str, Any]],
    start_index: int,
    current_cache_hit: bool,
    predicted_cache_hits: list[bool],
    latency_history_by_agent: dict[str, list[float]],
    global_latency_history: list[float],
    delay_policy: str,
    constant_delay: float,
    default_request_latency_prior_s: float,
) -> float:
    total_s = 0.0
    for index in range(start_index, len(turns)):
        turn = turns[index]
        cache_hit = current_cache_hit if index == start_index else predicted_cache_hits[index]
        if cache_hit:
            continue
        total_s += predicted_request_latency_for_agent(
            str(turn.get("agent", "")),
            latency_history_by_agent,
            global_latency_history,
            default_request_latency_prior_s,
        )
        total_s += tool_delay_for_turn(turn, delay_policy, constant_delay)
    return total_s


def select_slo_budget_mode(
    *,
    request_target_s: float | None,
    predicted_request_latency_s: float,
    episode_target_s: float | None,
    slack_before_turn_s: float | None,
) -> tuple[str, list[str]]:
    if request_target_s is None and episode_target_s is None:
        return "off", []

    reasons: list[str] = []
    severe_request_pressure = False
    mild_request_pressure = False
    if request_target_s is not None:
        severe_request_pressure = predicted_request_latency_s > request_target_s
        mild_request_pressure = predicted_request_latency_s > (0.85 * request_target_s)
        if severe_request_pressure:
            reasons.append("request_target_breach_risk")
        elif mild_request_pressure:
            reasons.append("request_target_pressure")

    severe_episode_pressure = False
    mild_episode_pressure = False
    if episode_target_s is not None and slack_before_turn_s is not None:
        severe_episode_pressure = slack_before_turn_s < 0.0
        mild_episode_pressure = slack_before_turn_s < max(
            15.0,
            episode_target_s * 0.05,
        )
        if severe_episode_pressure:
            reasons.append("episode_deadline_risk")
        elif mild_episode_pressure:
            reasons.append("episode_deadline_pressure")

    if severe_request_pressure or severe_episode_pressure:
        return "tight", reasons
    if mild_request_pressure or mild_episode_pressure:
        return "guarded", reasons
    return "normal", reasons


def effective_budget_settings(
    *,
    base_max_reference_chars: int,
    base_max_completion_tokens: int,
    base_context_safety_margin: int,
    min_reference_chars: int,
    slo_class: str,
    budget_mode: str,
    apply_budget_policy: bool,
) -> dict[str, int | str | bool]:
    if not apply_budget_policy or budget_mode in {"off", "normal"}:
        return {
            "budget_mode": budget_mode,
            "applied": apply_budget_policy and budget_mode != "off",
            "max_reference_chars": base_max_reference_chars,
            "max_completion_tokens": base_max_completion_tokens,
            "context_safety_margin": base_context_safety_margin,
        }

    if budget_mode == "guarded":
        return {
            "budget_mode": budget_mode,
            "applied": True,
            "max_reference_chars": max(
                min_reference_chars,
                int(base_max_reference_chars * 0.75),
            ),
            "max_completion_tokens": max(64, int(base_max_completion_tokens * 0.75)),
            "context_safety_margin": base_context_safety_margin + 256,
        }

    tight_completion_scale = 0.50 if slo_class == "interactive" else 0.65
    tight_reference_scale = 0.50 if slo_class == "interactive" else 0.65
    tight_margin_bump = 512 if slo_class == "interactive" else 384
    return {
        "budget_mode": budget_mode,
        "applied": True,
        "max_reference_chars": max(
            min_reference_chars,
            int(base_max_reference_chars * tight_reference_scale),
        ),
        "max_completion_tokens": max(64, int(base_max_completion_tokens * tight_completion_scale)),
        "context_safety_margin": base_context_safety_margin + tight_margin_bump,
    }


def replay_trace_with_reuse(
    trace: dict[str, Any],
    subgoals: list[str],
    client: OpenAI,
    model_name: str,
    context_mode: str,
    slo_class: str,
    max_reference_chars: int,
    max_turns: int | None,
    temperature: float,
    max_completion_tokens: int,
    seed: int | None,
    delay_policy: str,
    constant_delay: float,
    max_model_len: int | None,
    context_safety_margin: int,
    min_reference_chars: int,
    chars_per_token_estimate: float,
    show_top: int,
    slo_policy: str = "off",
    request_target_s: float | None = None,
    agent_request_targets: dict[str, float] | None = None,
    episode_target_s: float | None = None,
    default_request_latency_prior_s: float = DEFAULT_REQUEST_LATENCY_PRIOR_S,
    engine_mode: str = ENGINE_MODE_STOCK,
    job_id_override: str | None = None,
    progress_callback: Callable[[int, int, dict[str, Any], dict[str, Any]], None]
    | None = None,
) -> dict[str, Any]:
    if engine_mode not in ENGINE_MODE_CHOICES:
        raise ValueError(
            f"engine_mode must be one of {ENGINE_MODE_CHOICES}, "
            f"got {engine_mode!r}")
    continuum_job_id = job_id_override or trace["instance_id"]
    contexts: dict[str, list[dict[str, str]]] = {}
    turns = trace["llm_turns"]
    if max_turns is not None:
        turns = turns[:max_turns]
        subgoals = subgoals[:max_turns]

    results: list[dict[str, Any]] = []
    total_tool_sleep_s = 0.0
    total_cache_lookup_latency_s = 0.0
    cache: dict[str, dict[str, Any]] = {}
    solve_t0 = time.time()
    total_turns = len(turns)
    agent_request_targets = agent_request_targets or {}
    precomputed_cache_keys, predicted_cache_hits = precompute_cache_keys(
        turns,
        subgoals,
    )
    # For --engine-mode continuum, flag is_last_step on the final turn that
    # actually reaches the server (cache hits are skipped locally and must
    # not look like an in-flight pinned job to Continuum).
    last_vllm_turn_index: int | None = None
    if engine_mode == ENGINE_MODE_CONTINUUM:
        for idx in range(len(turns) - 1, -1, -1):
            if not predicted_cache_hits[idx]:
                last_vllm_turn_index = idx
                break
    latency_history_by_agent: dict[str, list[float]] = {}
    global_latency_history: list[float] = []

    for turn_index_in_trace, (turn, subgoal) in enumerate(
            zip(turns, subgoals, strict=False)):
        completed_turns = turn_index_in_trace + 1
        key = context_key(turn, context_mode)
        if key not in contexts:
            agent = turn["agent"] if context_mode != "flattened" else None
            contexts[key] = make_seed_messages(trace["problem_statement"], agent)

        cache_key = precomputed_cache_keys[turn_index_in_trace]
        lookup_t0 = time.time()
        cache_entry = cache.get(cache_key)
        cache_lookup_latency_s = time.time() - lookup_t0
        total_cache_lookup_latency_s += cache_lookup_latency_s
        current_cache_hit = cache_entry is not None
        current_request_target_s = agent_request_targets.get(
            str(turn.get("agent", "")),
            request_target_s,
        )
        predicted_request_latency_s = (
            0.0 if current_cache_hit else predicted_request_latency_for_agent(
                str(turn.get("agent", "")),
                latency_history_by_agent,
                global_latency_history,
                default_request_latency_prior_s,
            ))
        predicted_remaining_wall_time_s = estimate_remaining_wall_time_s(
            turns=turns,
            start_index=turn_index_in_trace,
            current_cache_hit=current_cache_hit,
            predicted_cache_hits=predicted_cache_hits,
            latency_history_by_agent=latency_history_by_agent,
            global_latency_history=global_latency_history,
            delay_policy=delay_policy,
            constant_delay=constant_delay,
            default_request_latency_prior_s=default_request_latency_prior_s,
        )
        elapsed_before_turn_s = time.time() - solve_t0
        projected_episode_time_s = elapsed_before_turn_s + predicted_remaining_wall_time_s
        slack_before_turn_s = (
            episode_target_s - projected_episode_time_s
            if episode_target_s is not None
            else None
        )
        selected_budget_mode, slo_policy_reasons = select_slo_budget_mode(
            request_target_s=current_request_target_s,
            predicted_request_latency_s=predicted_request_latency_s,
            episode_target_s=episode_target_s,
            slack_before_turn_s=slack_before_turn_s,
        )
        budget_settings = effective_budget_settings(
            base_max_reference_chars=max_reference_chars,
            base_max_completion_tokens=max_completion_tokens,
            base_context_safety_margin=context_safety_margin,
            min_reference_chars=min_reference_chars,
            slo_class=slo_class,
            budget_mode=selected_budget_mode,
            apply_budget_policy=slo_policy == "budget",
        )
        effective_max_reference_chars = int(budget_settings["max_reference_chars"])
        effective_max_completion_tokens = int(
            budget_settings["max_completion_tokens"])
        effective_context_safety_margin = int(
            budget_settings["context_safety_margin"])
        effective_input_token_budget = compute_input_token_budget(
            max_model_len,
            effective_max_completion_tokens,
            effective_context_safety_margin,
        )
        active_context = list(contexts[key])
        active_max_reference_chars = effective_max_reference_chars
        dropped_context_messages = 0
        context_retry_count = 0
        estimated_prompt_tokens: int | None = None

        (request_messages, active_context, active_max_reference_chars,
         dropped_for_budgeting,
         estimated_prompt_tokens) = fit_request_to_budget(
            context_messages=active_context,
            turn=turn,
            include_agent_name=context_mode == "flattened",
            max_reference_chars=active_max_reference_chars,
            input_token_budget=effective_input_token_budget,
            chars_per_token_estimate=chars_per_token_estimate,
            min_reference_chars=min_reference_chars,
        )
        dropped_context_messages += dropped_for_budgeting
        request_prompt_chars = sum(len(msg["content"]) for msg in request_messages)

        request_arrival_time: float | None = None
        request_departure_time: float | None = None
        request_latency_s = 0.0
        tool_sleep_s = 0.0
        tool_sleep_start_time: float | None = None
        tool_sleep_end_time: float | None = None
        response_text = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None
        cached_prompt_tokens: int | None = None
        new_prefill_tokens: int | None = None
        continuum_extra_body: dict[str, Any] | None = None
        cache_source_turn_index: int | None = None
        cache_source_prompt_tokens: int | None = None
        cache_source_completion_tokens: int | None = None

        if cache_entry is None:
            if engine_mode == ENGINE_MODE_CONTINUUM:
                turn_binding_type, turn_rg_key = compute_turn_binding_and_rg(
                    turn=turn,
                    subgoal=subgoal,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    slo_class=slo_class,
                    delay_policy=delay_policy,
                    constant_delay=constant_delay,
                    chars_per_token_estimate=chars_per_token_estimate,
                )
                continuum_extra_body = build_continuum_extra_body(
                    instance_id=continuum_job_id,
                    turn=turn,
                    is_last_step=(turn_index_in_trace == last_vllm_turn_index),
                    binding_type=turn_binding_type,
                    rg_key=turn_rg_key,
                )
            request_arrival_time = time.time()
            while True:
                create_kwargs: dict[str, Any] = {
                    "model": model_name,
                    "messages": request_messages,
                    "temperature": temperature,
                    "max_completion_tokens": effective_max_completion_tokens,
                    "seed": seed,
                }
                if continuum_extra_body is not None:
                    create_kwargs["extra_body"] = continuum_extra_body
                t0 = time.time()
                try:
                    response = client.chat.completions.create(**create_kwargs)
                    t1 = time.time()
                    request_departure_time = t1
                    request_latency_s = t1 - t0
                    break
                except Exception as exc:
                    if not is_context_length_error(exc):
                        raise
                    next_context, next_reference_chars, dropped_for_retry = reduce_context_for_retry(
                        active_context,
                        active_max_reference_chars,
                        min_reference_chars,
                        turn,
                    )
                    if (next_context == active_context and
                            next_reference_chars == active_max_reference_chars):
                        raise RuntimeError(
                            "Replay request exceeded the model context window and "
                            "context budgeting could not reduce it further."
                        ) from exc
                    active_context = next_context
                    active_max_reference_chars = next_reference_chars
                    dropped_context_messages += dropped_for_retry
                    context_retry_count += 1
                    (request_messages, active_context, active_max_reference_chars,
                     dropped_for_budgeting,
                     estimated_prompt_tokens) = fit_request_to_budget(
                        context_messages=active_context,
                        turn=turn,
                        include_agent_name=context_mode == "flattened",
                        max_reference_chars=active_max_reference_chars,
                        input_token_budget=effective_input_token_budget,
                        chars_per_token_estimate=chars_per_token_estimate,
                        min_reference_chars=min_reference_chars,
                    )
                    dropped_context_messages += dropped_for_budgeting
                    request_prompt_chars = sum(
                        len(msg["content"]) for msg in request_messages)

            response_text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            cached_prompt_tokens = extract_cached_prompt_tokens(usage)
            new_prefill_tokens = compute_new_prefill_tokens(
                prompt_tokens, cached_prompt_tokens)

            tool_sleep_s = tool_delay_for_turn(turn, delay_policy, constant_delay)
            if tool_sleep_s > 0.0:
                tool_sleep_start_time = time.time()
                time.sleep(tool_sleep_s)
                tool_sleep_end_time = time.time()
                total_tool_sleep_s += tool_sleep_s

            cache[cache_key] = {
                "turn_index": turn["turn_index"],
                "response_text": response_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
            latency_history_by_agent.setdefault(str(turn.get("agent", "")), []).append(
                request_latency_s)
            global_latency_history.append(request_latency_s)
            cache_hit = False
            executed_on_vllm = True
            observation = build_observation(turn, tool_sleep_s)
        else:
            response_text = cache_entry["response_text"]
            cache_source_turn_index = cache_entry["turn_index"]
            cache_source_prompt_tokens = cache_entry["prompt_tokens"]
            cache_source_completion_tokens = cache_entry["completion_tokens"]
            cache_hit = True
            executed_on_vllm = False
            observation = build_cache_hit_observation(
                turn,
                source_turn_index=cache_source_turn_index,
            )

        contexts[key] = [*request_messages, {
            "role": "assistant",
            "content": response_text,
        }]
        if observation:
            contexts[key].append({"role": "user", "content": observation})

        resource_group = build_resource_group(
            turn=turn,
            slo_class=slo_class,
            prompt_tokens=prompt_tokens if prompt_tokens is not None else cache_source_prompt_tokens,
            completion_tokens=(completion_tokens if completion_tokens is not None
                               else cache_source_completion_tokens),
            estimated_prompt_tokens=estimated_prompt_tokens,
            response_text=response_text,
            chars_per_token_estimate=chars_per_token_estimate,
            subgoal=subgoal,
        )

        turn_result = {
            "turn_index": turn["turn_index"],
            "agent": turn["agent"],
            "context_key": key,
            "contains_final_answer": turn["contains_final_answer"],
            "recorded_action": turn.get("action"),
            "request_arrival_time": request_arrival_time,
            "request_departure_time": request_departure_time,
            "request_latency_s": request_latency_s,
            "synthetic_tool_sleep_s": tool_sleep_s,
            "tool_sleep_start_time": tool_sleep_start_time,
            "tool_sleep_end_time": tool_sleep_end_time,
            "request_prompt_chars": request_prompt_chars,
            "response_chars": len(response_text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "new_prefill_tokens": new_prefill_tokens,
            "engine_mode": engine_mode,
            "continuum_extra_body": continuum_extra_body,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "policy_max_reference_chars": effective_max_reference_chars,
            "effective_max_reference_chars": active_max_reference_chars,
            "effective_max_completion_tokens": effective_max_completion_tokens,
            "effective_context_safety_margin": effective_context_safety_margin,
            "effective_input_token_budget": effective_input_token_budget,
            "dropped_context_messages": dropped_context_messages,
            "context_retry_count": context_retry_count,
            "request_messages": request_messages,
            "response_text": response_text,
            "cache_hit": cache_hit,
            "executed_on_vllm": executed_on_vllm,
            "cache_key": cache_key,
            "cache_source_turn_index": cache_source_turn_index,
            "cache_lookup_latency_s": cache_lookup_latency_s,
            "cache_source_prompt_tokens": cache_source_prompt_tokens,
            "cache_source_completion_tokens": cache_source_completion_tokens,
            "predicted_cache_hit": predicted_cache_hits[turn_index_in_trace],
            "request_target_s": current_request_target_s,
            "predicted_request_latency_s": predicted_request_latency_s,
            "projected_remaining_wall_time_s": predicted_remaining_wall_time_s,
            "projected_episode_time_s": projected_episode_time_s,
            "slack_before_turn_s": slack_before_turn_s,
            "slo_policy": slo_policy,
            "slo_budget_mode": str(budget_settings["budget_mode"]),
            "slo_budget_applied": bool(budget_settings["applied"]),
            "slo_policy_reasons": slo_policy_reasons,
            "resource_group": resource_group,
            "resource_group_key": resource_group["key"],
        }
        results.append(turn_result)

        if progress_callback is not None:
            progress_callback(completed_turns, total_turns, turn, turn_result)

    solve_t1 = time.time()
    request_latencies = [item["request_latency_s"] for item in results]
    prompt_token_values = [
        item["prompt_tokens"] for item in results
        if item["prompt_tokens"] is not None
    ]
    completion_token_values = [
        item["completion_tokens"] for item in results
        if item["completion_tokens"] is not None
    ]
    cached_prompt_token_values = [
        item["cached_prompt_tokens"] for item in results
        if item.get("cached_prompt_tokens") is not None
    ]
    new_prefill_token_values = [
        item["new_prefill_tokens"] for item in results
        if item.get("new_prefill_tokens") is not None
    ]
    total_prompt_tokens_reported = sum(prompt_token_values)
    total_cached_prompt_tokens = (
        sum(cached_prompt_token_values) if cached_prompt_token_values else 0)
    total_new_prefill_tokens = (
        sum(new_prefill_token_values) if new_prefill_token_values else 0)
    prefill_reuse_ratio = (
        total_cached_prompt_tokens / total_prompt_tokens_reported
        if total_prompt_tokens_reported > 0 else 0.0)
    scheduler_timestamps = build_scheduler_timestamps(
        trace["instance_id"],
        results,
    )

    num_cache_hits = sum(item["cache_hit"] for item in results)
    num_executed_on_vllm = sum(item["executed_on_vllm"] for item in results)
    num_turns_slo_budgeted = sum(item["slo_budget_applied"] for item in results)
    slo_budget_mode_counts = Counter(
        str(item["slo_budget_mode"]) for item in results)

    replay_result = {
        "instance_id": trace["instance_id"],
        "problem_statement": trace["problem_statement"],
        "source_summary": trace["summary"],
        "settings": {
            "model_name": model_name,
            "context_mode": context_mode,
            "slo_class": slo_class,
            "reuse_mode": "exact_repeat_full_stage_memoization",
            "max_reference_chars": max_reference_chars,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "max_model_len": max_model_len,
            "context_safety_margin": context_safety_margin,
            "min_reference_chars": min_reference_chars,
            "chars_per_token_estimate": chars_per_token_estimate,
            "delay_policy": delay_policy,
            "constant_delay": constant_delay,
            "seed": seed,
            "max_turns": max_turns,
            "slo_policy": slo_policy,
            "request_target_s": request_target_s,
            "agent_request_targets": agent_request_targets,
            "episode_target_s": episode_target_s,
            "default_request_latency_prior_s": default_request_latency_prior_s,
            "engine_mode": engine_mode,
            "continuum_job_id": (
                continuum_job_id
                if engine_mode == ENGINE_MODE_CONTINUUM else None),
        },
        "reuse_policy": {
            "mode": "exact_repeat_full_stage_memoization",
            "exact_repeat_only": True,
            "slo_class": slo_class,
        },
        "slo_control": {
            "policy": slo_policy,
            "request_metric": "request_latency_s",
            "default_request_target_s": request_target_s,
            "agent_request_targets": agent_request_targets,
            "episode_metric": "wall_solve_time_s",
            "episode_target_s": episode_target_s,
            "default_request_latency_prior_s": default_request_latency_prior_s,
            "num_turns_slo_budgeted": num_turns_slo_budgeted,
            "budget_mode_counts": dict(slo_budget_mode_counts),
        },
        "timing": {
            "wall_solve_time_s": solve_t1 - solve_t0,
            "optimized_wall_solve_time_s": solve_t1 - solve_t0,
            "lm_only_solve_time_s": sum(request_latencies),
            "synthetic_tool_sleep_s": total_tool_sleep_s,
            "num_replayed_turns": len(results),
            "avg_request_latency_s": (sum(request_latencies) /
                                       len(request_latencies)
                                       if request_latencies else 0.0),
            "median_request_latency_s":
            (percentile(request_latencies, 50) if request_latencies else 0.0),
            "p95_request_latency_s":
            (percentile(request_latencies, 95) if request_latencies else 0.0),
            "p99_request_latency_s":
            (percentile(request_latencies, 99) if request_latencies else 0.0),
            "gpu_seconds_per_job": sum(request_latencies),
            "total_prompt_tokens": total_prompt_tokens_reported,
            "total_completion_tokens": sum(completion_token_values),
            "total_cached_prompt_tokens": total_cached_prompt_tokens,
            "total_new_prefill_tokens": total_new_prefill_tokens,
            "prefill_reuse_ratio": prefill_reuse_ratio,
            "num_turns_with_cached_prompt_tokens":
            len(cached_prompt_token_values),
            "num_turns_context_trimmed": sum(
                1 for item in results
                if item["dropped_context_messages"] > 0
                or item["effective_max_reference_chars"] != item["policy_max_reference_chars"]
            ),
            "num_turns_slo_budgeted": num_turns_slo_budgeted,
            "num_context_retry_events": sum(
                item["context_retry_count"] for item in results),
            "total_dropped_context_messages": sum(
                item["dropped_context_messages"] for item in results),
            "num_vllm_requests_executed": num_executed_on_vllm,
            "num_vllm_requests_avoided": num_cache_hits,
            "num_cache_hits": num_cache_hits,
            "total_cache_lookup_latency_s": total_cache_lookup_latency_s,
        },
        "resource_groups": summarize_resource_groups(results),
        "repeated_work": summarize_repeated_work(results, show_top=show_top),
        "scheduler_timestamps": scheduler_timestamps,
        "turn_metrics": results,
    }
    if (request_target_s is not None or agent_request_targets
            or episode_target_s is not None):
        replay_result["slo_summary"] = build_slo_report(
            replay_result,
            request_metric="request_latency_s",
            default_request_target_s=request_target_s,
            agent_request_targets=agent_request_targets,
            episode_metric="wall_solve_time_s",
            episode_target_s=episode_target_s,
        )
    return replay_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one HyperAgent trajectory against vLLM with exact-repeat reuse"
    )
    parser.add_argument("input",
                        type=Path,
                        help="Raw HyperAgent JSON or extracted trace JSON")
    parser.add_argument("--output",
                        type=Path,
                        default=None,
                        help="Path to save reuse-aware replay results JSON")
    parser.add_argument("--model",
                        required=True,
                        help="Model id to request from the vLLM server")
    parser.add_argument("--base-url",
                        default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible vLLM base URL")
    parser.add_argument("--api-key",
                        default="EMPTY",
                        help="API key for the OpenAI client")
    parser.add_argument("--context-mode",
                        choices=["flattened", "per_agent"],
                        default="per_agent")
    parser.add_argument("--slo-class",
                        default="interactive",
                        choices=["interactive", "batch"])
    parser.add_argument("--slo-policy",
                        choices=["off", "monitor", "budget"],
                        default=None,
                        help="How request and episode SLOs affect reuse replay budgeting")
    parser.add_argument("--request-target-s",
                        type=float,
                        default=None,
                        help="Default per-request latency target in seconds")
    parser.add_argument("--agent-request-targets",
                        type=Path,
                        default=None,
                        help="JSON file mapping agent name to per-request latency target in seconds")
    parser.add_argument("--episode-target-s",
                        type=float,
                        default=None,
                        help="Trajectory-level wall-clock target in seconds")
    parser.add_argument("--max-reference-chars",
                        type=int,
                        default=4000)
    parser.add_argument("--max-turns",
                        type=int,
                        default=None,
                        help="Replay only the first N turns")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens",
                        type=int,
                        default=512)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--context-safety-margin",
                        type=int,
                        default=DEFAULT_CONTEXT_SAFETY_MARGIN)
    parser.add_argument("--min-reference-chars",
                        type=int,
                        default=DEFAULT_MIN_REFERENCE_CHARS)
    parser.add_argument("--chars-per-token-estimate",
                        type=float,
                        default=DEFAULT_CHARS_PER_TOKEN_ESTIMATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay-policy",
                        choices=["none", "constant", "heuristic"],
                        default="heuristic")
    parser.add_argument("--constant-delay",
                        type=float,
                        default=0.2)
    parser.add_argument("--show-top",
                        type=int,
                        default=20,
                        help="How many repeated-work entries to keep in summaries")
    parser.add_argument("--launch-server",
                        action="store_true",
                        help="Launch `vllm serve` before replaying")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-timeout-s",
                        type=float,
                        default=180.0)
    parser.add_argument("--server-log",
                        type=Path,
                        default=None)
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="Additional argument to pass through to `vllm serve`",
    )
    parser.add_argument(
        "--engine-mode",
        choices=list(ENGINE_MODE_CHOICES),
        default=ENGINE_MODE_STOCK,
        help=(
            "Serving backend to target. `continuum` adds vllm-continuum "
            "scheduling metadata (job_id, this_func_call, is_last_step) "
            "only on the turns that actually reach the server."
        ),
    )
    parser.add_argument(
        "--job-id-override",
        type=str,
        default=None,
        help=(
            "Override the Continuum job_id (defaults to the trace "
            "instance_id). Only meaningful with --engine-mode continuum."
        ),
    )
    args = parser.parse_args()

    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be > 0")
    if args.context_safety_margin < 0:
        raise ValueError("--context-safety-margin must be >= 0")
    if args.min_reference_chars < 0:
        raise ValueError("--min-reference-chars must be >= 0")
    if args.chars_per_token_estimate <= 0:
        raise ValueError("--chars-per-token-estimate must be > 0")
    if args.show_top < 0:
        raise ValueError("--show-top must be >= 0")
    if args.request_target_s is not None and args.request_target_s <= 0:
        raise ValueError("--request-target-s must be > 0")
    if args.episode_target_s is not None and args.episode_target_s <= 0:
        raise ValueError("--episode-target-s must be > 0")

    agent_request_targets = load_agent_targets(args.agent_request_targets)
    if args.slo_policy is None:
        args.slo_policy = (
            "budget"
            if (args.request_target_s is not None or agent_request_targets
                or args.episode_target_s is not None)
            else "off"
        )

    if (args.base_url == DEFAULT_BASE_URL and
            (args.host != "127.0.0.1" or args.port != 8000)):
        args.base_url = f"http://{args.host}:{args.port}/v1"

    trace, subgoals = load_trace_with_subgoals(args.input)
    output_path = args.output or default_reuse_output_path(args.input)
    scheduler_timestamps_path = scheduler_timestamps_path_for(output_path)

    server_proc = None
    if args.launch_server:
        server_proc = launch_server(
            model_name=args.model,
            port=args.port,
            host=args.host,
            serve_args=args.serve_arg,
            log_path=args.server_log,
        )

    try:
        wait_for_server(
            args.base_url,
            args.server_timeout_s,
            server_proc=server_proc,
            server_log=args.server_log,
        )
        client = OpenAI(base_url=args.base_url,
                        api_key=args.api_key,
                        timeout=max(args.server_timeout_s, 30.0))
        results = replay_trace_with_reuse(
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
            max_model_len=args.max_model_len,
            context_safety_margin=args.context_safety_margin,
            min_reference_chars=args.min_reference_chars,
            chars_per_token_estimate=args.chars_per_token_estimate,
            seed=args.seed,
            delay_policy=args.delay_policy,
            constant_delay=args.constant_delay,
            show_top=args.show_top,
            slo_policy=args.slo_policy,
            request_target_s=args.request_target_s,
            agent_request_targets=agent_request_targets,
            episode_target_s=args.episode_target_s,
            engine_mode=args.engine_mode,
            job_id_override=args.job_id_override,
            progress_callback=lambda completed_turns, total_turns, turn,
            turn_result: print(
                "[progress] "
                f"Completed {completed_turns}/{total_turns} turns "
                f"({(completed_turns / total_turns * 100.0) if total_turns else 100.0:.1f}%); "
                f"turn_index={turn['turn_index']}; "
                f"cache_hit={'yes' if turn_result['cache_hit'] else 'no'}; "
                f"executed_on_vllm={'yes' if turn_result['executed_on_vllm'] else 'no'}; "
                f"slo_mode={turn_result['slo_budget_mode']}",
                flush=True,
            ),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        scheduler_timestamps_path.write_text(
            json.dumps(results["scheduler_timestamps"], indent=2))
        print(f"Saved reuse-aware replay results to {output_path}")
        print(f"Saved scheduler timestamps to {scheduler_timestamps_path}")
        print(json.dumps(results["timing"], indent=2))
    finally:
        if server_proc is not None:
            kill_process_tree(server_proc)


if __name__ == "__main__":
    main()
