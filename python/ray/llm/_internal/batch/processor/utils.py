"""Shared utility functions for processor builders."""

import logging
from typing import AbstractSet, Any, Dict, List, Optional, Tuple, Union

from ray.data import ActorPoolStrategy
from ray.llm._internal.batch.stages.configs import _StageConfigBase

logger = logging.getLogger(__name__)

# Canonical (user-facing) names for the vLLM processor's CPU actor-pool stages and
# the GPU engine boundary. These are the names accepted in ``stage_groups`` and used
# by ``plan_stage_fusion``. The order mirrors the pipeline order built by
# ``build_vllm_engine_processor`` (``prepare_image`` and ``prepare_multimodal`` are
# mutually exclusive, so at most one appears in any enabled pipeline).
ENGINE_STAGE_NAME = "vllm_engine"
PRE_ENGINE_CPU_STAGE_NAMES = (
    "prepare_image",
    "prepare_multimodal",
    "chat_template",
    "tokenize",
)
POST_ENGINE_CPU_STAGE_NAMES = ("detokenize",)
FUSABLE_CPU_STAGE_NAMES = frozenset(
    PRE_ENGINE_CPU_STAGE_NAMES + POST_ENGINE_CPU_STAGE_NAMES
)

# CPUs to reserve when estimating how many are free for the CPU stages: the Ray driver
# plus the user preprocess/postprocess ``map`` tasks. The GPU engine itself holds 0
# Ray-CPU, so it is not counted. This is a best-effort threshold for the opt-in "auto"
# mode, not a calibrated value -- the CPU-starvation stall it targets did not reproduce
# on current Ray (see the PR description), so "auto" stays opt-in.
FUSION_DRIVER_CPU_RESERVE = 3.0


def get_value_or_fallback(value: Any, fallback: Any) -> Any:
    """Return value if not None, otherwise return fallback."""
    return value if value is not None else fallback


def validate_stage_groups_names(stage_groups: List[List[str]]) -> List[List[str]]:
    """Field-level (config-time) validation of ``stage_groups``.

    Checks that don't need the enabled-pipeline context: each group is a non-empty
    list of known canonical CPU stage names, the engine is not referenced, names are
    not duplicated across groups, and no group mixes pre- and post-engine stages
    (which cannot be fused across the GPU boundary). Enabled-membership and contiguity
    are re-checked in the builder (where the enabled set is known).
    """
    pre = set(PRE_ENGINE_CPU_STAGE_NAMES)
    post = set(POST_ENGINE_CPU_STAGE_NAMES)
    seen: set = set()
    for group in stage_groups:
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("Each entry in stage_groups must be a non-empty list.")
        for name in group:
            if name == ENGINE_STAGE_NAME:
                raise ValueError(
                    f"stage_groups cannot include the engine stage "
                    f"{ENGINE_STAGE_NAME!r}; it is a fusion boundary."
                )
            if name not in FUSABLE_CPU_STAGE_NAMES:
                raise ValueError(
                    f"Unknown stage name {name!r} in stage_groups. Valid names: "
                    f"{sorted(FUSABLE_CPU_STAGE_NAMES)}."
                )
            if name in seen:
                raise ValueError(
                    f"Stage {name!r} appears in more than one stage_groups group."
                )
            seen.add(name)
        if (set(group) & pre) and (set(group) & post):
            raise ValueError(
                f"stage_groups group {list(group)} mixes pre-engine and post-engine "
                "stages; fusion cannot cross the GPU engine boundary."
            )
    return stage_groups


def extract_resource_kwargs(
    runtime_env: Optional[Dict[str, Any]],
    num_cpus: Optional[float],
    memory: Optional[float],
) -> Dict[str, Any]:
    """Extract non-None resource kwargs for map_batches."""
    kwargs = {}
    if runtime_env is not None:
        kwargs["runtime_env"] = runtime_env
    if num_cpus is not None:
        kwargs["num_cpus"] = num_cpus
    if memory is not None:
        kwargs["memory"] = memory
    return kwargs


def normalize_cpu_stage_concurrency(
    concurrency: Optional[Union[int, Tuple[int, int]]]
) -> Dict[str, int]:
    """Normalize concurrency for CPU stages (int -> (1, int) for autoscaling)."""
    if concurrency is None:
        return {"size": 1}  # Default to minimal autoscaling pool
    if isinstance(concurrency, int):
        return {"min_size": 1, "max_size": concurrency}
    return {
        "min_size": concurrency[0],
        "max_size": concurrency[1],
    }


def build_cpu_stage_map_kwargs(
    stage_cfg: _StageConfigBase,
) -> Dict[str, Any]:
    """Build map_batches_kwargs for CPU stages."""
    concurrency = normalize_cpu_stage_concurrency(stage_cfg.concurrency)
    return dict(
        zero_copy_batch=True,
        compute=ActorPoolStrategy(**concurrency),
        batch_size=stage_cfg.batch_size,
        **extract_resource_kwargs(
            stage_cfg.runtime_env,
            stage_cfg.num_cpus,
            stage_cfg.memory,
        ),
    )


def _concurrency_bounds(
    concurrency: Optional[Union[int, Tuple[int, int]]]
) -> Tuple[int, int]:
    """Return the (min, max) actor-pool bounds implied by a CPU stage concurrency."""
    norm = normalize_cpu_stage_concurrency(concurrency)
    if "size" in norm:
        return norm["size"], norm["size"]
    return norm["min_size"], norm["max_size"]


def build_fused_cpu_stage_map_kwargs(
    stage_cfgs: List[_StageConfigBase],
) -> Dict[str, Any]:
    """Build map_batches_kwargs for a *fused* group of CPU stages.

    The fused group runs all member stages sequentially inside a single actor pool,
    so we synthesize one set of resources from the members: the element-wise max of
    their concurrency bounds, and the max of any explicit ``num_cpus`` / ``memory``
    reservations (so the pool is sized for the most demanding member). ``batch_size``
    and ``runtime_env`` are taken from the first member (all CPU stages inherit the
    processor-level defaults, and the processor overrides ``batch_size`` at call time).

    Args:
        stage_cfgs: The resolved configs of the member stages, in pipeline order.

    Returns:
        The map_batches kwargs for the single fused actor pool.
    """
    if not stage_cfgs:
        raise ValueError("Cannot build fused map kwargs from an empty stage group.")

    min_bounds, max_bounds = zip(
        *(_concurrency_bounds(cfg.concurrency) for cfg in stage_cfgs)
    )
    group_min, group_max = max(min_bounds), max(max_bounds)
    concurrency = (
        {"size": group_max}
        if group_min == group_max
        else {"min_size": group_min, "max_size": group_max}
    )

    num_cpus_values = [c.num_cpus for c in stage_cfgs if c.num_cpus is not None]
    memory_values = [c.memory for c in stage_cfgs if c.memory is not None]
    fused_num_cpus = max(num_cpus_values) if num_cpus_values else None
    fused_memory = max(memory_values) if memory_values else None

    return dict(
        zero_copy_batch=True,
        compute=ActorPoolStrategy(**concurrency),
        batch_size=stage_cfgs[0].batch_size,
        **extract_resource_kwargs(
            stage_cfgs[0].runtime_env,
            fused_num_cpus,
            fused_memory,
        ),
    )


def estimate_free_cpus_for_fusion(total_cpus: Optional[float]) -> Optional[float]:
    """Estimate the CPUs available to the CPU actor-pool stages.

    The signal is the *total* cluster CPUs (``ray.cluster_resources()["CPU"]``), which
    is stable at build time. We intentionally do NOT use ``ray.available_resources()``:
    the builder decides fusion right after ``ray.init`` and before the engine / stage
    actors start, so availability there still ~= total and would mislead.

    On a single-GPU node the vLLM engine actor holds GPU only (0 Ray-CPU), so it does not
    reduce the CPUs available to the CPU stages; we reserve only for the Ray driver and the
    user preprocess/postprocess ``map`` tasks. On autoscaling clusters the GPU worker node
    may not be up yet, so ``total_cpus`` can be partial (head-node only) — that only biases
    the estimate *down*, which biases toward fusing (the safe default).

    Args:
        total_cpus: ``ray.cluster_resources()["CPU"]`` (or None if unavailable).

    Returns:
        The estimated free CPUs, or None if ``total_cpus`` is unknown / non-positive
        (callers treat None as "unknown" and bias toward fusing).
    """
    if not total_cpus or total_cpus <= 0:
        return None
    return max(0.0, total_cpus - FUSION_DRIVER_CPU_RESERVE)


def _segment_enabled_stages(
    enabled_stage_names: List[str],
    barrier_stage_names: AbstractSet[str],
) -> List[Tuple[bool, List[str]]]:
    """Split the ordered enabled stages into (is_fusable, names) segments.

    Each barrier (e.g. the GPU engine) becomes its own non-fusable singleton segment;
    maximal runs of non-barrier stages become fusable segments. This guarantees fusion
    never crosses the engine boundary.
    """
    segments: List[Tuple[bool, List[str]]] = []
    for name in enabled_stage_names:
        is_barrier = name in barrier_stage_names
        if is_barrier:
            segments.append((False, [name]))
        elif segments and segments[-1][0]:
            segments[-1][1].append(name)
        else:
            segments.append((True, [name]))
    return segments


def _validate_stage_groups(
    stage_groups: List[List[str]],
    enabled_stage_names: List[str],
    barrier_stage_names: AbstractSet[str],
) -> Dict[str, int]:
    """Validate explicit ``stage_groups`` against the enabled pipeline.

    Rejects: empty groups, the engine/barrier appearing in a group, names that are not
    enabled, duplicate names across groups, and groups whose members are not a
    contiguous run within a single fusable segment (which would require reordering or
    crossing the engine boundary).

    Returns:
        A mapping of stage name -> group id for the (valid) multi-member groups.
    """
    position = {name: i for i, name in enumerate(enabled_stage_names)}
    name_to_gid: Dict[str, int] = {}
    seen: set = set()

    for gid, group in enumerate(stage_groups):
        if not group:
            raise ValueError("stage_groups must not contain empty groups.")
        for name in group:
            if name in barrier_stage_names:
                raise ValueError(
                    f"stage_groups cannot include the engine stage {name!r}; "
                    "the GPU engine is a fusion boundary."
                )
            if name not in position:
                raise ValueError(
                    f"stage_groups references stage {name!r}, which is not an enabled "
                    f"CPU stage. Enabled stages: {enabled_stage_names}."
                )
            if name in seen:
                raise ValueError(
                    f"stage_groups lists stage {name!r} in more than one group."
                )
            seen.add(name)
            name_to_gid[name] = gid

    # Contiguity / single-segment check: each group's enabled positions must form a
    # consecutive block with no foreign stage (or barrier) interleaved.
    for gid, group in enumerate(stage_groups):
        positions = sorted(position[name] for name in group)
        block = range(positions[0], positions[-1] + 1)
        if list(block) != positions:
            interleaved = [
                enabled_stage_names[p] for p in block if name_to_gid.get(
                    enabled_stage_names[p]
                ) != gid
            ]
            raise ValueError(
                f"stage_groups group {group} is not contiguous in the pipeline; "
                f"cannot fuse across {interleaved}. Groups must be adjacent stages on "
                "the same side of the GPU engine."
            )
    return name_to_gid


def plan_stage_fusion(
    enabled_stage_names: List[str],
    mode: Union[bool, str],
    stage_groups: Optional[List[List[str]]],
    free_cpus: Optional[float],
    *,
    barrier_stage_names: AbstractSet[str] = frozenset({ENGINE_STAGE_NAME}),
) -> List[List[str]]:
    """Decide how to group the enabled stages into actor pools. PURE: no ray, no I/O.

    Args:
        enabled_stage_names: Ordered canonical names of the enabled stages, including
            the engine (barrier) stage.
        mode: ``fuse_cpu_stages`` value -- ``False`` (never fuse), ``True`` (always fuse
            each contiguous CPU run), or ``"auto"`` (fuse based on ``free_cpus``).
        stage_groups: Optional explicit grouping (takes precedence over ``mode``). Each
            inner list is a set of adjacent CPU stage names to fuse together.
        free_cpus: CPUs estimated free for CPU stages (see
            ``estimate_free_cpus_for_fusion``). ``None`` means unknown -> bias to fuse.
        barrier_stage_names: Names that must never be fused (the GPU engine). Each is
            emitted as its own singleton group.

    Returns:
        The ordered list of groups (lists of canonical names). Barriers and unfused
        stages are singletons; fused stages share a list. Concatenating the groups
        reproduces ``enabled_stage_names`` in order.
    """
    segments = _segment_enabled_stages(enabled_stage_names, barrier_stage_names)
    num_cpu_stages = sum(
        len(names) for is_fusable, names in segments if is_fusable
    )

    # Explicit grouping wins over the mode heuristic.
    if stage_groups is not None:
        name_to_gid = _validate_stage_groups(
            stage_groups, enabled_stage_names, barrier_stage_names
        )
        groups: List[List[str]] = []
        for is_fusable, names in segments:
            if not is_fusable:
                groups.append(list(names))
                continue
            current: List[str] = []
            current_gid: Optional[int] = None
            for name in names:
                gid = name_to_gid.get(name)
                if gid is not None and gid == current_gid:
                    current.append(name)
                else:
                    if current:
                        groups.append(current)
                    current = [name]
                    current_gid = gid
            if current:
                groups.append(current)
        return groups

    # Mode heuristic: decide whether to fuse each contiguous CPU segment wholesale.
    if mode is False:
        should_fuse = False
    elif mode is True:
        should_fuse = True
    elif mode == "auto":
        # Fuse when the CPU actor-pool stages can't each get a dedicated free CPU
        # alongside the engine + driver overhead. Ties fuse (bias toward fusing); an
        # unknown CPU count also fuses.
        should_fuse = free_cpus is None or free_cpus <= num_cpu_stages
    else:
        raise ValueError(
            f"Invalid fuse_cpu_stages value {mode!r}; expected True, False, or 'auto'."
        )

    groups = []
    for is_fusable, names in segments:
        if is_fusable and should_fuse:
            groups.append(list(names))
        else:
            groups.extend([name] for name in names)
    return groups
