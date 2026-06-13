"""Unit tests for CPU stage fusion (fuse_cpu_stages / stage_groups).

These are CPU-only and require no GPU, no ray.init, and no tokenizer downloads:
- the planner (`plan_stage_fusion`) and helpers are pure functions;
- the composite UDF is exercised with lightweight fake member UDFs;
- graph-shape is asserted via `_apply_stage_fusion` over cheaply-constructed stage
  objects (constructing a stage does NOT instantiate its UDF / download a model).
"""

import asyncio
import sys

import pydantic
import pytest

from ray.data._internal.compute import ActorPoolStrategy
from ray.llm._internal.batch.processor.utils import (
    ENGINE_STAGE_NAME,
    FUSION_DRIVER_CPU_RESERVE,
    build_fused_cpu_stage_map_kwargs,
    estimate_free_cpus_for_fusion,
    plan_stage_fusion,
    validate_stage_groups_names,
)
from ray.llm._internal.batch.processor.vllm_engine_proc import (
    _apply_stage_fusion,
    vLLMEngineProcessorConfig,
)
from ray.llm._internal.batch.stages import (
    ChatTemplateStage,
    DetokenizeStage,
    FusedStage,
    PrepareMultimodalStage,
    TokenizeStage,
    vLLMEngineStage,
)
from ray.llm._internal.batch.stages.base import StatefulStageUDF
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    PrepareMultimodalStageConfig,
    TokenizerStageConfig,
)
from ray.llm._internal.batch.stages.fused_stage import FusedStageUDF

# Full multimodal pipeline (4 fusable CPU stages + engine barrier).
MULTIMODAL_PIPELINE = [
    "prepare_multimodal",
    "chat_template",
    "tokenize",
    ENGINE_STAGE_NAME,
    "detokenize",
]
PRE_ENGINE_GROUP = ["prepare_multimodal", "chat_template", "tokenize"]


# --------------------------------------------------------------------------- #
# Pure planner: the four modes.
# --------------------------------------------------------------------------- #
def test_mode_false_never_fuses():
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, False, None, free_cpus=0.0)
    assert groups == [[name] for name in MULTIMODAL_PIPELINE]


def test_mode_true_fuses_pre_engine_segment():
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, True, None, free_cpus=9999.0)
    assert groups == [PRE_ENGINE_GROUP, [ENGINE_STAGE_NAME], ["detokenize"]]


def test_mode_auto_fuses_when_cpu_constrained():
    # 4 fusable stages, free_cpus == 4 -> tie -> fuse (bias toward fusing).
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, "auto", None, free_cpus=4.0)
    assert groups == [PRE_ENGINE_GROUP, [ENGINE_STAGE_NAME], ["detokenize"]]


def test_mode_auto_does_not_fuse_when_cpu_rich():
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, "auto", None, free_cpus=28.0)
    assert groups == [[name] for name in MULTIMODAL_PIPELINE]


def test_mode_auto_unknown_cpus_biases_to_fuse():
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, "auto", None, free_cpus=None)
    assert groups == [PRE_ENGINE_GROUP, [ENGINE_STAGE_NAME], ["detokenize"]]


def test_mode_auto_just_above_threshold_does_not_fuse():
    # free_cpus (5) > num_cpu_stages (4) -> do not fuse.
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, "auto", None, free_cpus=5.0)
    assert groups == [[name] for name in MULTIMODAL_PIPELINE]


def test_mode_invalid_raises():
    with pytest.raises(ValueError, match="Invalid fuse_cpu_stages"):
        plan_stage_fusion(MULTIMODAL_PIPELINE, "sometimes", None, None)


# --------------------------------------------------------------------------- #
# Pure planner: structural invariants.
# --------------------------------------------------------------------------- #
def test_engine_is_never_fused():
    for mode in (True, "auto"):
        groups = plan_stage_fusion(MULTIMODAL_PIPELINE, mode, None, free_cpus=0.0)
        assert [ENGINE_STAGE_NAME] in groups


def test_detokenize_stays_separate_from_pre_engine():
    # Post-engine CPU stage must not fuse with pre-engine stages.
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, True, None, free_cpus=0.0)
    assert ["detokenize"] in groups
    assert all("detokenize" not in g or g == ["detokenize"] for g in groups)


def test_single_pre_engine_stage_is_not_wrapped():
    pipeline = ["tokenize", ENGINE_STAGE_NAME, "detokenize"]
    groups = plan_stage_fusion(pipeline, True, None, free_cpus=0.0)
    assert groups == [["tokenize"], [ENGINE_STAGE_NAME], ["detokenize"]]


def test_groups_concatenate_back_to_enabled_order():
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, "auto", None, free_cpus=1.0)
    flattened = [name for group in groups for name in group]
    assert flattened == MULTIMODAL_PIPELINE


# --------------------------------------------------------------------------- #
# Pure planner: explicit stage_groups.
# --------------------------------------------------------------------------- #
def test_explicit_groups_match_task_example():
    groups = plan_stage_fusion(
        MULTIMODAL_PIPELINE,
        "auto",
        [["prepare_multimodal", "chat_template", "tokenize"], ["detokenize"]],
        free_cpus=9999.0,  # ignored: explicit groups take precedence
    )
    assert groups == [PRE_ENGINE_GROUP, [ENGINE_STAGE_NAME], ["detokenize"]]


def test_explicit_partial_group_leaves_rest_as_singletons():
    groups = plan_stage_fusion(
        MULTIMODAL_PIPELINE, "auto", [["chat_template", "tokenize"]], None
    )
    assert groups == [
        ["prepare_multimodal"],
        ["chat_template", "tokenize"],
        [ENGINE_STAGE_NAME],
        ["detokenize"],
    ]


def test_explicit_groups_take_precedence_over_mode_false():
    groups = plan_stage_fusion(
        MULTIMODAL_PIPELINE, False, [["chat_template", "tokenize"]], None
    )
    assert ["chat_template", "tokenize"] in groups


def test_explicit_group_rejects_cross_engine_boundary():
    with pytest.raises(ValueError, match="not contiguous"):
        plan_stage_fusion(
            MULTIMODAL_PIPELINE, "auto", [["tokenize", "detokenize"]], None
        )


def test_explicit_group_rejects_non_contiguous():
    with pytest.raises(ValueError, match="not contiguous"):
        plan_stage_fusion(
            MULTIMODAL_PIPELINE, "auto", [["prepare_multimodal", "tokenize"]], None
        )


def test_explicit_group_rejects_disabled_stage():
    with pytest.raises(ValueError, match="not an enabled"):
        plan_stage_fusion(
            ["chat_template", "tokenize", ENGINE_STAGE_NAME],
            "auto",
            [["chat_template", "detokenize"]],
            None,
        )


def test_explicit_group_rejects_engine():
    with pytest.raises(ValueError, match="engine"):
        plan_stage_fusion(
            MULTIMODAL_PIPELINE, "auto", [["tokenize", ENGINE_STAGE_NAME]], None
        )


# --------------------------------------------------------------------------- #
# Auto heuristic CPU estimate (the calibration the live repro validates).
# --------------------------------------------------------------------------- #
def test_estimate_free_cpus_reserves_driver_only():
    # The engine holds 0 Ray-CPU (GPU-only), so only the driver/user-map reserve is
    # subtracted from the cluster's total CPUs (engine size does not factor in).
    assert estimate_free_cpus_for_fusion(8) == 8 - FUSION_DRIVER_CPU_RESERVE


def test_estimate_free_cpus_unknown_total_is_none():
    assert estimate_free_cpus_for_fusion(None) is None
    assert estimate_free_cpus_for_fusion(0) is None


def test_estimate_never_negative():
    assert estimate_free_cpus_for_fusion(1) == 0.0


def test_small_node_multimodal_auto_fuses_end_to_end_estimate():
    # 4 vCPU -> free 1, which is <= 4 CPU stages -> fuse.
    free = estimate_free_cpus_for_fusion(4)
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, "auto", None, free)
    assert groups[0] == PRE_ENGINE_GROUP


def test_large_node_multimodal_auto_does_not_fuse_end_to_end_estimate():
    # 32 vCPU -> free 29, which is > 4 CPU stages -> no fuse.
    free = estimate_free_cpus_for_fusion(32)
    groups = plan_stage_fusion(MULTIMODAL_PIPELINE, "auto", None, free)
    assert groups == [[name] for name in MULTIMODAL_PIPELINE]


# --------------------------------------------------------------------------- #
# Fused-pool map kwargs.
# --------------------------------------------------------------------------- #
def test_build_fused_map_kwargs_takes_max_resources():
    cfgs = [
        ChatTemplateStageConfig(batch_size=64, concurrency=2, num_cpus=1.0),
        TokenizerStageConfig(batch_size=64, concurrency=4, num_cpus=2.0, memory=100.0),
    ]
    kwargs = build_fused_cpu_stage_map_kwargs(cfgs)
    assert kwargs["zero_copy_batch"] is True
    assert kwargs["batch_size"] == 64
    assert kwargs["num_cpus"] == 2.0  # max across members
    assert kwargs["memory"] == 100.0
    assert isinstance(kwargs["compute"], ActorPoolStrategy)


def test_build_fused_map_kwargs_no_resources_when_unset():
    kwargs = build_fused_cpu_stage_map_kwargs(
        [ChatTemplateStageConfig(), TokenizerStageConfig()]
    )
    assert "num_cpus" not in kwargs
    assert "memory" not in kwargs
    assert isinstance(kwargs["compute"], ActorPoolStrategy)


def test_build_fused_map_kwargs_empty_raises():
    with pytest.raises(ValueError):
        build_fused_cpu_stage_map_kwargs([])


# --------------------------------------------------------------------------- #
# Composite UDF mechanism: chaining member udf()s preserves the column contract.
# --------------------------------------------------------------------------- #
class _AddOneUDF(StatefulStageUDF):
    """Reads ``in_key`` and writes ``out_key = in_key + 1`` per row."""

    def __init__(self, data_column, expected_input_keys, in_key, out_key):
        super().__init__(data_column, expected_input_keys)
        self.in_key = in_key
        self.out_key = out_key

    async def udf(self, rows):
        for row in rows:
            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                self.out_key: row[self.in_key] + 1,
            }


class _ReverseAddOneUDF(StatefulStageUDF):
    """Like _AddOneUDF but yields outputs in REVERSED order (still echoing IDX)."""

    def __init__(self, data_column, expected_input_keys, in_key, out_key):
        super().__init__(data_column, expected_input_keys)
        self.in_key = in_key
        self.out_key = out_key

    async def udf(self, rows):
        for row in reversed(rows):
            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                self.out_key: row[self.in_key] + 1,
            }


class _EchoColumnUDF(StatefulStageUDF):
    """Re-emits a pre-existing column (exercises the produced-accumulation path)."""

    def __init__(self, data_column, expected_input_keys, key):
        super().__init__(data_column, expected_input_keys)
        self.key = key

    async def udf(self, rows):
        for row in rows:
            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                self.key: row[self.key],
            }


class _BadIndexUDF(StatefulStageUDF):
    """Emits an index not present in the input batch (violates the member contract)."""

    async def udf(self, rows):
        for _ in rows:
            yield {self.IDX_IN_BATCH_COLUMN: 999, "x": 1}


class _NoIndexUDF(StatefulStageUDF):
    """Drops IDX_IN_BATCH_COLUMN from its output (violates the member contract)."""

    async def udf(self, rows):
        for _ in rows:
            yield {"x": 1}


def _run_udf(udf, batch):
    async def _collect():
        return [out async for out in udf(batch)]

    return asyncio.run(_collect())


def test_fused_udf_threads_columns_through_members():
    fused = FusedStageUDF(
        data_column="__data",
        expected_input_keys=["a"],
        fused_members=[
            {
                "fn": _AddOneUDF,
                "fn_constructor_kwargs": {"in_key": "a", "out_key": "b"},
                "expected_input_keys": ["a"],
            },
            {
                "fn": _AddOneUDF,
                "fn_constructor_kwargs": {"in_key": "b", "out_key": "c"},
                "expected_input_keys": ["b"],
            },
        ],
    )
    outputs = _run_udf(fused, {"__data": [{"a": 1}, {"a": 10}]})

    assert len(outputs) == 1
    rows = outputs[0]["__data"]
    assert rows[0] == {"a": 1, "b": 2, "c": 3}
    assert rows[1] == {"a": 10, "b": 11, "c": 12}
    # The reserved index column must not leak into the output.
    assert all(StatefulStageUDF.IDX_IN_BATCH_COLUMN not in row for row in rows)


def test_fused_udf_validates_first_member_required_keys():
    fused = FusedStageUDF(
        data_column="__data",
        expected_input_keys=["a"],  # first member's required key
        fused_members=[
            {
                "fn": _AddOneUDF,
                "fn_constructor_kwargs": {"in_key": "a", "out_key": "b"},
                "expected_input_keys": ["a"],
            },
        ],
    )
    with pytest.raises(ValueError, match="Required input keys"):
        _run_udf(fused, {"__data": [{"wrong_key": 1}]})


def test_fused_udf_reconciles_out_of_order_member_output():
    # First member yields in reversed order; results must still match by index, and
    # the second member must read the correctly-merged column from the first.
    fused = FusedStageUDF(
        data_column="__data",
        expected_input_keys=["a"],
        fused_members=[
            {
                "fn": _ReverseAddOneUDF,
                "fn_constructor_kwargs": {"in_key": "a", "out_key": "b"},
                "expected_input_keys": ["a"],
            },
            {
                "fn": _AddOneUDF,
                "fn_constructor_kwargs": {"in_key": "b", "out_key": "c"},
                "expected_input_keys": ["b"],
            },
        ],
    )
    rows = _run_udf(fused, {"__data": [{"a": 1}, {"a": 10}, {"a": 100}]})[0]["__data"]
    assert rows[0] == {"a": 1, "b": 2, "c": 3}
    assert rows[1] == {"a": 10, "b": 11, "c": 12}
    assert rows[2] == {"a": 100, "b": 101, "c": 102}


def test_fused_udf_member_can_reemit_existing_column():
    # A member that re-emits an existing column must not corrupt the row; the next
    # member still reads the value and adds its own.
    fused = FusedStageUDF(
        data_column="__data",
        expected_input_keys=["a"],
        fused_members=[
            {
                "fn": _EchoColumnUDF,
                "fn_constructor_kwargs": {"key": "a"},
                "expected_input_keys": ["a"],
            },
            {
                "fn": _AddOneUDF,
                "fn_constructor_kwargs": {"in_key": "a", "out_key": "b"},
                "expected_input_keys": ["a"],
            },
        ],
    )
    rows = _run_udf(fused, {"__data": [{"a": 5}, {"a": 7}]})[0]["__data"]
    assert rows[0] == {"a": 5, "b": 6}
    assert rows[1] == {"a": 7, "b": 8}


def test_fused_udf_raises_on_unknown_member_index():
    fused = FusedStageUDF(
        data_column="__data",
        expected_input_keys=["a"],
        fused_members=[
            {
                "fn": _BadIndexUDF,
                "fn_constructor_kwargs": {},
                "expected_input_keys": ["a"],
            },
        ],
    )
    with pytest.raises(ValueError, match="unknown row index"):
        _run_udf(fused, {"__data": [{"a": 1}]})


def test_fused_udf_raises_when_member_drops_index():
    fused = FusedStageUDF(
        data_column="__data",
        expected_input_keys=["a"],
        fused_members=[
            {
                "fn": _NoIndexUDF,
                "fn_constructor_kwargs": {},
                "expected_input_keys": ["a"],
            },
        ],
    )
    with pytest.raises(ValueError, match="without the"):
        _run_udf(fused, {"__data": [{"a": 1}]})


# --------------------------------------------------------------------------- #
# FusedStage metadata.
# --------------------------------------------------------------------------- #
def test_fused_stage_name_and_keys():
    stage = FusedStage(
        fn_constructor_kwargs={"fused_members": []},
        member_stage_names=["ChatTemplateStage", "TokenizeStage"],
        required_input_keys={"messages": "desc"},
        optional_input_keys={"foo": "bar"},
    )
    assert stage.get_stage_name() == "FusedStage(ChatTemplateStage+TokenizeStage)"
    assert stage.get_required_input_keys() == {"messages": "desc"}
    assert stage.get_optional_input_keys() == {"foo": "bar"}


# --------------------------------------------------------------------------- #
# Config-time validation of the new fields.
# --------------------------------------------------------------------------- #
def test_config_default_is_false():
    config = vLLMEngineProcessorConfig(model_source="test-model")
    assert config.fuse_cpu_stages is False
    assert config.stage_groups is None


@pytest.mark.parametrize("value", [True, False, "auto"])
def test_config_accepts_valid_fuse_values(value):
    config = vLLMEngineProcessorConfig(model_source="test-model", fuse_cpu_stages=value)
    assert config.fuse_cpu_stages == value


@pytest.mark.parametrize("value", ["yes", "always", 2, "True"])
def test_config_rejects_invalid_fuse_values(value):
    with pytest.raises(pydantic.ValidationError):
        vLLMEngineProcessorConfig(model_source="test-model", fuse_cpu_stages=value)


def test_config_accepts_valid_stage_groups():
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        stage_groups=[["prepare_multimodal", "chat_template", "tokenize"], ["detokenize"]],
    )
    assert config.stage_groups == [
        ["prepare_multimodal", "chat_template", "tokenize"],
        ["detokenize"],
    ]


def test_config_rejects_unknown_stage_name():
    with pytest.raises(pydantic.ValidationError, match="Unknown stage name"):
        vLLMEngineProcessorConfig(model_source="test-model", stage_groups=[["bogus"]])


def test_config_rejects_engine_in_groups():
    with pytest.raises(pydantic.ValidationError):
        vLLMEngineProcessorConfig(
            model_source="test-model", stage_groups=[[ENGINE_STAGE_NAME]]
        )


def test_config_rejects_duplicate_stage_across_groups():
    with pytest.raises(pydantic.ValidationError, match="more than one"):
        vLLMEngineProcessorConfig(
            model_source="test-model", stage_groups=[["tokenize"], ["tokenize"]]
        )


def test_config_rejects_cross_boundary_group():
    with pytest.raises(pydantic.ValidationError):
        vLLMEngineProcessorConfig(
            model_source="test-model", stage_groups=[["tokenize", "detokenize"]]
        )


def test_validate_stage_groups_names_accepts_valid():
    groups = [["chat_template", "tokenize"]]
    assert validate_stage_groups_names(groups) == groups


# --------------------------------------------------------------------------- #
# Graph shape via _apply_stage_fusion (no ray.init / downloads needed).
# --------------------------------------------------------------------------- #
def _built_multimodal_stages():
    """Cheap (name, stage, cfg) tuples; constructing a stage does not load a model."""
    return [
        ("prepare_multimodal", PrepareMultimodalStage(), PrepareMultimodalStageConfig()),
        ("chat_template", ChatTemplateStage(), ChatTemplateStageConfig()),
        ("tokenize", TokenizeStage(), TokenizerStageConfig()),
        (
            ENGINE_STAGE_NAME,
            vLLMEngineStage(
                fn_constructor_kwargs={"engine_kwargs": {}}, map_batches_kwargs={}
            ),
            None,
        ),
        ("detokenize", DetokenizeStage(), DetokenizeStageConfig()),
    ]


def _stage_names(stages):
    return [s.get_stage_name() for s in stages]


def test_apply_fusion_auto_fuses_on_small_node(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 4.0})
    config = vLLMEngineProcessorConfig(
        model_source="test-model", fuse_cpu_stages="auto", concurrency=1
    )
    stages = _apply_stage_fusion(config, _built_multimodal_stages())
    assert _stage_names(stages) == [
        "FusedStage(PrepareMultimodalStage+ChatTemplateStage+TokenizeStage)",
        "vLLMEngineStage",
        "DetokenizeStage",
    ]
    # The fused stage carries the first member's required input keys.
    assert "messages" in stages[0].get_required_input_keys()


def test_apply_fusion_auto_keeps_pools_on_large_node(monkeypatch):
    import ray

    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 64.0})
    config = vLLMEngineProcessorConfig(
        model_source="test-model", fuse_cpu_stages="auto", concurrency=1
    )
    stages = _apply_stage_fusion(config, _built_multimodal_stages())
    assert _stage_names(stages) == [
        "PrepareMultimodalStage",
        "ChatTemplateStage",
        "TokenizeStage",
        "vLLMEngineStage",
        "DetokenizeStage",
    ]


def test_apply_fusion_false_keeps_all_pools():
    config = vLLMEngineProcessorConfig(
        model_source="test-model", fuse_cpu_stages=False, concurrency=1
    )
    stages = _apply_stage_fusion(config, _built_multimodal_stages())
    assert len(stages) == 5
    assert not any(isinstance(s, FusedStage) for s in stages)


def test_apply_fusion_true_fuses_regardless_of_node():
    config = vLLMEngineProcessorConfig(
        model_source="test-model", fuse_cpu_stages=True, concurrency=1
    )
    stages = _apply_stage_fusion(config, _built_multimodal_stages())
    assert _stage_names(stages) == [
        "FusedStage(PrepareMultimodalStage+ChatTemplateStage+TokenizeStage)",
        "vLLMEngineStage",
        "DetokenizeStage",
    ]


def test_apply_fusion_explicit_stage_groups():
    config = vLLMEngineProcessorConfig(
        model_source="test-model",
        stage_groups=[["chat_template", "tokenize"]],
        concurrency=1,
    )
    stages = _apply_stage_fusion(config, _built_multimodal_stages())
    assert _stage_names(stages) == [
        "PrepareMultimodalStage",
        "FusedStage(ChatTemplateStage+TokenizeStage)",
        "vLLMEngineStage",
        "DetokenizeStage",
    ]


def test_apply_fusion_logs_once_when_active(caplog):
    import logging

    mod_logger = logging.getLogger(
        "ray.llm._internal.batch.processor.vllm_engine_proc"
    )
    config = vLLMEngineProcessorConfig(
        model_source="test-model", fuse_cpu_stages=True, concurrency=1
    )
    # Ray configures its loggers with propagate=False, so records never reach the
    # root logger that caplog captures by default. Attach caplog's handler directly
    # to the module logger, and disable propagation to avoid double-counting.
    orig_level, orig_propagate = mod_logger.level, mod_logger.propagate
    caplog.handler.setLevel(logging.INFO)
    mod_logger.setLevel(logging.INFO)
    mod_logger.propagate = False
    mod_logger.addHandler(caplog.handler)
    try:
        _apply_stage_fusion(config, _built_multimodal_stages())
    finally:
        mod_logger.removeHandler(caplog.handler)
        mod_logger.setLevel(orig_level)
        mod_logger.propagate = orig_propagate
    fusion_logs = [r for r in caplog.records if "stage fusion active" in r.getMessage()]
    assert len(fusion_logs) == 1


# --------------------------------------------------------------------------- #
# Backward-compat: the new fields default to a no-op (fusion off) and are quiet.
# --------------------------------------------------------------------------- #
def test_default_is_noop_and_emits_no_warnings():
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = vLLMEngineProcessorConfig(model_source="test-model")
    assert config.fuse_cpu_stages is False
    assert config.stage_groups is None
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not user_warnings


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
