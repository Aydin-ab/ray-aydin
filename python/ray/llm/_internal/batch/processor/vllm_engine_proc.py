"""The vLLM engine processor."""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

import transformers
from pydantic import Field, field_validator, model_validator

import ray
from ray.data.block import UserDefinedFunction
from ray.llm._internal.batch.constants import TypeVLLMTaskType, vLLMTaskType
from ray.llm._internal.batch.observability.usage_telemetry.usage import (
    BatchModelTelemetry,
    TelemetryAgent,
    get_or_create_telemetry_agent,
)
from ray.llm._internal.batch.processor.base import (
    DEFAULT_MAX_TASKS_IN_FLIGHT,
    OfflineProcessorConfig,
    Processor,
    ProcessorBuilder,
)
from ray.llm._internal.batch.processor.utils import (
    ENGINE_STAGE_NAME,
    build_cpu_stage_map_kwargs,
    build_fused_cpu_stage_map_kwargs,
    estimate_free_cpus_for_fusion,
    get_value_or_fallback,
    plan_stage_fusion,
    validate_stage_groups_names,
)
from ray.llm._internal.batch.stages import (
    ChatTemplateStage,
    DetokenizeStage,
    FusedStage,
    PrepareImageStage,
    PrepareMultimodalStage,
    TokenizeStage,
    vLLMEngineStage,
)
from ray.llm._internal.batch.stages.configs import (
    ChatTemplateStageConfig,
    DetokenizeStageConfig,
    PrepareImageStageConfig,
    PrepareMultimodalStageConfig,
    TokenizerStageConfig,
    resolve_stage_config,
)
from ray.llm._internal.common.observability.telemetry_utils import DEFAULT_GPU_TYPE
from ray.llm._internal.common.placement import PlacementGroupConfig
from ray.llm._internal.common.utils.download_utils import (
    STREAMING_LOAD_FORMATS,
    NodeModelDownloadable,
    download_model_files,
)

logger = logging.getLogger(__name__)


DEFAULT_MODEL_ARCHITECTURE = "UNKNOWN_MODEL_ARCHITECTURE"


class vLLMEngineProcessorConfig(OfflineProcessorConfig):
    """The configuration for the vLLM engine processor."""

    # vLLM stage configurations.
    engine_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="The kwargs to pass to the vLLM engine. See "
        "https://docs.vllm.ai/en/latest/serving/engine_args.html "
        "for more details.",
    )
    task_type: TypeVLLMTaskType = Field(
        default=vLLMTaskType.GENERATE,
        description="The task type to use. If not specified, will use "
        "'generate' by default.",
    )
    log_engine_metrics: bool = Field(
        default=True,
        description="Enable vLLM engine metrics export via Ray's Prometheus endpoint. "
        "When enabled, metrics like prefix cache hit rate, TTFT, TPOT, KV cache "
        "utilization, and scheduler state are available at Ray's metrics endpoint. "
        "Requires Ray to be initialized with _metrics_export_port "
        "(e.g., ray.init(_metrics_export_port=8080)).",
    )
    # LoRA configurations.
    dynamic_lora_loading_path: Optional[str] = Field(
        default=None,
        description="The path to the dynamic LoRA adapter. It is expected "
        "to hold subfolders each for a different lora checkpoint. If not "
        "specified and LoRA is enabled, then the 'model' in LoRA "
        "requests will be interpreted as model ID used by HF transformers.",
    )
    # Custom placement group config for TP/PP.
    placement_group_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Ray placement group configuration for scheduling vLLM engine workers. "
        "Can specify either 'bundle_per_worker' (auto-replicated by tp*pp) or 'bundles' "
        "(full list of resource dicts). Optionally include 'strategy' key "
        "('PACK', 'STRICT_PACK', 'SPREAD', or 'STRICT_SPREAD'). "
        "Example with bundle_per_worker: {'bundle_per_worker': {'CPU': 1, 'GPU': 1}, 'strategy': 'SPREAD'}. "
        "Example with bundles: {'bundles': [{'CPU': 1, 'GPU': 1}] * 4, 'strategy': 'SPREAD'}.",
    )
    # Stage fusion configuration.
    fuse_cpu_stages: Union[bool, Literal["auto"]] = Field(
        default=False,
        description="Opt-in optimization to consolidate the CPU stages "
        "(prepare_multimodal/prepare_image, chat_template, tokenize, detokenize) into a "
        "single Ray actor pool instead of one pool per stage. Each separate pool can "
        "scale up to `concurrency` actors reserving ~1 CPU each, so on small, "
        "CPU-constrained GPU nodes the per-stage pools can oversubscribe CPUs and split "
        "object-store memory across many operators, reducing throughput; consolidating "
        "them can help. Options: False (default) keeps one pool per stage; True always "
        "fuses the pre-engine CPU stages into one pool; 'auto' is a best-effort heuristic "
        "that fuses them when the node looks CPU-constrained for the pipeline. detokenize "
        "(after the GPU engine) always stays separate. When fusion activates a one-line "
        "message is logged. Override grouping with `stage_groups`.",
    )
    stage_groups: Optional[List[List[str]]] = Field(
        default=None,
        description="Explicit CPU stage fusion groups, taking precedence over "
        "`fuse_cpu_stages`. Each inner list names adjacent CPU stages to fuse into one "
        "actor pool, e.g. [['prepare_multimodal', 'chat_template', 'tokenize'], "
        "['detokenize']]. Valid names: 'prepare_image', 'prepare_multimodal', "
        "'chat_template', 'tokenize', 'detokenize'. A group cannot cross the GPU engine "
        "boundary or include disabled stages. Stages not listed run in their own pool.",
    )

    @field_validator("fuse_cpu_stages", mode="before")
    @classmethod
    def validate_fuse_cpu_stages(cls, value):
        if isinstance(value, bool) or value == "auto":
            return value
        raise ValueError(
            f"Invalid fuse_cpu_stages value {value!r}; expected True, False, or 'auto'."
        )

    @field_validator("stage_groups")
    @classmethod
    def validate_stage_groups_field(cls, value):
        if value is None:
            return None
        return validate_stage_groups_names(value)

    @model_validator(mode="before")
    @classmethod
    def validate_task_type(cls, values):
        task_type = values.get("task_type", vLLMTaskType.GENERATE)
        if task_type not in vLLMTaskType.values():
            raise ValueError(f"Invalid task type: {task_type}")

        engine_kwargs = values.get("engine_kwargs", {})
        engine_kwargs_task_type = engine_kwargs.get("task_type", "")
        if engine_kwargs_task_type != task_type:
            if engine_kwargs_task_type:
                logger.warning(
                    "The task_type set in engine kwargs (%s) is different from the "
                    "config (%s). Overriding the task_type in engine kwargs to %s.",
                    engine_kwargs_task_type,
                    task_type,
                    task_type,
                )
            engine_kwargs["task_type"] = task_type
        values["engine_kwargs"] = engine_kwargs
        return values

    @field_validator("placement_group_config")
    @classmethod
    def validate_placement_group_config(cls, value):
        if value is None:
            return None
        # Validate through PlacementGroupConfig, then dump back to dict
        validated = PlacementGroupConfig(**value)
        return validated.model_dump()


def _apply_stage_fusion(config, built_stages):
    """Group the enabled CPU stages into actor pools per the fusion config.

    Builds the final ordered stage list: contiguous CPU stages that the planner groups
    together are wrapped into a single ``FusedStage`` (one actor pool); everything else
    is kept as-is. Emits a one-line log when any group is fused.

    Args:
        config: The vLLM processor config (provides ``fuse_cpu_stages``/``stage_groups``
            and the engine sizing used to estimate free CPUs for ``"auto"``).
        built_stages: Ordered ``(canonical_name, stage, stage_cfg)`` tuples; the engine
            carries a ``None`` cfg.

    Returns:
        The final ordered list of ``StatefulStage`` objects to pass to ``Processor``.
    """
    enabled_names = [name for name, _, _ in built_stages]

    # Only the "auto" heuristic needs the CPU count; True/False/explicit do not.
    free_cpus = None
    if config.fuse_cpu_stages == "auto" and config.stage_groups is None:
        try:
            total_cpus = ray.cluster_resources().get("CPU", 0.0)
        except Exception:
            total_cpus = None
        free_cpus = estimate_free_cpus_for_fusion(total_cpus)

    groups = plan_stage_fusion(
        enabled_names,
        config.fuse_cpu_stages,
        config.stage_groups,
        free_cpus,
    )

    by_name = {name: (stage, cfg) for name, stage, cfg in built_stages}
    final_stages = []
    fused_groups = []
    for group in groups:
        if len(group) == 1:
            final_stages.append(by_name[group[0]][0])
            continue
        member_stages = [by_name[name][0] for name in group]
        member_cfgs = [by_name[name][1] for name in group]
        first_stage = member_stages[0]
        fused_members = [
            {
                "fn": stage.fn,
                "fn_constructor_kwargs": stage.fn_constructor_kwargs,
                "expected_input_keys": list(stage.get_required_input_keys().keys()),
            }
            for stage in member_stages
        ]
        final_stages.append(
            FusedStage(
                fn_constructor_kwargs={"fused_members": fused_members},
                map_batches_kwargs=build_fused_cpu_stage_map_kwargs(member_cfgs),
                member_stage_names=[type(s).__name__ for s in member_stages],
                required_input_keys=first_stage.get_required_input_keys(),
                optional_input_keys=first_stage.get_optional_input_keys(),
            )
        )
        fused_groups.append(group)

    if fused_groups:
        logger.info(
            "Ray Data LLM stage fusion active: fused CPU stage group(s) %s into a "
            "single actor pool each to reduce CPU contention (fuse_cpu_stages=%r, "
            "estimated_free_cpus=%s). Set fuse_cpu_stages=False to disable, or use "
            "stage_groups to customize grouping.",
            fused_groups,
            config.fuse_cpu_stages,
            "unknown" if free_cpus is None else round(free_cpus, 1),
        )

    return final_stages


def build_vllm_engine_processor(
    config: vLLMEngineProcessorConfig,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    preprocess: Optional[UserDefinedFunction] = None,
    postprocess: Optional[UserDefinedFunction] = None,
    preprocess_map_kwargs: Optional[Dict[str, Any]] = None,
    postprocess_map_kwargs: Optional[Dict[str, Any]] = None,
    telemetry_agent: Optional[TelemetryAgent] = None,
) -> Processor:
    """Construct a Processor and configure stages.

    Args:
        config: The configuration for the processor.
        chat_template_kwargs: The optional kwargs to pass to apply_chat_template.
        preprocess: An optional lambda function that takes a row (dict) as input
            and returns a preprocessed row (dict). The output row must contain the
            required fields for the following processing stages.
        postprocess: An optional lambda function that takes a row (dict) as input
            and returns a postprocessed row (dict).
        preprocess_map_kwargs: Optional kwargs to pass to Dataset.map() for the
            preprocess stage (e.g., num_cpus, memory, concurrency).
        postprocess_map_kwargs: Optional kwargs to pass to Dataset.map() for the
            postprocess stage (e.g., num_cpus, memory, concurrency).
        telemetry_agent: An optional telemetry agent for collecting usage telemetry.

    Returns:
        The constructed processor.
    """
    ray.init(runtime_env=config.runtime_env, ignore_reinit_error=True)

    # Collected as (canonical_name, stage, stage_cfg) tuples in pipeline order; the
    # engine carries a None cfg. CPU-stage fusion is applied after all stages are built.
    built_stages = []

    # Prepare processor defaults for merging into stage configs
    trust_remote_code = config.engine_kwargs.get("trust_remote_code", False)
    processor_defaults = {
        "batch_size": config.batch_size,
        "concurrency": config.concurrency,
        "runtime_env": config.runtime_env,
        "model_source": config.model_source,
    }

    # Resolve and build PrepareImageStage if enabled
    image_stage_cfg = resolve_stage_config(
        config.prepare_image_stage,
        PrepareImageStageConfig,
        processor_defaults,
    )

    # Resolve and build PrepareMultimodalStage if enabled
    prepare_multimodal_stage_cfg = resolve_stage_config(
        config.prepare_multimodal_stage,
        PrepareMultimodalStageConfig,
        processor_defaults,
    )

    if image_stage_cfg.enabled and prepare_multimodal_stage_cfg.enabled:
        raise ValueError(
            "Cannot enable both 'prepare_image_stage' and 'prepare_multimodal_stage' "
            "simultaneously. The 'prepare_multimodal_stage' handles image processing "
            "along with other multimodal inputs. Please disable one of them."
        )

    if image_stage_cfg.enabled:
        built_stages.append(
            (
                "prepare_image",
                PrepareImageStage(
                    map_batches_kwargs=build_cpu_stage_map_kwargs(image_stage_cfg),
                ),
                image_stage_cfg,
            )
        )

    if prepare_multimodal_stage_cfg.enabled:
        base_model_config_kwargs = (
            prepare_multimodal_stage_cfg.model_config_kwargs or {}
        )
        # Respect the model source from the processor
        model_config_kwargs = {
            **base_model_config_kwargs,
            "model": processor_defaults.get("model_source"),
        }
        built_stages.append(
            (
                "prepare_multimodal",
                PrepareMultimodalStage(
                    fn_constructor_kwargs=dict(
                        model_config_kwargs=model_config_kwargs,
                        chat_template_content_format=prepare_multimodal_stage_cfg.chat_template_content_format,
                        apply_sys_msg_formatting=prepare_multimodal_stage_cfg.apply_sys_msg_formatting,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(
                        prepare_multimodal_stage_cfg
                    ),
                ),
                prepare_multimodal_stage_cfg,
            )
        )

    # Resolve and build ChatTemplateStage if enabled
    chat_template_stage_cfg = resolve_stage_config(
        getattr(config, "chat_template_stage", config.apply_chat_template),
        ChatTemplateStageConfig,
        processor_defaults,
    )
    if chat_template_stage_cfg.enabled:
        built_stages.append(
            (
                "chat_template",
                ChatTemplateStage(
                    fn_constructor_kwargs=dict(
                        model=chat_template_stage_cfg.model_source,
                        chat_template=get_value_or_fallback(
                            chat_template_stage_cfg.chat_template, config.chat_template
                        ),
                        chat_template_kwargs=get_value_or_fallback(
                            chat_template_stage_cfg.chat_template_kwargs,
                            chat_template_kwargs,
                        ),
                        trust_remote_code=trust_remote_code,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(
                        chat_template_stage_cfg
                    ),
                ),
                chat_template_stage_cfg,
            )
        )

    # Resolve and build TokenizeStage if enabled
    tokenize_stage_cfg = resolve_stage_config(
        getattr(config, "tokenize_stage", config.tokenize),
        TokenizerStageConfig,
        processor_defaults,
    )
    if tokenize_stage_cfg.enabled:
        built_stages.append(
            (
                "tokenize",
                TokenizeStage(
                    fn_constructor_kwargs=dict(
                        model=tokenize_stage_cfg.model_source,
                        trust_remote_code=trust_remote_code,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(tokenize_stage_cfg),
                ),
                tokenize_stage_cfg,
            )
        )

    # Core stage -- the vLLM engine.

    built_stages.append(
        (
            ENGINE_STAGE_NAME,
            vLLMEngineStage(
                fn_constructor_kwargs=dict(
                    batch_size=config.batch_size,
                    max_concurrent_batches=config.max_concurrent_batches,
                    model=config.model_source,
                    engine_kwargs=config.engine_kwargs,
                    task_type=config.task_type,
                    max_pending_requests=config.max_pending_requests,
                    dynamic_lora_loading_path=config.dynamic_lora_loading_path,
                    placement_group_config=config.placement_group_config,
                    should_continue_on_error=config.should_continue_on_error,
                    log_engine_metrics=config.log_engine_metrics,
                ),
                map_batches_kwargs=dict(
                    zero_copy_batch=True,
                    # The number of running replicas. This is a deprecated field, but
                    # we need to set `max_tasks_in_flight_per_actor` through `compute`,
                    # which initiates enough many overlapping UDF calls per actor, to
                    # saturate `max_concurrency`.
                    compute=ray.data.ActorPoolStrategy(
                        **config.get_concurrency(autoscaling_enabled=True),
                        max_tasks_in_flight_per_actor=config.experimental.get(
                            "max_tasks_in_flight_per_actor", DEFAULT_MAX_TASKS_IN_FLIGHT
                        ),
                    ),
                    # The number of running batches "per actor" in Ray Core level.
                    # This is used to make sure we overlap batches to avoid the tail
                    # latency of each batch.
                    max_concurrency=config.max_concurrent_batches,
                    accelerator_type=config.accelerator_type,
                    runtime_env=config.runtime_env,
                ),
            ),
            None,
        )
    )

    # Resolve and build DetokenizeStage if enabled
    detokenize_stage_cfg = resolve_stage_config(
        getattr(config, "detokenize_stage", config.detokenize),
        DetokenizeStageConfig,
        processor_defaults,
    )
    if detokenize_stage_cfg.enabled:
        built_stages.append(
            (
                "detokenize",
                DetokenizeStage(
                    fn_constructor_kwargs=dict(
                        model=detokenize_stage_cfg.model_source,
                        trust_remote_code=trust_remote_code,
                    ),
                    map_batches_kwargs=build_cpu_stage_map_kwargs(
                        detokenize_stage_cfg
                    ),
                ),
                detokenize_stage_cfg,
            )
        )

    # We download the config files here so that we can report the underlying architecture to the telemetry system.
    # This should be a lightweight operation.
    # Use EXCLUDE_SAFETENSORS for streaming formats or trust_remote_code models,
    # since custom model architectures require Python config files to be downloaded.
    if config.engine_kwargs.get(
        "load_format", None
    ) in STREAMING_LOAD_FORMATS or config.engine_kwargs.get("trust_remote_code", False):
        download_model_mode = NodeModelDownloadable.EXCLUDE_SAFETENSORS
    else:
        download_model_mode = NodeModelDownloadable.TOKENIZER_ONLY
    model_path = download_model_files(
        model_id=config.model_source,
        mirror_config=None,
        download_model=download_model_mode,
        download_extra_files=False,
    )

    try:
        hf_config = transformers.AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=config.engine_kwargs.get("trust_remote_code", False),
        )
    except Exception:
        # Failed to retrieve HuggingFace config for telemetry purposes.
        # This is non-fatal: we fall back to DEFAULT_MODEL_ARCHITECTURE for telemetry.
        # The actual model loading happens later in vLLM, which may support models
        # that aren't available via HuggingFace's AutoConfig.
        logger.warning(
            f"Failed to retrieve HuggingFace config for {config.model_source}"
        )
        hf_config = None

    architectures = getattr(hf_config, "architectures", [])
    architecture = architectures[0] if architectures else DEFAULT_MODEL_ARCHITECTURE

    telemetry_agent = get_or_create_telemetry_agent()
    telemetry_agent.push_telemetry_report(
        BatchModelTelemetry(
            processor_config_name=type(config).__name__,
            model_architecture=architecture,
            batch_size=config.batch_size,
            accelerator_type=config.accelerator_type or DEFAULT_GPU_TYPE,
            concurrency=config.concurrency,
            task_type=config.task_type,
            pipeline_parallel_size=config.engine_kwargs.get(
                "pipeline_parallel_size", 1
            ),
            tensor_parallel_size=config.engine_kwargs.get("tensor_parallel_size", 1),
        )
    )

    # Apply CPU-stage fusion (fuse_cpu_stages / stage_groups) over the enabled stages.
    stages = _apply_stage_fusion(config, built_stages)

    processor = Processor(
        config,
        stages,
        preprocess=preprocess,
        postprocess=postprocess,
        preprocess_map_kwargs=preprocess_map_kwargs,
        postprocess_map_kwargs=postprocess_map_kwargs,
    )
    return processor


ProcessorBuilder.register(vLLMEngineProcessorConfig, build_vllm_engine_processor)
