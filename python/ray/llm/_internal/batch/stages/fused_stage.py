"""Fused stage: run several CPU stages sequentially inside one actor pool.

Each Ray Data LLM stage reads and writes the same ``__data`` column, and
``StatefulStageUDF.__call__`` is a pure envelope transform: it pops ``__data``,
validates inputs, assigns ``IDX_IN_BATCH_COLUMN``, runs ``udf(rows)``, merges the
UDF outputs back by index, and re-emits ``{__data: rows}``. Because each member
stage's ``udf`` only reads the columns it needs and echoes ``IDX_IN_BATCH_COLUMN``,
we can fuse a contiguous run of CPU stages by chaining their ``udf`` methods inside
one ``FusedStageUDF.udf`` -- reusing the base ``__call__`` for the envelope handling.

The result deploys as a SINGLE ``map_batches`` actor pool that runs all member
stages per batch, instead of one actor pool per stage. This reduces CPU contention on
small GPU nodes by consolidating actor pools that would each otherwise reserve a CPU.
"""

from typing import Any, AsyncIterator, Dict, List, Type

from ray.llm._internal.batch.stages.base import StatefulStage, StatefulStageUDF


class FusedStageUDF(StatefulStageUDF):
    """A UDF that runs a sequence of member stage UDFs sequentially per batch.

    The inherited ``StatefulStageUDF.__call__`` handles the ``__data`` envelope,
    input validation (against the *first* member's required keys), error-row
    handling, ``IDX_IN_BATCH_COLUMN`` assignment, and the final merge. This class
    only overrides ``udf`` to thread the rows through each member's ``udf``.

    Member contract: each member UDF must **echo** ``IDX_IN_BATCH_COLUMN`` from its
    input rows in every output row -- outputs may arrive out of order and are
    reconciled by that index. Members must not re-derive the row position; an output
    whose index is missing or unknown raises a clear error rather than mismapping.
    """

    def __init__(
        self,
        data_column: str,
        expected_input_keys: List[str],
        fused_members: List[Dict[str, Any]],
    ):
        """Initialize the fused UDF and construct every member UDF.

        Args:
            data_column: The internal data column name (shared by all members).
            expected_input_keys: The first member's required input keys (validated by
                the base ``__call__``); later members' inputs are produced internally.
            fused_members: Ordered specs, one per member stage. Each is a dict with
                ``fn`` (the member ``StatefulStageUDF`` class),
                ``fn_constructor_kwargs`` (the member's constructor kwargs), and
                ``expected_input_keys`` (the member's own required input keys).
        """
        super().__init__(data_column, expected_input_keys)
        self.member_udfs: List[StatefulStageUDF] = [
            spec["fn"](
                data_column=data_column,
                expected_input_keys=spec["expected_input_keys"],
                **spec["fn_constructor_kwargs"],
            )
            for spec in fused_members
        ]

    async def udf(self, rows: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """Run rows through each member UDF in order, threading outputs forward.

        Args:
            rows: The input rows (each already carries ``IDX_IN_BATCH_COLUMN``).

        Yields:
            One dict per row containing ``IDX_IN_BATCH_COLUMN`` and the union of the
            columns produced by all member stages.
        """
        # Work on shallow copies so each member sees the rows updated by the previous
        # member, while we accumulate only the newly produced columns to yield. Member
        # UDFs may emit out of order, so we always reconcile by IDX_IN_BATCH_COLUMN.
        working: Dict[Any, Dict[str, Any]] = {
            row[self.IDX_IN_BATCH_COLUMN]: dict(row) for row in rows
        }
        produced: Dict[Any, Dict[str, Any]] = {
            row[self.IDX_IN_BATCH_COLUMN]: {} for row in rows
        }

        for member_udf in self.member_udfs:
            current_rows = list(working.values())
            async for output in member_udf.udf(current_rows):
                if self.IDX_IN_BATCH_COLUMN not in output:
                    raise ValueError(
                        f"Fused member {type(member_udf).__name__} returned an output "
                        f"row without the {self.IDX_IN_BATCH_COLUMN!r} column; fused "
                        "member UDFs must echo the row index from their inputs."
                    )
                idx = output[self.IDX_IN_BATCH_COLUMN]
                if idx not in working:
                    raise ValueError(
                        f"Fused member {type(member_udf).__name__} returned an unknown "
                        f"row index {idx!r} not present in the input batch; fused member "
                        "UDFs must echo IDX_IN_BATCH_COLUMN rather than re-deriving the "
                        "row position."
                    )
                for key, value in output.items():
                    if key == self.IDX_IN_BATCH_COLUMN:
                        continue
                    working[idx][key] = value
                    produced[idx][key] = value

        for idx, new_columns in produced.items():
            yield {self.IDX_IN_BATCH_COLUMN: idx, **new_columns}


class FusedStage(StatefulStage):
    """A stage that fuses a contiguous run of CPU stages into one actor pool.

    The member stages are passed to ``FusedStageUDF`` via
    ``fn_constructor_kwargs["fused_members"]``. ``member_stage_names`` is kept only for
    a descriptive, deterministic stage name in ``Processor.list_stage_names()``.
    """

    fn: Type[StatefulStageUDF] = FusedStageUDF

    # Display-only metadata (not passed to the UDF constructor).
    member_stage_names: List[str] = []
    # The first member's required/optional input keys, surfaced so that
    # ``Processor.log_input_column_names`` reports the columns a user must provide.
    required_input_keys: Dict[str, str] = {}
    optional_input_keys: Dict[str, str] = {}

    def get_stage_name(self) -> str:
        return "FusedStage(" + "+".join(self.member_stage_names) + ")"

    def get_required_input_keys(self) -> Dict[str, str]:
        return self.required_input_keys

    def get_optional_input_keys(self) -> Dict[str, str]:
        return self.optional_input_keys
