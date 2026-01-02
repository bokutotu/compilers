"""Build ISL domain and schedule from minimal IR."""

from __future__ import annotations

import islpy as isl

from ir_types import MatrixOp


def _param_prefix(*extents: int | str) -> str:
    params: list[str] = []
    for extent in extents:
        if isinstance(extent, str) and extent not in params:
            params.append(extent)
    if not params:
        return ""
    return f"[{', '.join(params)}] -> "


def _index_names(rank: int) -> list[str]:
    if rank < 1:
        raise ValueError("MatrixPtr.dims must not be empty")
    if rank == 1:
        return ["i"]
    if rank == 2:
        return ["i", "j"]
    return [f"i{idx}" for idx in range(rank)]


def _op_dims(op: MatrixOp) -> list[int | str]:
    return [dim.extent for dim in op.out.dims]


def build_domain(
    op: MatrixOp,
    ctx: isl.Context | None = None,
) -> isl.UnionSet:
    """Create an ISL domain for an element-wise op."""
    ctx = ctx or isl.Context()
    dims = _op_dims(op)
    indices = _index_names(len(dims))
    index_list = ", ".join(indices)
    constraints = " and ".join(
        f"0 <= {name} < {extent}" for name, extent in zip(indices, dims, strict=True)
    )
    prefix = _param_prefix(*dims)
    return isl.UnionSet(
        f"{prefix}{{ {op.name}[{index_list}] : {constraints} }}",
        ctx,
    )


def build_schedule(
    op: MatrixOp,
    ctx: isl.Context | None = None,
) -> isl.UnionMap:
    """Create an ISL schedule map for an element-wise op."""
    ctx = ctx or isl.Context()
    dims = _op_dims(op)
    indices = _index_names(len(dims))
    index_list = ", ".join(indices)
    constraints = " and ".join(
        f"0 <= {name} < {extent}" for name, extent in zip(indices, dims, strict=True)
    )
    prefix = _param_prefix(*dims)
    return isl.UnionMap(
        f"{prefix}{{ {op.name}[{index_list}] -> [{index_list}] : {constraints} }}",
        ctx,
    )


def build_domain_and_schedule(
    op: MatrixOp,
    ctx: isl.Context | None = None,
) -> tuple[isl.UnionSet, isl.UnionMap]:
    """Create domain and schedule sharing the same context."""
    ctx = ctx or isl.Context()
    domain = build_domain(op=op, ctx=ctx)
    schedule = build_schedule(op=op, ctx=ctx)
    return domain, schedule
