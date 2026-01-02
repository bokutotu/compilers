from __future__ import annotations

import islpy as isl

from ir_types_new import Axis, PrimFunc


def _axis_dims(a: Axis) -> str:
    return f"{a.lower} <= {a.name} < {a.extent}"


def _collect_params(axes: tuple[Axis, ...]) -> list[str]:
    """軸のextentとlowerからシンボリックパラメータを収集する"""
    params: list[str] = []
    for axis in axes:
        if isinstance(axis.extent, str) and axis.extent not in params:
            params.append(axis.extent)
        if isinstance(axis.lower, str) and axis.lower not in params:
            params.append(axis.lower)
    return params


def _make_param_str(params: list[str]) -> str:
    """ISL形式のパラメータ文字列を作成する"""
    if not params:
        return ""
    return f"[{', '.join(params)}] -> "


def build_domain(
    func: PrimFunc,
    ctx: isl.Context | None = None
) -> isl.UnionSet:
    ctx = ctx or isl.Context()
    stmt_name = func.compute.name
    domain_axes = func.compute.domain.axis
    params = _collect_params(domain_axes)
    param_str = _make_param_str(params)
    index_names = ", ".join([a.name for a in domain_axes])
    constraints = " and ".join([_axis_dims(a) for a in domain_axes])
    return isl.UnionSet(
        f"{param_str}{{ {stmt_name}[{index_names}] : {constraints} }}",
        ctx
    )


def build_schedule(
    func: PrimFunc,
    ctx: isl.Context | None = None
) -> isl.UnionMap:
    ctx = ctx or isl.Context()
    stmt_name = func.compute.name
    domain_axes = func.compute.domain.axis
    params = _collect_params(domain_axes)
    param_str = _make_param_str(params)
    index_names = ", ".join([a.name for a in domain_axes])
    constraints = " and ".join([_axis_dims(a) for a in domain_axes])
    return isl.UnionMap(
        f"{param_str}{{ {stmt_name}[{index_names}] -> [{index_names}] : {constraints} }}",
        ctx
    )


def build_domain_and_schedule(
    func: PrimFunc,
    ctx: isl.Context | None = None
) -> tuple[isl.UnionSet, isl.UnionMap]:
    domain = build_domain(func, ctx)
    schedule = build_schedule(func, ctx)
    return domain, schedule
