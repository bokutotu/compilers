from __future__ import annotations

import islpy as isl

from ir_types import AffineConstraint, Axis, Domain, PrimFunc


def _axis_dims(a: Axis) -> str:
    return f"{a.lower} <= {a.name} < {a.extent}"


def _collect_params_from_axes(axes: tuple[Axis, ...]) -> list[str]:
    params: list[str] = []
    for axis in axes:
        if isinstance(axis.extent, str) and axis.extent not in params:
            params.append(axis.extent)
        if isinstance(axis.lower, str) and axis.lower not in params:
            params.append(axis.lower)
    return params


def _collect_params_from_constraints(
    constraints: tuple[AffineConstraint, ...],
) -> list[str]:
    params: list[str] = []
    for constraint in constraints:
        for p in constraint.collect_params():
            if p not in params:
                params.append(p)
    return params


def _collect_params(domain: Domain) -> list[str]:
    """軸とアフィン制約からシンボリックパラメータを収集"""
    params = _collect_params_from_axes(domain.axis)
    for p in _collect_params_from_constraints(domain.constraints):
        if p not in params:
            params.append(p)
    return params


def _make_param_str(params: list[str]) -> str:
    if not params:
        return ""
    return f"[{', '.join(params)}] -> "


def _build_constraints_str(domain: Domain) -> str:
    """軸の範囲とアフィン制約を結合した制約文字列を生成"""
    parts: list[str] = []
    # 軸の範囲制約
    for axis in domain.axis:
        parts.append(_axis_dims(axis))
    # 追加のアフィン制約
    for constraint in domain.constraints:
        parts.append(constraint.to_isl())
    return " and ".join(parts)


def build_domain(func: PrimFunc, ctx: isl.Context | None = None) -> isl.UnionSet:
    ctx = ctx or isl.Context()
    stmt_name = func.compute.name
    domain = func.compute.domain
    params = _collect_params(domain)
    param_str = _make_param_str(params)
    index_names = ", ".join([a.name for a in domain.axis])
    constraints = _build_constraints_str(domain)
    return isl.UnionSet(
        f"{param_str}{{ {stmt_name}[{index_names}] : {constraints} }}", ctx
    )


def build_schedule(func: PrimFunc, ctx: isl.Context | None = None) -> isl.UnionMap:
    ctx = ctx or isl.Context()
    stmt_name = func.compute.name
    domain = func.compute.domain
    loop_order = func.schedule.loop_order
    params = _collect_params(domain)
    param_str = _make_param_str(params)
    index_names = ", ".join([a.name for a in domain.axis])
    schedule_order = ", ".join(loop_order)
    constraints = _build_constraints_str(domain)
    return isl.UnionMap(
        f"{param_str}{{ {stmt_name}[{index_names}] -> [{schedule_order}] : {constraints} }}",  # noqa: E501
        ctx,
    )


def build_domain_and_schedule(
    func: PrimFunc, ctx: isl.Context | None = None
) -> tuple[isl.UnionSet, isl.UnionMap]:
    ctx = ctx or isl.Context()
    domain = build_domain(func, ctx)
    schedule = build_schedule(func, ctx)
    return domain, schedule
