from __future__ import annotations

import re
from collections.abc import Sequence

import islpy as isl

from ast_types import Block, ForLoop, Guard
from ir_to_isl import (
    build_domain,
    build_read_access,
    build_write_access,
    compute_raw_dependence,
    compute_war_dependence,
    compute_waw_dependence,
    constraint_to_isl,
)
from ir_types import Compute, PrimFunc, Schedule, Tensor
from isl_ast import build_ast_from_domain_and_schedule
from isl_ast_converter import convert_ast_node


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not cleaned:
        return "anon"
    if cleaned[0].isdigit():
        return f"f_{cleaned}"
    return cleaned


def _build_header(compute: Compute) -> tuple[str, str, str]:
    domain = compute.domain
    param_str = f"[{', '.join(domain.params)}]" if domain.params else "[]"

    iter_names = [it.name for it in domain.iterators]
    tuple_str = f"{compute.name}[{', '.join(iter_names)}]"

    if domain.constraints:
        const_parts = [constraint_to_isl(c) for c in domain.constraints]
        const_str = " and ".join(const_parts)
    else:
        const_str = "1 = 1"

    return param_str, tuple_str, const_str


def _tag_primfunc(func: PrimFunc, func_idx: int) -> PrimFunc:
    prefix = f"f{func_idx}_{_sanitize_name(func.name)}"
    tagged_computes = tuple(
        Compute(
            name=f"{prefix}__{_sanitize_name(compute.name)}",
            domain=compute.domain,
            body=compute.body,
        )
        for compute in func.computes
    )
    return PrimFunc(
        name=func.name,
        params=func.params,
        computes=tagged_computes,
        schedule=func.schedule,
    )


def _merge_params(funcs: Sequence[PrimFunc]) -> tuple[Tensor, ...]:
    seen: dict[str, Tensor] = {}
    ordered: list[str] = []
    for func in funcs:
        for tensor in func.params:
            existing = seen.get(tensor.name)
            if existing is None:
                seen[tensor.name] = tensor
                ordered.append(tensor.name)
                continue
            if existing.shape != tensor.shape or existing.dtype != tensor.dtype:
                raise ValueError(
                    f"Tensor param conflict for '{tensor.name}': "
                    f"{existing.shape}/{existing.dtype} vs {tensor.shape}/{tensor.dtype}"
                )
    return tuple(seen[name] for name in ordered)


def _collect_param_names(funcs: Sequence[PrimFunc]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for func in funcs:
        for compute in func.computes:
            for name in compute.domain.params:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def _build_param_space(ctx: isl.Context, params: list[str]) -> isl.Space:
    if not params:
        return isl.Set("{ : }", ctx).get_space()
    param_str = ", ".join(params)
    return isl.Set(f"[{param_str}] -> {{ : }}", ctx).get_space()


def _build_union_domain(
    funcs: Sequence[PrimFunc],
    ctx: isl.Context,
    param_space: isl.Space,
) -> isl.UnionSet:
    u_set = isl.UnionSet("{ }", ctx)
    for func in funcs:
        domain = build_domain(func, ctx).align_params(param_space)
        u_set = u_set.union(domain)
    return u_set


def _build_union_accesses(
    funcs: Sequence[PrimFunc],
    ctx: isl.Context,
    param_space: isl.Space,
) -> tuple[isl.UnionMap, isl.UnionMap]:
    write_access = isl.UnionMap("{ }", ctx)
    read_access = isl.UnionMap("{ }", ctx)
    for func in funcs:
        write_access = write_access.union(
            build_write_access(func, ctx).align_params(param_space)
        )
        read_access = read_access.union(
            build_read_access(func, ctx).align_params(param_space)
        )
    return write_access, read_access


def _build_base_schedule(
    funcs: Sequence[PrimFunc],
    ctx: isl.Context,
    param_space: isl.Space,
    max_loop_depth: int,
) -> isl.UnionMap:
    schedule = isl.UnionMap("{ }", ctx)
    for func_idx, func in enumerate(funcs):
        loop_order = func.schedule.loop_order
        for stmt_id, compute in enumerate(func.computes):
            param_str, src_tuple_str, const_str = _build_header(compute)

            iter_names = {it.name for it in compute.domain.iterators}
            sched_iters = [name for name in loop_order if name in iter_names]

            if len(sched_iters) < max_loop_depth:
                sched_iters.extend(["0"] * (max_loop_depth - len(sched_iters)))

            dst_dims = [str(func_idx)] + sched_iters + [str(stmt_id)]
            dst_tuple_str = f"[{', '.join(dst_dims)}]"
            isl_str = (
                f"{param_str} -> {{ {src_tuple_str} -> {dst_tuple_str} : {const_str} }}"
            )
            schedule = schedule.union(isl.UnionMap(isl_str, ctx))

    return schedule.align_params(param_space)


def _build_fused_schedule(
    funcs: Sequence[PrimFunc],
    ctx: isl.Context,
    param_space: isl.Space,
    max_loop_depth: int,
) -> isl.UnionMap:
    schedule = isl.UnionMap("{ }", ctx)
    stmt_counter = 0
    for func_idx, func in enumerate(funcs):
        loop_order = func.schedule.loop_order
        for compute in func.computes:
            param_str, src_tuple_str, const_str = _build_header(compute)

            iter_names = {it.name for it in compute.domain.iterators}
            sched_iters = [name for name in loop_order if name in iter_names]

            if len(sched_iters) < max_loop_depth:
                sched_iters.extend(["0"] * (max_loop_depth - len(sched_iters)))

            dst_dims = sched_iters + [str(func_idx), str(stmt_counter)]
            dst_tuple_str = f"[{', '.join(dst_dims)}]"
            isl_str = (
                f"{param_str} -> {{ {src_tuple_str} -> {dst_tuple_str} : {const_str} }}"
            )
            schedule = schedule.union(isl.UnionMap(isl_str, ctx))
            stmt_counter += 1

    return schedule.align_params(param_space)


def _compute_optimized_schedule(
    domain: isl.UnionSet,
    dependences: isl.UnionMap,
) -> isl.Schedule:
    sc = isl.ScheduleConstraints.on_domain(domain)
    sc = sc.set_validity(dependences)
    sc = sc.set_coincidence(dependences)
    sc = sc.set_proximity(dependences)
    return sc.compute_schedule()


def _schedule_respects_deps(
    schedule: isl.UnionMap,
    dependences: isl.UnionMap,
) -> bool:
    try:
        before = schedule.lex_lt_union_map(schedule)
        return dependences.is_subset(before)
    except isl.Error:
        return False


def _unwrap_single_loop_root(node: Block | ForLoop | Guard) -> ForLoop | None:
    if isinstance(node, ForLoop):
        return node
    if isinstance(node, Guard):
        return _unwrap_single_loop_root(node.then)
    if isinstance(node, Block):
        if len(node.stmts) != 1:
            return None
        return _unwrap_single_loop_root(node.stmts[0])
    return None


def _ensure_single_loop_nest(ast_root: Block | ForLoop | Guard) -> None:
    if _unwrap_single_loop_root(ast_root) is None:
        raise ValueError("Cannot fuse PrimFunc list into a single loop nest")


def build_fused_ast(
    funcs: Sequence[PrimFunc],
) -> tuple[Block | ForLoop | Guard, PrimFunc]:
    if not funcs:
        raise ValueError("compile_fused() received an empty PrimFunc list")

    ctx = isl.Context()
    tagged_funcs = [_tag_primfunc(func, idx) for idx, func in enumerate(funcs)]

    param_names = _collect_param_names(tagged_funcs)
    param_space = _build_param_space(ctx, param_names)

    domain = _build_union_domain(tagged_funcs, ctx, param_space)

    max_loop_depth = max(
        (len(func.schedule.loop_order) for func in tagged_funcs), default=0
    )
    base_schedule = _build_base_schedule(
        tagged_funcs, ctx, param_space, max_loop_depth
    )

    write_access, read_access = _build_union_accesses(tagged_funcs, ctx, param_space)

    raw_dep = compute_raw_dependence(base_schedule, write_access, read_access)
    war_dep = compute_war_dependence(base_schedule, write_access, read_access)
    waw_dep = compute_waw_dependence(base_schedule, write_access)
    all_deps = raw_dep.union(war_dep).union(waw_dep)

    fused_schedule = _build_fused_schedule(
        tagged_funcs, ctx, param_space, max_loop_depth
    )
    if _schedule_respects_deps(fused_schedule, all_deps):
        schedule = fused_schedule
        ast = build_ast_from_domain_and_schedule(domain, schedule)
    else:
        schedule = _compute_optimized_schedule(domain, all_deps)
        context_set = isl.Set("{ : }", ctx)
        build = isl.AstBuild.from_context(context_set)
        ast = build.node_from_schedule(schedule)

    parsed_ast = convert_ast_node(ast)
    _ensure_single_loop_nest(parsed_ast)

    fused_name = "fused_" + "_".join(_sanitize_name(func.name) for func in funcs)
    fused_func = PrimFunc(
        name=fused_name,
        params=_merge_params(funcs),
        computes=tuple(
            compute for func in tagged_funcs for compute in func.computes
        ),
        schedule=Schedule(loop_order=()),
    )

    return parsed_ast, fused_func
