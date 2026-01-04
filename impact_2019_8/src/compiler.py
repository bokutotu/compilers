from __future__ import annotations

import islpy as isl

from codegen import isl_ast_to_c
from ir_to_isl import build_domain, build_schedule
from ir_types import PrimFunc
from isl_ast import build_ast_from_domain_and_schedule
from isl_ast_converter import convert_ast_node
from optimization_types import Tile
from optimize import apply_tiling_to_schedule, compute_optimized_schedule


def compile(
    func: PrimFunc,
    schedule: isl.UnionMap | isl.Schedule | None = None,
    optimize: bool = False,
    tiles: list[Tile] | None = None,
) -> str:
    if schedule is None:
        if optimize:
            schedule = compute_optimized_schedule(func)
            if tiles:
                schedule = apply_tiling_to_schedule(schedule, tiles)
        else:
            ctx = isl.Context()
            isl_domain = build_domain(func, ctx)
            isl_schedule = build_schedule(func, ctx)
            ast = build_ast_from_domain_and_schedule(isl_domain, isl_schedule)
            parsed_ast = convert_ast_node(ast)
            return isl_ast_to_c(parsed_ast, func)

    if isinstance(schedule, isl.Schedule):
        ctx = schedule.get_ctx()
        context_set = isl.Set("{ : }", ctx)
        build = isl.AstBuild.from_context(context_set)
        ast = build.node_from_schedule(schedule)
    else:
        ctx = schedule.get_ctx()
        isl_domain = build_domain(func, ctx)
        ast = build_ast_from_domain_and_schedule(isl_domain, schedule)

    parsed_ast = convert_ast_node(ast)
    return isl_ast_to_c(parsed_ast, func)
