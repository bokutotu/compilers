"""ISLを使用したAST生成モジュール."""

import islpy as isl


def build_ast_from_domain_and_schedule(
    domain: isl.UnionSet,
    schedule: isl.UnionMap,
) -> isl.AstNode:
    """ドメインとスケジュールからASTを生成する."""
    ctx = domain.get_ctx()
    schedule = schedule.intersect_domain(domain)
    build = isl.AstBuild.alloc(ctx)
    return build.node_from_schedule_map(schedule)
