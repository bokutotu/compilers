"""ISLを使用したAST生成モジュール."""

import islpy as isl


def generate_simple_loop_ast(n: int = 10) -> isl.AstNode:
    """1重ループのASTを生成する.

    Args:
        n: ループの上限値

    Returns:
        生成されたASTノード
    """
    ctx = isl.Context()

    # ドメインを定義: { S[i] : 0 <= i < n }
    domain = isl.UnionSet(f"{{ S[i] : 0 <= i < {n} }}", ctx)

    # スケジュールを定義: S[i] -> [i]
    schedule = isl.UnionMap(f"{{ S[i] -> [i] : 0 <= i < {n} }}", ctx)

    # AST生成
    build = isl.AstBuild.alloc(ctx)
    ast = build.node_from_schedule_map(schedule)

    return ast
