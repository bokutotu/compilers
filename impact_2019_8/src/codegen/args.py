"""ASTを走査して関数引数を収集する関数群."""

from __future__ import annotations

from collections.abc import Mapping

from ast_types import Body, Call, Expr, ForLoop, Id, User
from ir_types import MatrixOp


def require_id(expr: Expr, label: str) -> str:
    """式がIdであることを検証し、名前を返す."""
    if not isinstance(expr, Id):
        raise ValueError(f"{label} must be an Id")
    return expr.name


def split_call_args(
    op: MatrixOp, args: list[Expr]
) -> tuple[tuple[str, str, str] | None, list[Expr]]:
    """Call argsをマトリックス名とインデックスに分割する."""
    rank = len(op.out.dims)
    operand_count = 3
    if len(args) == rank:
        return None, args
    if len(args) == operand_count + rank:
        left_name = require_id(args[0], "left matrix")
        right_name = require_id(args[1], "right matrix")
        out_name = require_id(args[2], "out matrix")
        return (left_name, right_name, out_name), args[operand_count:]
    raise ValueError(
        "Call args must be (indices) or "
        "(left, right, out, indices) for statement "
        f"'{op.name}', got {len(args)}"
    )


def collect_function_args(
    ast: ForLoop, domain_exprs: Mapping[str, MatrixOp]
) -> list[str]:
    """ASTを走査して関数の引数（マトリックス名）を収集する."""
    names: list[str] = []
    fallback: list[str] = []

    def add_unique(target: list[str], name: str) -> None:
        if name not in target:
            target.append(name)

    def handle_call(call: Call) -> None:
        if not call.args:
            return
        stmt_id = call.args[0]
        if not isinstance(stmt_id, Id):
            return
        op = domain_exprs.get(stmt_id.name)
        if op is None:
            return
        matrix_names, _ = split_call_args(op, call.args[1:])
        if matrix_names is not None:
            for name in matrix_names:
                add_unique(names, name)
        else:
            add_unique(fallback, op.left.name)
            add_unique(fallback, op.right.name)
            add_unique(fallback, op.out.name)

    def walk(body: Body) -> None:
        if isinstance(body, User):
            handle_call(body.expr)
        elif isinstance(body, ForLoop):
            walk(body.body)

    walk(ast.body)
    return names or fallback
