"""式をC言語の式に変換する関数群."""

from __future__ import annotations

from ast_types import BinOp, Call, Expr, Id, Val

OP_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "le": "<=",
    "lt": "<",
    "ge": ">=",
    "gt": ">",
    "eq": "==",
}


def generate_expr(expr: Expr) -> str:
    """式をC言語の式に変換する."""
    if isinstance(expr, Id):
        return expr.name
    elif isinstance(expr, Val):
        return str(expr.value)
    elif isinstance(expr, BinOp):
        left = generate_expr(expr.left)
        right = generate_expr(expr.right)
        op = OP_MAP.get(expr.op, expr.op)
        return f"({left} {op} {right})"
    elif isinstance(expr, Call):
        return _generate_call_as_sum(expr)
    else:
        raise ValueError(f"Unknown expression type: {type(expr)}")


def _generate_call_as_sum(call: Call) -> str:
    """Call（S関数）を足し算として生成する."""
    if not call.args:
        return "0"
    args_str = [generate_expr(arg) for arg in call.args]
    return " + ".join(args_str)


def generate_cond(cond: BinOp) -> str:
    """条件式を生成する."""
    op = OP_MAP.get(cond.op, cond.op)
    left = generate_expr(cond.left)
    right = generate_expr(cond.right)
    return f"{left} {op} {right}"


def generate_index_expr(expr: Expr) -> str:
    """インデックス式を生成する."""
    if isinstance(expr, Call):
        raise ValueError("Index expression must not be a Call")
    return generate_expr(expr)
