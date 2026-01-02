"""ComputeからC言語の代入文を生成する関数群."""

from __future__ import annotations

from src.ast_types import Call
from src.ir_types import (
    BinaryOp,
    BinOpKind,
    Compute,
    Const,
    Expr,
    Index,
    Load,
    ReduceStore,
    Store,
    Tensor,
)

from .expr import generate_index_expr

_BIN_OP_SYMBOL: dict[BinOpKind, str] = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}

_BIN_OP_PRECEDENCE: dict[BinOpKind, int] = {
    "add": 1,
    "sub": 1,
    "mul": 2,
    "div": 2,
}

_UPDATE_OP: dict[str, str] = {
    "add": "+=",
    "sub": "-=",
    "mul": "*=",
    "div": "/=",
}


def _format_tensor_access(tensor: Tensor, indices: list[str]) -> str:
    if len(indices) != len(tensor.shape):
        raise ValueError(
            f"Index rank mismatch for {tensor.name}: "
            f"got {len(indices)} indices, expected {len(tensor.shape)}"
        )

    if not indices:
        return tensor.name
    if len(indices) == 1:
        return f"{tensor.name}[{indices[0]}]"

    offset = indices[0]
    for dim in range(1, len(indices)):
        extent = tensor.shape[dim]
        offset = f"({offset}*{extent} + {indices[dim]})"
    return f"{tensor.name}[{offset}]"


def _resolve_indices(index: Index, axis_to_var: dict[str, str]) -> list[str]:
    return [axis_to_var.get(axis, axis) for axis in index]


def _needs_parens(
    parent_op: BinOpKind, child_op: BinOpKind, child_is_right: bool
) -> bool:
    parent_prec = _BIN_OP_PRECEDENCE[parent_op]
    child_prec = _BIN_OP_PRECEDENCE[child_op]

    if child_prec < parent_prec:
        return True
    if child_prec > parent_prec:
        return False
    if not child_is_right:
        return False

    if parent_op in ("sub", "div"):
        return True
    return bool(parent_op == "mul" and child_op == "div")


def _generate_ir_expr(
    expr: Expr,
    axis_to_var: dict[str, str],
    parent_op: BinOpKind | None = None,
    child_is_right: bool = False,
) -> str:
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Load):
        indices = _resolve_indices(expr.index, axis_to_var)
        return _format_tensor_access(expr.tensor, indices)
    if isinstance(expr, BinaryOp):
        left = _generate_ir_expr(expr.left, axis_to_var, expr.op, False)
        right = _generate_ir_expr(expr.right, axis_to_var, expr.op, True)
        rendered = f"{left} {_BIN_OP_SYMBOL[expr.op]} {right}"
        if parent_op is not None and _needs_parens(parent_op, expr.op, child_is_right):
            return f"({rendered})"
        return rendered

    raise ValueError(f"Unknown Expr type: {type(expr)}")


def _generate_reduction_init_cond(
    compute: Compute,
    target_index: Index,
    axis_to_var: dict[str, str],
) -> str:
    reduce_axes = [ax for ax in compute.domain.axis if ax.kind == "reduce"]
    if not reduce_axes:
        reduce_axes = [ax for ax in compute.domain.axis if ax.name not in target_index]
    if not reduce_axes:
        return "1"
    parts = []
    for ax in reduce_axes:
        var = axis_to_var.get(ax.name)
        if var is None:
            raise ValueError(f"Missing loop variable for axis '{ax.name}'")
        parts.append(f"{var} == {ax.lower}")
    return " && ".join(parts)


def generate_user_stmt(call: Call, compute: Compute) -> str:
    num_indices = len(compute.domain.axis)
    index_exprs = call.args[-num_indices:] if call.args and num_indices > 0 else []
    indices = [generate_index_expr(arg) for arg in index_exprs]

    axis_to_var = {axis.name: indices[i] for i, axis in enumerate(compute.domain.axis)}

    stmt = compute.stmt
    if isinstance(stmt, Store):
        target_ref = _format_tensor_access(
            stmt.target, _resolve_indices(stmt.index, axis_to_var)
        )
        value = _generate_ir_expr(stmt.value, axis_to_var)
        return f"{target_ref} = {value};"

    if isinstance(stmt, ReduceStore):
        target_ref = _format_tensor_access(
            stmt.target, _resolve_indices(stmt.index, axis_to_var)
        )
        value = _generate_ir_expr(stmt.value, axis_to_var)

        lines: list[str] = []
        if stmt.init is not None:
            cond = _generate_reduction_init_cond(compute, stmt.index, axis_to_var)
            init_value = _generate_ir_expr(stmt.init, axis_to_var)
            lines.append(f"if ({cond}) {target_ref} = {init_value};")

        if stmt.op in _UPDATE_OP:
            lines.append(f"{target_ref} {_UPDATE_OP[stmt.op]} {value};")
            return "\n".join(lines)

        if stmt.op == "max":
            lines.append(
                f"{target_ref} = ({target_ref} > {value}) ? {target_ref} : {value};"
            )
            return "\n".join(lines)

        if stmt.op == "min":
            lines.append(
                f"{target_ref} = ({target_ref} < {value}) ? {target_ref} : {value};"
            )
            return "\n".join(lines)

        raise ValueError(f"Unsupported reduce op: {stmt.op}")

    raise ValueError(f"Unknown Stmt type: {type(stmt)}")
