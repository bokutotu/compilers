"""MatrixOpからC言語の代入文を生成する関数群."""

from __future__ import annotations

from collections.abc import Mapping

from ast_types import Call, Id
from ir_types import MatrixOp, MatrixPtr

from .args import split_call_args
from .expr import generate_index_expr

OP_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def format_matrix_access(matrix: MatrixPtr, indices: list[str]) -> str:
    """マトリックスアクセス式を生成する."""
    if len(indices) != len(matrix.dims):
        raise ValueError(
            f"{matrix.name} expects {len(matrix.dims)} indices, got {len(indices)}"
        )
    suffix = "".join(f"[{index}]" for index in indices)
    return f"{matrix.name}{suffix}"


def _validate_indices(op: MatrixOp, indices: list[str]) -> None:
    """インデックス数を検証する."""
    expected = len(op.out.dims)
    if len(indices) != expected:
        raise ValueError(
            f"Index rank mismatch: expected {expected}, got {len(indices)}"
        )


def _resolve_matrices(
    op: MatrixOp, matrix_names: tuple[str, str, str] | None
) -> tuple[MatrixPtr, MatrixPtr, MatrixPtr]:
    """マトリックス名を解決する."""
    if matrix_names is None:
        return op.left, op.right, op.out
    left_name, right_name, out_name = matrix_names
    return (
        MatrixPtr(name=left_name, dims=op.left.dims),
        MatrixPtr(name=right_name, dims=op.right.dims),
        MatrixPtr(name=out_name, dims=op.out.dims),
    )


def generate_op_assignment(
    op: MatrixOp,
    indices: list[str],
    matrix_names: tuple[str, str, str] | None,
) -> str:
    """MatrixOpを代入式に変換する."""
    if op.op not in OP_MAP:
        raise ValueError(f"Unsupported op: {op.op}")
    _validate_indices(op, indices)
    left, right, out = _resolve_matrices(op, matrix_names)
    out_ref = format_matrix_access(out, indices)
    left_ref = format_matrix_access(left, indices)
    right_ref = format_matrix_access(right, indices)
    return f"{out_ref} = {left_ref} {OP_MAP[op.op]} {right_ref}"


def generate_user_stmt(call: Call, domain_exprs: Mapping[str, MatrixOp]) -> str:
    """ユーザー文をIRに基づいてC文に変換する."""
    if not call.args:
        raise ValueError("User call must have a statement id")
    stmt_id = call.args[0]
    if not isinstance(stmt_id, Id):
        raise ValueError("User call must start with an Id")
    op = domain_exprs.get(stmt_id.name)
    if op is None:
        raise ValueError(f"Unknown statement id: {stmt_id.name}")
    matrix_names, index_exprs = split_call_args(op, call.args[1:])
    indices = [generate_index_expr(arg) for arg in index_exprs]
    return f"{generate_op_assignment(op, indices, matrix_names)};"
