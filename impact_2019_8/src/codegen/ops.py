"""ComputeからC言語の代入文を生成する関数群."""

from __future__ import annotations

from collections.abc import Mapping

from ast_types import Call, Id
from ir_types import Compute, Index, Tensor

from .args import split_call_args
from .expr import generate_index_expr

OP_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def format_tensor_access(tensor: Tensor, indices: list[str]) -> str:
    """テンソルアクセス式を生成する."""
    if len(indices) != len(tensor.shape):
        raise ValueError(
            f"{tensor.name} expects {len(tensor.shape)} indices, got {len(indices)}"
        )
    suffix = "".join(f"[{index}]" for index in indices)
    return f"{tensor.name}{suffix}"


def _resolve_tensor_indices(
    tensor_index: Index, axis_to_var: dict[str, str]
) -> list[str]:
    """テンソルのindexを解決する."""
    return [axis_to_var[axis] for axis in tensor_index]


def _resolve_tensors(
    op: Compute, tensor_names: tuple[str, str, str] | None
) -> tuple[Tensor, Tensor, Tensor]:
    """テンソル名を解決する."""
    if tensor_names is None:
        return op.a, op.b, op.out
    a_name, b_name, out_name = tensor_names
    return (
        Tensor(name=a_name, shape=op.a.shape),
        Tensor(name=b_name, shape=op.b.shape),
        Tensor(name=out_name, shape=op.out.shape),
    )


def generate_op_assignment(
    op: Compute,
    axis_to_var: dict[str, str],
    tensor_names: tuple[str, str, str] | None,
) -> str:
    """Computeを代入式に変換する."""
    if op.op not in OP_MAP:
        raise ValueError(f"Unsupported op: {op.op}")
    a, b, out = _resolve_tensors(op, tensor_names)
    a_indices = _resolve_tensor_indices(op.a_index, axis_to_var)
    b_indices = _resolve_tensor_indices(op.b_index, axis_to_var)
    out_indices = _resolve_tensor_indices(op.out_index, axis_to_var)
    out_ref = format_tensor_access(out, out_indices)
    a_ref = format_tensor_access(a, a_indices)
    b_ref = format_tensor_access(b, b_indices)
    return f"{out_ref} = {a_ref} {OP_MAP[op.op]} {b_ref}"


def generate_user_stmt(call: Call, domain_exprs: Mapping[str, Compute]) -> str:
    """ユーザー文をIRに基づいてC文に変換する."""
    if not call.args:
        raise ValueError("User call must have a statement id")
    stmt_id = call.args[0]
    if not isinstance(stmt_id, Id):
        raise ValueError("User call must start with an Id")
    op = domain_exprs.get(stmt_id.name)
    if op is None:
        raise ValueError(f"Unknown statement id: {stmt_id.name}")
    tensor_names, index_exprs = split_call_args(op, call.args[1:])
    indices = [generate_index_expr(arg) for arg in index_exprs]
    # Domainの軸名とISL変数のマッピングを作成
    axis_to_var = {axis.name: indices[i] for i, axis in enumerate(op.domain.axis)}
    return f"{generate_op_assignment(op, axis_to_var, tensor_names)};"
