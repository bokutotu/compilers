"""ComputeからC言語の代入文を生成する関数群."""

from __future__ import annotations

from ast_types import Call
from ir_types import Compute, Index, Tensor

from .expr import generate_index_expr

OP_MAP = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
}


def _format_tensor_access(tensor: Tensor, indices: list[str]) -> str:
    """テンソルアクセス式を生成する."""
    suffix = "".join(f"[{index}]" for index in indices)
    return f"{tensor.name}{suffix}"


def _resolve_indices(index: Index, axis_to_var: dict[str, str]) -> list[str]:
    """インデックスを解決する."""
    return [axis_to_var[axis] for axis in index]


def generate_user_stmt(call: Call, compute: Compute) -> str:
    """ユーザー文をC文に変換する."""
    if compute.op not in OP_MAP:
        raise ValueError(f"Unsupported op: {compute.op}")

    # Callの引数の末尾からドメインの軸数分だけインデックスとして取得
    num_indices = len(compute.domain.axis)
    index_exprs = call.args[-num_indices:] if call.args and num_indices > 0 else []
    indices = [generate_index_expr(arg) for arg in index_exprs]

    # Domainの軸名とISL変数のマッピングを作成
    axis_to_var = {
        axis.name: indices[i] for i, axis in enumerate(compute.domain.axis)
    }

    # テンソルアクセス式を生成
    out_indices = _resolve_indices(compute.out_index, axis_to_var)
    a_indices = _resolve_indices(compute.a_index, axis_to_var)
    b_indices = _resolve_indices(compute.b_index, axis_to_var)

    out_ref = _format_tensor_access(compute.out, out_indices)
    a_ref = _format_tensor_access(compute.a, a_indices)
    b_ref = _format_tensor_access(compute.b, b_indices)

    return f"{out_ref} = {a_ref} {OP_MAP[compute.op]} {b_ref};"
