"""ComputeからC言語の代入文を生成する関数群."""

from __future__ import annotations

from ast_types import Call
from ir_types import (
    Access,
    BinaryOp,
    BinOpKind,
    Compute,
    Expr,
    FloatConst,
    IntConst,
    Load,
    ReduceStore,
    Store,
    Tensor,
    Var,
)

from .expr import generate_index_expr

_BIN_OP_SYMBOL: dict[BinOpKind, str] = {
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
}

_BIN_OP_PRECEDENCE: dict[BinOpKind, int] = {
    "Add": 1,
    "Sub": 1,
    "Mul": 2,
    "Div": 2,
}

_UPDATE_OP: dict[str, str] = {
    "Sum": "+=",
    "Prod": "*=",
}


def _wrap_if_needed(expr: str) -> str:
    """式が演算子を含む場合、括弧で囲む."""
    # 空白を含む場合は演算子がある可能性が高い
    if " " in expr:
        return f"({expr})"
    return expr


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

    # 最初のインデックスが複雑な式の場合は括弧で囲む
    offset = _wrap_if_needed(indices[0])
    for dim in range(1, len(indices)):
        extent = tensor.shape[dim]
        # extentがExprの場合は文字列化
        if isinstance(extent, Var):
            extent_str = extent.name
        elif isinstance(extent, IntConst):
            extent_str = str(extent.value)
        else:
            extent_str = str(extent)
        offset = f"({offset}*{extent_str} + {_wrap_if_needed(indices[dim])})"
    return f"{tensor.name}[{offset}]"


def _resolve_indices(access: Access, axis_to_var: dict[str, str]) -> list[str]:
    result = []
    for idx_expr in access.index:
        if isinstance(idx_expr, Var):
            result.append(axis_to_var.get(idx_expr.name, idx_expr.name))
        elif isinstance(idx_expr, IntConst):
            result.append(str(idx_expr.value))
        else:
            # 複雑な式の場合は再帰的に処理
            result.append(_generate_ir_expr(idx_expr, axis_to_var))
    return result


def _needs_parens(
    parent_op: BinOpKind, child_op: BinOpKind, child_is_right: bool
) -> bool:
    parent_prec = _BIN_OP_PRECEDENCE.get(parent_op, 0)
    child_prec = _BIN_OP_PRECEDENCE.get(child_op, 0)

    if child_prec < parent_prec:
        return True
    if child_prec > parent_prec:
        return False
    if not child_is_right:
        return False

    if parent_op in ("Sub", "Div"):
        return True
    return bool(parent_op == "Mul" and child_op == "Div")


def _generate_ir_expr(
    expr: Expr,
    axis_to_var: dict[str, str],
    parent_op: BinOpKind | None = None,
    child_is_right: bool = False,
) -> str:
    if isinstance(expr, IntConst):
        return str(expr.value)
    if isinstance(expr, FloatConst):
        return str(expr.value)
    if isinstance(expr, Var):
        return axis_to_var.get(expr.name, expr.name)
    if isinstance(expr, Load):
        indices = _resolve_indices(expr.access, axis_to_var)
        return _format_tensor_access(expr.access.tensor, indices)
    if isinstance(expr, BinaryOp):
        left = _generate_ir_expr(expr.lhs, axis_to_var, expr.op, False)
        right = _generate_ir_expr(expr.rhs, axis_to_var, expr.op, True)
        symbol = _BIN_OP_SYMBOL.get(expr.op, expr.op)
        rendered = f"{left} {symbol} {right}"
        if parent_op is not None and _needs_parens(parent_op, expr.op, child_is_right):
            return f"({rendered})"
        return rendered

    raise ValueError(f"Unknown Expr type: {type(expr)}")


def _generate_reduction_init_cond(
    compute: Compute,
    target_access: Access,
    axis_to_var: dict[str, str],
) -> str:
    # ターゲットのインデックスに含まれる変数名を取得
    target_vars = set()
    for idx_expr in target_access.index:
        if isinstance(idx_expr, Var):
            target_vars.add(idx_expr.name)

    # reduce軸を探す
    reduce_axes = [it for it in compute.domain.iterators if it.kind == "reduce"]
    if not reduce_axes:
        reduce_axes = [
            it for it in compute.domain.iterators if it.name not in target_vars
        ]
    if not reduce_axes:
        return "1"

    parts = []
    for it in reduce_axes:
        var = axis_to_var.get(it.name)
        if var is None:
            raise ValueError(f"Missing loop variable for iterator '{it.name}'")
        # 初期値は0を仮定（制約から取得するのが本来の実装）
        parts.append(f"{var} == 0")
    return " && ".join(parts)


def generate_user_stmt(call: Call, compute: Compute) -> str:
    num_indices = len(compute.domain.iterators)
    index_exprs = call.args[-num_indices:] if call.args and num_indices > 0 else []
    indices = [generate_index_expr(arg) for arg in index_exprs]

    axis_to_var = {it.name: indices[i] for i, it in enumerate(compute.domain.iterators)}

    stmt = compute.body
    if isinstance(stmt, Store):
        target_ref = _format_tensor_access(
            stmt.access.tensor, _resolve_indices(stmt.access, axis_to_var)
        )
        value = _generate_ir_expr(stmt.value, axis_to_var)
        return f"{target_ref} = {value};"

    if isinstance(stmt, ReduceStore):
        target_ref = _format_tensor_access(
            stmt.access.tensor, _resolve_indices(stmt.access, axis_to_var)
        )
        value = _generate_ir_expr(stmt.value, axis_to_var)

        # TODO: 現在の実装は毎イテレーションでif文を評価するため非効率
        # - SIMD化が阻害される
        # - 分岐予測ミスが発生しやすい
        # 改善案:
        # 1. ループ分割: 初期化を別ループとして生成
        # 2. ループピーリング: 最初のイテレーションを分離
        # ISLで初期化を別statementとして扱い、依存関係を定義するのが本筋
        lines: list[str] = []
        if stmt.init is not None:
            cond = _generate_reduction_init_cond(compute, stmt.access, axis_to_var)
            init_value = _generate_ir_expr(stmt.init, axis_to_var)
            lines.append(f"if ({cond}) {target_ref} = {init_value};")

        if stmt.op in _UPDATE_OP:
            lines.append(f"{target_ref} {_UPDATE_OP[stmt.op]} {value};")
            return "\n".join(lines)

        if stmt.op == "Max":
            lines.append(
                f"{target_ref} = ({target_ref} > {value}) ? {target_ref} : {value};"
            )
            return "\n".join(lines)

        if stmt.op == "Min":
            lines.append(
                f"{target_ref} = ({target_ref} < {value}) ? {target_ref} : {value};"
            )
            return "\n".join(lines)

        raise ValueError(f"Unsupported reduce op: {stmt.op}")

    raise ValueError(f"Unknown Stmt type: {type(stmt)}")
