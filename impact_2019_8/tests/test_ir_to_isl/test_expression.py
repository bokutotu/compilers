"""式変換のテスト."""

from ir_to_isl import constraint_to_isl, expr_to_isl
from ir_types import BinaryOp, Compare, IntConst, Var


def test_expr_to_isl():
    """Exprのexpr_to_isl変換テスト."""
    # 単純な変数
    assert expr_to_isl(Var("i")) == "i"

    # 定数
    assert expr_to_isl(IntConst(5)) == "5"

    # 足し算: i + j
    expr = BinaryOp(op="Add", lhs=Var("i"), rhs=Var("j"))
    assert expr_to_isl(expr) == "(i + j)"

    # 引き算: i - j
    expr = BinaryOp(op="Sub", lhs=Var("i"), rhs=Var("j"))
    assert expr_to_isl(expr) == "(i - j)"

    # 複雑な式: 2*i + j
    expr = BinaryOp(
        op="Add",
        lhs=BinaryOp(op="Mul", lhs=IntConst(2), rhs=Var("i")),
        rhs=Var("j"),
    )
    assert expr_to_isl(expr) == "((2 * i) + j)"


def test_constraint_to_isl():
    """Constraintのconstraint_to_isl変換テスト."""
    # j <= i
    constraint = Compare(lhs=Var("j"), op="LE", rhs=Var("i"))
    assert constraint_to_isl(constraint) == "j <= i"

    # i + j < N
    constraint = Compare(
        lhs=BinaryOp(op="Add", lhs=Var("i"), rhs=Var("j")),
        op="LT",
        rhs=Var("N"),
    )
    assert constraint_to_isl(constraint) == "(i + j) < N"

    # 2*i >= j
    constraint = Compare(
        lhs=BinaryOp(op="Mul", lhs=IntConst(2), rhs=Var("i")),
        op="GE",
        rhs=Var("j"),
    )
    assert constraint_to_isl(constraint) == "(2 * i) >= j"
