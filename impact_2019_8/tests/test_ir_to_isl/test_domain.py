"""ドメイン構築のテスト."""

import islpy as isl

from ir_to_isl import build_domain
from ir_types import BinaryOp, Compare, IntConst, Iterator, Var
from tests.test_ir_to_isl.conftest import make_simple_func


def test_build_domain_literal_extents():
    """数値の範囲からドメインを生成する."""
    ctx = isl.Context()
    func = make_simple_func(
        iterators=(Iterator(name="i"),),
        params=(),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(4)),
        ),
    )
    domain = build_domain(func, ctx)

    expected = isl.UnionSet("{ S[i] : 0 <= i < 4 }", ctx)
    assert domain.is_equal(expected)


def test_build_domain_and_schedule_symbolic_extents():
    """記号パラメータでドメインを生成する."""
    ctx = isl.Context()
    func = make_simple_func(
        iterators=(Iterator(name="i"),),
        params=("N",),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
        ),
    )
    domain = build_domain(func, ctx)

    expected_domain = isl.UnionSet("[N] -> { S[i] : 0 <= i < N }", ctx)
    assert domain.is_equal(expected_domain)


def test_build_domain_with_constraint_triangular():
    """三角行列のドメイン: 0 <= j <= i を生成する."""
    ctx = isl.Context()

    func = make_simple_func(
        iterators=(
            Iterator(name="i"),
            Iterator(name="j"),
        ),
        params=("N",),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=Var("N")),
            Compare(lhs=Var("j"), op="LE", rhs=Var("i")),
        ),
    )
    domain = build_domain(func, ctx)

    expected = isl.UnionSet(
        "[N] -> { S[i, j] : 0 <= i < N and 0 <= j < N and j <= i }", ctx
    )
    assert domain.is_equal(expected)


def test_build_domain_with_constraint_sum():
    """i + j < N の制約を生成する."""
    ctx = isl.Context()

    func = make_simple_func(
        iterators=(
            Iterator(name="i"),
            Iterator(name="j"),
        ),
        params=("N",),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=Var("N")),
            Compare(
                lhs=BinaryOp(op="Add", lhs=Var("i"), rhs=Var("j")),
                op="LT",
                rhs=Var("N"),
            ),
        ),
    )
    domain = build_domain(func, ctx)

    expected = isl.UnionSet(
        "[N] -> { S[i, j] : 0 <= i < N and 0 <= j < N and i + j < N }", ctx
    )
    assert domain.is_equal(expected)
