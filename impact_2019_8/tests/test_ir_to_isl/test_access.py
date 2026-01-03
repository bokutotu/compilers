"""アクセス解析のテスト."""

import islpy as isl

from ir_to_isl import build_read_access, build_write_access
from ir_types import Compare, IntConst, Iterator, Var
from tests.test_ir_to_isl.conftest import make_simple_func


def test_build_write_access():
    """Write accessのテスト: C[i, j] への書き込み."""
    ctx = isl.Context()
    func = make_simple_func(
        iterators=(
            Iterator(name="i"),
            Iterator(name="j"),
        ),
        params=("N", "M"),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=Var("M")),
        ),
    )
    write_access = build_write_access(func, ctx)

    expected = isl.UnionMap(
        "[N, M] -> { S[i, j] -> C[i, j] : 0 <= i < N and 0 <= j < M }", ctx
    )
    assert write_access.is_equal(expected)


def test_build_read_access():
    """Read accessのテスト: A[i, j] と B[i, j] からの読み込み."""
    ctx = isl.Context()
    func = make_simple_func(
        iterators=(
            Iterator(name="i"),
            Iterator(name="j"),
        ),
        params=("N", "M"),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=Var("M")),
        ),
    )
    read_access = build_read_access(func, ctx)

    expected_a = isl.UnionMap(
        "[N, M] -> { S[i, j] -> A[i, j] : 0 <= i < N and 0 <= j < M }", ctx
    )
    expected_b = isl.UnionMap(
        "[N, M] -> { S[i, j] -> B[i, j] : 0 <= i < N and 0 <= j < M }", ctx
    )
    expected = expected_a.union(expected_b)
    assert read_access.is_equal(expected)
