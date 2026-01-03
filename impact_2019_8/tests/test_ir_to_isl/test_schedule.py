"""スケジュール構築のテスト."""

import islpy as isl

from ir_to_isl import build_schedule
from ir_types import Compare, IntConst, Iterator, Var
from tests.test_ir_to_isl.conftest import make_simple_func


def test_build_schedule_literal_extents():
    """数値の範囲からスケジュールを生成する."""
    ctx = isl.Context()
    func = make_simple_func(
        iterators=(Iterator(name="i"),),
        params=(),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(4)),
        ),
    )
    schedule = build_schedule(func, ctx)

    expected = isl.UnionMap("{ S[i] -> [i] : 0 <= i < 4 }", ctx)
    assert schedule.is_equal(expected)


def test_build_schedule_symbolic_extents():
    """記号パラメータでスケジュールを生成する."""
    ctx = isl.Context()
    func = make_simple_func(
        iterators=(Iterator(name="i"),),
        params=("N",),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
        ),
    )
    schedule = build_schedule(func, ctx)

    expected_schedule = isl.UnionMap("[N] -> { S[i] -> [i] : 0 <= i < N }", ctx)
    assert schedule.is_equal(expected_schedule)
