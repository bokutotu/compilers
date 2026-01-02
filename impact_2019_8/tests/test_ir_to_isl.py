"""ir_to_islモジュールのテスト."""

import islpy as isl

from src.ir_to_isl import build_domain, build_domain_and_schedule, build_schedule
from src.ir_types import Dim, MatrixOp, MatrixPtr


def _make_op(dims: list[Dim]) -> MatrixOp:
    return MatrixOp(
        name="S",
        op="add",
        left=MatrixPtr("A", dims=dims),
        right=MatrixPtr("B", dims=dims),
        out=MatrixPtr("C", dims=dims),
    )


def test_build_domain_literal_extents():
    """数値の範囲からドメインを生成する."""
    ctx = isl.Context()
    op = _make_op([Dim(4)])
    domain = build_domain(op, ctx=ctx)

    expected = isl.UnionSet(
        "{ S[i] : 0 <= i < 4 }",
        ctx,
    )
    assert domain.is_equal(expected)


def test_build_schedule_literal_extents():
    """数値の範囲からスケジュールを生成する."""
    ctx = isl.Context()
    op = _make_op([Dim(4)])
    schedule = build_schedule(op, ctx=ctx)

    expected = isl.UnionMap(
        "{ S[i] -> [i] : 0 <= i < 4 }",
        ctx,
    )
    assert schedule.is_equal(expected)


def test_build_domain_and_schedule_symbolic_extents():
    """記号パラメータでドメインとスケジュールを生成する."""
    ctx = isl.Context()
    op = _make_op([Dim("N")])
    domain, schedule = build_domain_and_schedule(op, ctx=ctx)

    expected_domain = isl.UnionSet(
        "[N] -> { S[i] : 0 <= i < N }",
        ctx,
    )
    expected_schedule = isl.UnionMap(
        "[N] -> { S[i] -> [i] : 0 <= i < N }",
        ctx,
    )
    assert domain.is_equal(expected_domain)
    assert schedule.is_equal(expected_schedule)
