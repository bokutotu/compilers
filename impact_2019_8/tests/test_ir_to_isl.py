"""ir_to_islモジュールのテスト."""

import islpy as isl

from ir_to_isl import build_domain, build_domain_and_schedule, build_schedule
from ir_types import (
    Axis,
    BinaryOp,
    Compute,
    Domain,
    Load,
    PrimFunc,
    Schedule,
    Store,
    Tensor,
)


def _make_func(axes: tuple[Axis, ...]) -> PrimFunc:
    a = Tensor(name="A", shape=tuple(ax.extent for ax in axes))
    b = Tensor(name="B", shape=tuple(ax.extent for ax in axes))
    out = Tensor(name="C", shape=tuple(ax.extent for ax in axes))
    axis_names = tuple(ax.name for ax in axes)
    return PrimFunc(
        name="kernel",
        compute=Compute(
            name="S",
            domain=Domain(axes),
            stmt=Store(
                target=out,
                index=axis_names,
                value=BinaryOp(
                    op="add",
                    left=Load(tensor=a, index=axis_names),
                    right=Load(tensor=b, index=axis_names),
                ),
            ),
        ),
        schedule=Schedule(loop_order=axis_names),
        params=(a, b, out),
    )


def test_build_domain_literal_extents():
    """数値の範囲からドメインを生成する."""
    ctx = isl.Context()
    func = _make_func((Axis(name="i", extent=4, lower=0),))
    domain = build_domain(func, ctx)

    expected = isl.UnionSet("{ S[i] : 0 <= i < 4 }", ctx)
    assert domain.is_equal(expected)


def test_build_schedule_literal_extents():
    """数値の範囲からスケジュールを生成する."""
    ctx = isl.Context()
    func = _make_func((Axis(name="i", extent=4, lower=0),))
    schedule = build_schedule(func, ctx)

    expected = isl.UnionMap("{ S[i] -> [i] : 0 <= i < 4 }", ctx)
    assert schedule.is_equal(expected)


def test_build_domain_and_schedule_symbolic_extents():
    """記号パラメータでドメインとスケジュールを生成する."""
    ctx = isl.Context()
    func = _make_func((Axis(name="i", extent="N", lower=0),))
    domain, schedule = build_domain_and_schedule(func, ctx)

    expected_domain = isl.UnionSet("[N] -> { S[i] : 0 <= i < N }", ctx)
    expected_schedule = isl.UnionMap("[N] -> { S[i] -> [i] : 0 <= i < N }", ctx)
    assert domain.is_equal(expected_domain)
    assert schedule.is_equal(expected_schedule)
