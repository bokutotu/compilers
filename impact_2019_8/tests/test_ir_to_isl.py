"""ir_to_islモジュールのテスト."""

import islpy as isl

from src.ir_to_isl import build_domain, build_domain_and_schedule, build_schedule
from src.ir_types import Axis, Compute, Domain, PrimFunc, Schedule, Tensor


def _make_func(axes: tuple[Axis, ...]) -> PrimFunc:
    a = Tensor(name="A", shape=tuple(ax.extent for ax in axes))
    b = Tensor(name="B", shape=tuple(ax.extent for ax in axes))
    out = Tensor(name="C", shape=tuple(ax.extent for ax in axes))
    return PrimFunc(
        name="kernel",
        compute=Compute(
            name="S",
            op="add",
            a=a,
            b=b,
            out=out,
            domain=Domain(axes),
        ),
        schedule=Schedule(loop_order=tuple(ax.name for ax in axes)),
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
