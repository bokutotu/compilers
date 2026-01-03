"""ir_to_islモジュールのテスト."""

import islpy as isl

from ir_to_isl import (
    build_domain,
    build_domain_and_schedule,
    build_read_access,
    build_schedule,
    build_write_access,
)
from ir_types import (
    AffineConstraint,
    AffineExpr,
    AffineTerm,
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


def _make_func(
    axes: tuple[Axis, ...],
    constraints: tuple[AffineConstraint, ...] = (),
) -> PrimFunc:
    a = Tensor(name="A", shape=tuple(ax.extent for ax in axes))
    b = Tensor(name="B", shape=tuple(ax.extent for ax in axes))
    out = Tensor(name="C", shape=tuple(ax.extent for ax in axes))
    axis_names = tuple(ax.name for ax in axes)
    return PrimFunc(
        name="kernel",
        compute=Compute(
            name="S",
            domain=Domain(axes, constraints),
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


def test_build_domain_with_affine_constraint_triangular():
    """三角行列のドメイン: 0 <= j <= i を生成する."""
    ctx = isl.Context()

    # j <= i という制約を追加
    constraint = AffineConstraint(
        lhs=AffineExpr.from_var("j"),
        op="LE",
        rhs=AffineExpr.from_var("i"),
    )

    func = _make_func(
        axes=(
            Axis(name="i", extent="N", lower=0),
            Axis(name="j", extent="N", lower=0),
        ),
        constraints=(constraint,),
    )
    domain = build_domain(func, ctx)

    expected = isl.UnionSet("[N] -> { S[i, j] : 0 <= i < N and 0 <= j < N and j <= i }", ctx)
    assert domain.is_equal(expected)


def test_build_domain_with_affine_constraint_sum():
    """i + j < N の制約を生成する."""
    ctx = isl.Context()

    # i + j < N という制約
    constraint = AffineConstraint(
        lhs=AffineExpr.from_var("i") + "j",
        op="LT",
        rhs=AffineExpr.from_var("N"),
    )

    func = _make_func(
        axes=(
            Axis(name="i", extent="N", lower=0),
            Axis(name="j", extent="N", lower=0),
        ),
        constraints=(constraint,),
    )
    domain = build_domain(func, ctx)

    expected = isl.UnionSet(
        "[N] -> { S[i, j] : 0 <= i < N and 0 <= j < N and i + j < N }", ctx
    )
    assert domain.is_equal(expected)


def test_affine_expr_to_isl():
    """AffineExprのISL変換テスト."""
    # 単純な変数
    expr = AffineExpr.from_var("i")
    assert expr.to_isl() == "i"

    # 定数
    expr = AffineExpr.from_const(5)
    assert expr.to_isl() == "5"

    # 係数付き変数
    expr = AffineExpr.from_var("i", coeff=2)
    assert expr.to_isl() == "2*i"

    # 足し算: i + j
    expr = AffineExpr.from_var("i") + "j"
    assert expr.to_isl() == "i + j"

    # 引き算: i - j
    expr = AffineExpr.from_var("i") - "j"
    assert expr.to_isl() == "i - j"

    # 複雑な式: 2*i + 3*j - 1
    expr = AffineExpr.from_var("i", coeff=2) + AffineExpr.from_var("j", coeff=3) - 1
    assert expr.to_isl() == "2*i + 3*j - 1"


def test_affine_constraint_to_isl():
    """AffineConstraintのISL変換テスト."""
    # j <= i
    constraint = AffineConstraint(
        lhs=AffineExpr.from_var("j"),
        op="LE",
        rhs=AffineExpr.from_var("i"),
    )
    assert constraint.to_isl() == "j <= i"

    # i + j < N
    constraint = AffineConstraint(
        lhs=AffineExpr.from_var("i") + "j",
        op="LT",
        rhs=AffineExpr.from_var("N"),
    )
    assert constraint.to_isl() == "i + j < N"

    # 2*i >= j
    constraint = AffineConstraint(
        lhs=AffineExpr.from_var("i", coeff=2),
        op="GE",
        rhs=AffineExpr.from_var("j"),
    )
    assert constraint.to_isl() == "2*i >= j"


def test_build_write_access():
    """Write accessのテスト: C[i, j] への書き込み."""
    ctx = isl.Context()
    func = _make_func(
        (
            Axis(name="i", extent="N", lower=0),
            Axis(name="j", extent="M", lower=0),
        )
    )
    write_access = build_write_access(func, ctx)

    expected = isl.UnionMap(
        "[N, M] -> { S[i, j] -> C[i, j] : 0 <= i < N and 0 <= j < M }", ctx
    )
    assert write_access.is_equal(expected)


def test_build_read_access():
    """Read accessのテスト: A[i, j] と B[i, j] からの読み込み."""
    ctx = isl.Context()
    func = _make_func(
        (
            Axis(name="i", extent="N", lower=0),
            Axis(name="j", extent="M", lower=0),
        )
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
