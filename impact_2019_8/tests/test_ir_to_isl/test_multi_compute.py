"""複数Computeを持つPrimFuncのテスト."""

import islpy as isl

from ir_to_isl import (
    build_domain,
    build_read_access,
    build_schedule,
    build_write_access,
    compute_raw_dependence,
)
from ir_types import (
    Access,
    BinaryOp,
    Compare,
    Compute,
    Domain,
    IntConst,
    Iterator,
    Load,
    PrimFunc,
    Schedule,
    Store,
    Tensor,
    Var,
)


def _make_chained_computes_func() -> PrimFunc:
    """連鎖する2つのCompute: B[i] = A[i] + 1, C[i] = B[i] * 2."""
    a = Tensor(name="A", shape=(Var("N"),))
    b = Tensor(name="B", shape=(Var("N"),))
    c = Tensor(name="C", shape=(Var("N"),))

    return PrimFunc(
        name="kernel",
        params=(a, b, c),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="i"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=b, index=(Var("i"),)),
                    value=BinaryOp(
                        op="Add",
                        lhs=Load(access=Access(tensor=a, index=(Var("i"),))),
                        rhs=IntConst(1),
                    ),
                ),
            ),
            Compute(
                name="S2",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="j"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=c, index=(Var("j"),)),
                    value=BinaryOp(
                        op="Mul",
                        lhs=Load(access=Access(tensor=b, index=(Var("j"),))),
                        rhs=IntConst(2),
                    ),
                ),
            ),
        ),
        schedule=Schedule(loop_order=("i", "j")),
    )


def test_multi_compute_domain():
    """複数Computeのドメインテスト."""
    ctx = isl.Context()
    func = _make_chained_computes_func()
    domain = build_domain(func, ctx)

    expected = isl.UnionSet(
        "[N] -> { S1[i] : 0 <= i < N; S2[j] : 0 <= j < N }", ctx
    )
    assert domain.is_equal(expected)


def test_multi_compute_schedule():
    """複数Computeのスケジュールテスト: S1[i] -> [i, 0], S2[j] -> [j, 1]."""
    ctx = isl.Context()
    func = _make_chained_computes_func()
    schedule = build_schedule(func, ctx)

    expected = isl.UnionMap(
        "[N] -> { S1[i] -> [i, 0] : 0 <= i < N; S2[j] -> [j, 1] : 0 <= j < N }",
        ctx,
    )
    assert schedule.is_equal(expected)


def test_multi_compute_write_access():
    """複数Computeのwrite accessテスト."""
    ctx = isl.Context()
    func = _make_chained_computes_func()
    write_access = build_write_access(func, ctx)

    expected = isl.UnionMap(
        "[N] -> { S1[i] -> B[i] : 0 <= i < N; S2[j] -> C[j] : 0 <= j < N }",
        ctx,
    )
    assert write_access.is_equal(expected)


def test_multi_compute_read_access():
    """複数Computeのread accessテスト."""
    ctx = isl.Context()
    func = _make_chained_computes_func()
    read_access = build_read_access(func, ctx)

    expected = isl.UnionMap(
        "[N] -> { S1[i] -> A[i] : 0 <= i < N; S2[j] -> B[j] : 0 <= j < N }",
        ctx,
    )
    assert read_access.is_equal(expected)


def test_multi_compute_raw_dependence():
    """連鎖するComputeのRAW依存: S1がBに書き込み、S2がBを読む."""
    ctx = isl.Context()
    func = _make_chained_computes_func()

    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)
    read_access = build_read_access(func, ctx)

    raw = compute_raw_dependence(schedule, write_access, read_access)

    expected = isl.UnionMap("[N] -> { S1[i] -> S2[i] : 0 <= i < N }", ctx)
    assert raw.is_equal(expected)


def _make_triple_computes_func() -> PrimFunc:
    """3つのCompute: B[i] = A[i], C[j] = B[j], D[k] = C[k]."""
    a = Tensor(name="A", shape=(Var("N"),))
    b = Tensor(name="B", shape=(Var("N"),))
    c = Tensor(name="C", shape=(Var("N"),))
    d = Tensor(name="D", shape=(Var("N"),))

    return PrimFunc(
        name="kernel",
        params=(a, b, c, d),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="i"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=b, index=(Var("i"),)),
                    value=Load(access=Access(tensor=a, index=(Var("i"),))),
                ),
            ),
            Compute(
                name="S2",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="j"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=c, index=(Var("j"),)),
                    value=Load(access=Access(tensor=b, index=(Var("j"),))),
                ),
            ),
            Compute(
                name="S3",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="k"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("k")),
                        Compare(lhs=Var("k"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=d, index=(Var("k"),)),
                    value=Load(access=Access(tensor=c, index=(Var("k"),))),
                ),
            ),
        ),
        schedule=Schedule(loop_order=("i", "j", "k")),
    )


def test_triple_compute_domain():
    """3つのComputeのドメインテスト."""
    ctx = isl.Context()
    func = _make_triple_computes_func()
    domain = build_domain(func, ctx)

    expected = isl.UnionSet(
        "[N] -> { S1[i] : 0 <= i < N; S2[j] : 0 <= j < N; S3[k] : 0 <= k < N }",
        ctx,
    )
    assert domain.is_equal(expected)


def test_triple_compute_schedule():
    """3つのComputeのスケジュールテスト."""
    ctx = isl.Context()
    func = _make_triple_computes_func()
    schedule = build_schedule(func, ctx)

    expected = isl.UnionMap(
        "[N] -> { S1[i] -> [i, 0] : 0 <= i < N; "
        "S2[j] -> [j, 1] : 0 <= j < N; "
        "S3[k] -> [k, 2] : 0 <= k < N }",
        ctx,
    )
    assert schedule.is_equal(expected)


def test_triple_compute_raw_dependence():
    """3つのComputeのRAW依存: S1->S2, S2->S3."""
    ctx = isl.Context()
    func = _make_triple_computes_func()

    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)
    read_access = build_read_access(func, ctx)

    raw = compute_raw_dependence(schedule, write_access, read_access)

    expected = isl.UnionMap(
        "[N] -> { S1[i] -> S2[i] : 0 <= i < N; S2[j] -> S3[j] : 0 <= j < N }",
        ctx,
    )
    assert raw.is_equal(expected)


def _make_independent_computes_func() -> PrimFunc:
    """独立した2つのCompute: B[i] = A[i], D[j] = C[j] (依存なし)."""
    a = Tensor(name="A", shape=(Var("N"),))
    b = Tensor(name="B", shape=(Var("N"),))
    c = Tensor(name="C", shape=(Var("M"),))
    d = Tensor(name="D", shape=(Var("M"),))

    return PrimFunc(
        name="kernel",
        params=(a, b, c, d),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="i"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=b, index=(Var("i"),)),
                    value=Load(access=Access(tensor=a, index=(Var("i"),))),
                ),
            ),
            Compute(
                name="S2",
                domain=Domain(
                    params=("M",),
                    iterators=(Iterator(name="j"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=Var("M")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=d, index=(Var("j"),)),
                    value=Load(access=Access(tensor=c, index=(Var("j"),))),
                ),
            ),
        ),
        schedule=Schedule(loop_order=("i", "j")),
    )


def test_independent_computes_no_raw_dependence():
    """独立したComputeにはRAW依存がない."""
    ctx = isl.Context()
    func = _make_independent_computes_func()

    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)
    read_access = build_read_access(func, ctx)

    raw = compute_raw_dependence(schedule, write_access, read_access)

    assert raw.is_empty()


def _make_2d_multi_compute_func() -> PrimFunc:
    """2次元の複数Compute: B[i,j] = A[i,j], C[i,j] = B[i,j] + B[i-1,j]."""
    a = Tensor(name="A", shape=(Var("N"), Var("M")))
    b = Tensor(name="B", shape=(Var("N"), Var("M")))
    c = Tensor(name="C", shape=(Var("N"), Var("M")))

    return PrimFunc(
        name="kernel",
        params=(a, b, c),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=("N", "M"),
                    iterators=(Iterator(name="i"), Iterator(name="j")),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=Var("M")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=b, index=(Var("i"), Var("j"))),
                    value=Load(access=Access(tensor=a, index=(Var("i"), Var("j")))),
                ),
            ),
            Compute(
                name="S2",
                domain=Domain(
                    params=("N", "M"),
                    iterators=(Iterator(name="i"), Iterator(name="j")),
                    constraints=(
                        Compare(lhs=IntConst(1), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=Var("M")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=c, index=(Var("i"), Var("j"))),
                    value=BinaryOp(
                        op="Add",
                        lhs=Load(access=Access(tensor=b, index=(Var("i"), Var("j")))),
                        rhs=Load(
                            access=Access(
                                tensor=b,
                                index=(
                                    BinaryOp(op="Sub", lhs=Var("i"), rhs=IntConst(1)),
                                    Var("j"),
                                ),
                            )
                        ),
                    ),
                ),
            ),
        ),
        schedule=Schedule(loop_order=("i", "j")),
    )


def test_2d_multi_compute_domain():
    """2次元複数Computeのドメインテスト."""
    ctx = isl.Context()
    func = _make_2d_multi_compute_func()
    domain = build_domain(func, ctx)

    expected = isl.UnionSet(
        "[N, M] -> { S1[i, j] : 0 <= i < N and 0 <= j < M; "
        "S2[i, j] : 1 <= i < N and 0 <= j < M }",
        ctx,
    )
    assert domain.is_equal(expected)


def test_2d_multi_compute_schedule():
    """2次元複数Computeのスケジュールテスト."""
    ctx = isl.Context()
    func = _make_2d_multi_compute_func()
    schedule = build_schedule(func, ctx)

    expected = isl.UnionMap(
        "[N, M] -> { S1[i, j] -> [i, j, 0] : 0 <= i < N and 0 <= j < M; "
        "S2[i, j] -> [i, j, 1] : 1 <= i < N and 0 <= j < M }",
        ctx,
    )
    assert schedule.is_equal(expected)


def test_2d_multi_compute_raw_dependence():
    """2次元複数ComputeのRAW依存: S2はB[i,j]とB[i-1,j]を読む."""
    ctx = isl.Context()
    func = _make_2d_multi_compute_func()

    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)
    read_access = build_read_access(func, ctx)

    raw = compute_raw_dependence(schedule, write_access, read_access)

    expected = isl.UnionMap(
        "[N, M] -> { "
        "S1[i, j] -> S2[i, j] : 1 <= i < N and 0 <= j < M; "
        "S1[i, j] -> S2[i + 1, j] : 0 <= i < N - 1 and 0 <= j < M "
        "}",
        ctx,
    )
    assert raw.is_equal(expected)
