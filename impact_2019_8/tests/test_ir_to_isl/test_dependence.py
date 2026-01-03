"""依存解析のテスト."""

import islpy as isl

from ir_to_isl import (
    build_read_access,
    build_schedule,
    build_write_access,
    compute_all_dependences,
    compute_raw_dependence,
    compute_war_dependence,
    compute_waw_dependence,
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


def _make_raw_dependence_func() -> PrimFunc:
    """RAW依存を持つPrimFuncを生成: C[i] = C[i-1] + A[i]."""
    c = Tensor(name="C", shape=(Var("N"),))
    a = Tensor(name="A", shape=(Var("N"),))

    return PrimFunc(
        name="kernel",
        params=(a, c),
        computes=(
            Compute(
                name="S",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="i"),),
                    constraints=(
                        Compare(lhs=IntConst(1), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=c, index=(Var("i"),)),
                    value=BinaryOp(
                        op="Add",
                        lhs=Load(
                            access=Access(
                                tensor=c,
                                index=(
                                    BinaryOp(op="Sub", lhs=Var("i"), rhs=IntConst(1)),
                                ),
                            )
                        ),
                        rhs=Load(access=Access(tensor=a, index=(Var("i"),))),
                    ),
                ),
            ),
        ),
        schedule=Schedule(loop_order=("i",)),
    )


def test_compute_raw_dependence():
    """RAW依存のテスト: C[i] = C[i-1] + A[i] で S[i-1] -> S[i] の依存."""
    ctx = isl.Context()
    func = _make_raw_dependence_func()

    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)
    read_access = build_read_access(func, ctx)

    raw = compute_raw_dependence(schedule, write_access, read_access)

    expected = isl.UnionMap(
        "[N] -> { S[i] -> S[i'] : i' = i + 1 and 1 <= i and i' < N }", ctx
    )
    assert raw.is_equal(expected)


def _make_war_dependence_func() -> PrimFunc:
    """WAR依存を持つPrimFuncを生成: C[i] = C[i+1] + A[i]."""
    c = Tensor(name="C", shape=(Var("N"),))
    a = Tensor(name="A", shape=(Var("N"),))

    return PrimFunc(
        name="kernel",
        params=(a, c),
        computes=(
            Compute(
                name="S",
                domain=Domain(
                    params=("N",),
                    iterators=(Iterator(name="i"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
                    ),
                ),
                body=Store(
                    access=Access(tensor=c, index=(Var("i"),)),
                    value=BinaryOp(
                        op="Add",
                        lhs=Load(
                            access=Access(
                                tensor=c,
                                index=(
                                    BinaryOp(op="Add", lhs=Var("i"), rhs=IntConst(1)),
                                ),
                            )
                        ),
                        rhs=Load(access=Access(tensor=a, index=(Var("i"),))),
                    ),
                ),
            ),
        ),
        schedule=Schedule(loop_order=("i",)),
    )


def test_compute_war_dependence():
    """WAR依存のテスト: C[i] = C[i+1] + A[i] で S[i] -> S[i+1] の依存."""
    ctx = isl.Context()
    func = _make_war_dependence_func()

    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)
    read_access = build_read_access(func, ctx)

    war = compute_war_dependence(schedule, write_access, read_access)

    expected = isl.UnionMap(
        "[N] -> { S[i] -> S[i'] : i' = i + 1 and 0 <= i and i' < N }", ctx
    )
    assert war.is_equal(expected)


def _make_waw_dependence_func() -> PrimFunc:
    """WAW依存を持つPrimFuncを生成: 2つのComputeが同じ配列に書き込む."""
    c = Tensor(name="C", shape=(Var("N"),))
    a = Tensor(name="A", shape=(Var("N"),))

    return PrimFunc(
        name="kernel",
        params=(a, c),
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
                    access=Access(tensor=c, index=(Var("i"),)),
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
                    value=IntConst(0),
                ),
            ),
        ),
        schedule=Schedule(loop_order=("i", "j")),
    )


def test_compute_waw_dependence():
    """WAW依存のテスト: S1とS2が同じC[i]に書き込む."""
    ctx = isl.Context()
    func = _make_waw_dependence_func()

    schedule = build_schedule(func, ctx)
    write_access = build_write_access(func, ctx)

    waw = compute_waw_dependence(schedule, write_access)

    expected = isl.UnionMap("[N] -> { S1[i] -> S2[i] : 0 <= i < N }", ctx)
    assert waw.is_equal(expected)


def test_compute_all_dependences():
    """compute_all_dependencesのテスト."""
    ctx = isl.Context()
    func = _make_raw_dependence_func()

    deps = compute_all_dependences(func, ctx)

    assert "RAW" in deps
    assert "WAR" in deps
    assert "WAW" in deps
    assert isinstance(deps["RAW"], isl.UnionMap)
    assert isinstance(deps["WAR"], isl.UnionMap)
    assert isinstance(deps["WAW"], isl.UnionMap)
