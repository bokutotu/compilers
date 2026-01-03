"""ir_to_islモジュールのテスト."""

import islpy as isl

from ir_to_isl import (
    build_domain,
    build_read_access,
    build_schedule,
    build_write_access,
    compute_all_dependences,
    compute_raw_dependence,
    compute_war_dependence,
    compute_waw_dependence,
    constraint_to_isl,
    expr_to_isl,
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


def _make_func(
    iterators: tuple[Iterator, ...],
    params: tuple[str, ...],
    constraints: tuple[Compare, ...] = (),
) -> PrimFunc:
    """テスト用のPrimFuncを生成する."""
    # 各イテレータの範囲に応じた形状を生成
    if params:
        shape = tuple(Var(p) for p in params)
    else:
        shape = tuple(IntConst(4) for _ in iterators)

    a = Tensor(name="A", shape=shape)
    b = Tensor(name="B", shape=shape)
    out = Tensor(name="C", shape=shape)

    # インデックスはイテレータ名の変数
    index = tuple(Var(it.name) for it in iterators)

    return PrimFunc(
        name="kernel",
        params=(a, b, out),
        computes=(
            Compute(
                name="S",
                domain=Domain(
                    params=params,
                    iterators=iterators,
                    constraints=constraints,
                ),
                body=Store(
                    access=Access(tensor=out, index=index),
                    value=BinaryOp(
                        op="Add",
                        lhs=Load(access=Access(tensor=a, index=index)),
                        rhs=Load(access=Access(tensor=b, index=index)),
                    ),
                ),
            ),
        ),
        schedule=Schedule(loop_order=tuple(it.name for it in iterators)),
    )


def test_build_domain_literal_extents():
    """数値の範囲からドメインを生成する."""
    ctx = isl.Context()
    func = _make_func(
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


def test_build_schedule_literal_extents():
    """数値の範囲からスケジュールを生成する."""
    ctx = isl.Context()
    func = _make_func(
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


def test_build_domain_and_schedule_symbolic_extents():
    """記号パラメータでドメインとスケジュールを生成する."""
    ctx = isl.Context()
    func = _make_func(
        iterators=(Iterator(name="i"),),
        params=("N",),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=Var("N")),
        ),
    )
    domain = build_domain(func, ctx)
    schedule = build_schedule(func, ctx)

    expected_domain = isl.UnionSet("[N] -> { S[i] : 0 <= i < N }", ctx)
    expected_schedule = isl.UnionMap("[N] -> { S[i] -> [i] : 0 <= i < N }", ctx)
    assert domain.is_equal(expected_domain)
    assert schedule.is_equal(expected_schedule)


def test_build_domain_with_constraint_triangular():
    """三角行列のドメイン: 0 <= j <= i を生成する."""
    ctx = isl.Context()

    func = _make_func(
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
            Compare(lhs=Var("j"), op="LE", rhs=Var("i")),  # j <= i
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

    func = _make_func(
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
            # i + j < N
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


def test_build_write_access():
    """Write accessのテスト: C[i, j] への書き込み."""
    ctx = isl.Context()
    func = _make_func(
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
    func = _make_func(
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
                                index=(BinaryOp(op="Sub", lhs=Var("i"), rhs=IntConst(1)),),
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

    # C[i-1] を読み、C[i] を書く => S[i-1] から S[i] への依存
    # i-1 = j (書き込み位置) => j+1 = i (読み込み時のイテレータ)
    # つまり S[j] -> S[j+1] で 1 <= j < N-1 (j+1 < N より)
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
                                index=(BinaryOp(op="Add", lhs=Var("i"), rhs=IntConst(1)),),
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

    # C[i+1] を読み、C[i] を書く
    # 読み込み: S[i] -> C[i+1]
    # 書き込み: S[i] -> C[i]
    # WAR: 読み込みが先、書き込みが後で同じ位置
    # S[i] が C[i+1] を読む、S[i+1] が C[i+1] を書く => S[i] -> S[i+1]
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

    # 複数Computeなのでスケジュールは:
    #   S1[i] -> [0, i], S2[j] -> [1, j]
    # Write access:
    #   S1[i] -> C[i], S2[j] -> C[j]
    # 同じ位置に書き込む: i = j
    # 時間順序: [0, i] < [1, j] は常に真 (0 < 1)
    # したがって WAW: S1[i] -> S2[i] for 0 <= i < N
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
