"""compilerモジュールのテスト."""

from compiler import compile
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


def test_compile():
    """PrimFuncをコンパイルできることをテストする."""
    a = Tensor("A", (IntConst(10),))
    b = Tensor("B", (IntConst(10),))
    c = Tensor("C", (IntConst(10),))

    domain = Domain(
        params=(),
        iterators=(Iterator("i"),),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(10)),
        ),
    )
    schedule = Schedule(("i",))

    compute = Compute(
        name="S",
        domain=domain,
        body=Store(
            access=Access(tensor=c, index=(Var("i"),)),
            value=BinaryOp(
                op="Add",
                lhs=Load(access=Access(tensor=a, index=(Var("i"),))),
                rhs=Load(access=Access(tensor=b, index=(Var("i"),))),
            ),
        ),
    )

    func = PrimFunc(
        name="add_func",
        computes=(compute,),
        schedule=schedule,
        params=(a, b, c),
    )

    c_code = compile(func)

    expected = """\
void add_func(int *A, int *B, int *C) {
    for (int c0 = 0; c0 <= 9; c0++) {
        C[c0] = A[c0] + B[c0];
    }
}"""

    assert c_code == expected


def test_compile_triangular_domain():
    """三角行列ドメイン(j <= i)をコンパイルできることをテストする."""
    a = Tensor("A", (IntConst(4), IntConst(4)))
    b = Tensor("B", (IntConst(4), IntConst(4)))
    c = Tensor("C", (IntConst(4), IntConst(4)))

    domain = Domain(
        params=(),
        iterators=(Iterator("i"), Iterator("j")),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(4)),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=IntConst(4)),
            Compare(lhs=Var("j"), op="LE", rhs=Var("i")),  # j <= i
        ),
    )
    schedule = Schedule(("i", "j"))

    compute = Compute(
        name="S",
        domain=domain,
        body=Store(
            access=Access(tensor=c, index=(Var("i"), Var("j"))),
            value=BinaryOp(
                op="Add",
                lhs=Load(access=Access(tensor=a, index=(Var("i"), Var("j")))),
                rhs=Load(access=Access(tensor=b, index=(Var("i"), Var("j")))),
            ),
        ),
    )

    func = PrimFunc(
        name="triangular",
        computes=(compute,),
        schedule=schedule,
        params=(a, b, c),
    )

    c_code = compile(func)

    expected = """\
void triangular(int *A, int *B, int *C) {
    for (int c0 = 0; c0 <= 3; c0++) {
        for (int c1 = 0; c1 <= c0; c1++) {
            C[(c0*4 + c1)] = A[(c0*4 + c1)] + B[(c0*4 + c1)];
        }
    }
}"""

    assert c_code == expected


def test_compile_sum_constraint():
    """i + j < N の制約をコンパイルできることをテストする."""
    a = Tensor("A", (IntConst(4), IntConst(4)))
    b = Tensor("B", (IntConst(4), IntConst(4)))
    c = Tensor("C", (IntConst(4), IntConst(4)))

    # i + j < 4 という制約（上三角の一部）
    domain = Domain(
        params=(),
        iterators=(Iterator("i"), Iterator("j")),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(4)),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=IntConst(4)),
            Compare(
                lhs=BinaryOp(op="Add", lhs=Var("i"), rhs=Var("j")),
                op="LT",
                rhs=IntConst(4),
            ),
        ),
    )
    schedule = Schedule(("i", "j"))

    compute = Compute(
        name="S",
        domain=domain,
        body=Store(
            access=Access(tensor=c, index=(Var("i"), Var("j"))),
            value=BinaryOp(
                op="Add",
                lhs=Load(access=Access(tensor=a, index=(Var("i"), Var("j")))),
                rhs=Load(access=Access(tensor=b, index=(Var("i"), Var("j")))),
            ),
        ),
    )

    func = PrimFunc(
        name="upper_triangular",
        computes=(compute,),
        schedule=schedule,
        params=(a, b, c),
    )

    c_code = compile(func)

    # i + j < 4 なので、内側ループは j <= -i + 3 まで
    expected = """\
void upper_triangular(int *A, int *B, int *C) {
    for (int c0 = 0; c0 <= 3; c0++) {
        for (int c1 = 0; c1 <= ((-c0) + 3); c1++) {
            C[(c0*4 + c1)] = A[(c0*4 + c1)] + B[(c0*4 + c1)];
        }
    }
}"""

    assert c_code == expected


def test_compile_chained_computes():
    """連鎖する2つのCompute: B[i] = A[i] + 1, C[i] = B[i] * 2."""
    n = 10
    a = Tensor("A", (IntConst(n),))
    b = Tensor("B", (IntConst(n),))
    c = Tensor("C", (IntConst(n),))

    func = PrimFunc(
        name="chained",
        params=(a, b, c),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=(),
                    iterators=(Iterator("i"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=IntConst(n)),
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
                    params=(),
                    iterators=(Iterator("j"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=IntConst(n)),
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
        schedule=Schedule(("i", "j")),
    )

    c_code = compile(func)

    expected = """\
void chained(int *A, int *B, int *C) {
    for (int c1 = 0; c1 <= 9; c1++) {
        B[c1] = A[c1] + 1;
    }
    for (int c1 = 0; c1 <= 9; c1++) {
        C[c1] = B[c1] * 2;
    }
}"""

    assert c_code == expected


def test_compile_triple_computes():
    """3つのCompute: B[i] = A[i], C[j] = B[j], D[k] = C[k]."""
    n = 8
    a = Tensor("A", (IntConst(n),))
    b = Tensor("B", (IntConst(n),))
    c = Tensor("C", (IntConst(n),))
    d = Tensor("D", (IntConst(n),))

    func = PrimFunc(
        name="triple",
        params=(a, b, c, d),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=(),
                    iterators=(Iterator("i"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=IntConst(n)),
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
                    params=(),
                    iterators=(Iterator("j"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=IntConst(n)),
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
                    params=(),
                    iterators=(Iterator("k"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("k")),
                        Compare(lhs=Var("k"), op="LT", rhs=IntConst(n)),
                    ),
                ),
                body=Store(
                    access=Access(tensor=d, index=(Var("k"),)),
                    value=Load(access=Access(tensor=c, index=(Var("k"),))),
                ),
            ),
        ),
        schedule=Schedule(("i", "j", "k")),
    )

    c_code = compile(func)

    expected = """\
void triple(int *A, int *B, int *C, int *D) {
    for (int c1 = 0; c1 <= 7; c1++) {
        B[c1] = A[c1];
    }
    for (int c1 = 0; c1 <= 7; c1++) {
        C[c1] = B[c1];
    }
    for (int c1 = 0; c1 <= 7; c1++) {
        D[c1] = C[c1];
    }
}"""

    assert c_code == expected


def test_compile_independent_computes():
    """独立した2つのCompute: B[i] = A[i], D[j] = C[j] (依存なし)."""
    n = 10
    m = 8
    a = Tensor("A", (IntConst(n),))
    b = Tensor("B", (IntConst(n),))
    c = Tensor("C", (IntConst(m),))
    d = Tensor("D", (IntConst(m),))

    func = PrimFunc(
        name="independent",
        params=(a, b, c, d),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=(),
                    iterators=(Iterator("i"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=IntConst(n)),
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
                    params=(),
                    iterators=(Iterator("j"),),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=IntConst(m)),
                    ),
                ),
                body=Store(
                    access=Access(tensor=d, index=(Var("j"),)),
                    value=Load(access=Access(tensor=c, index=(Var("j"),))),
                ),
            ),
        ),
        schedule=Schedule(("i", "j")),
    )

    c_code = compile(func)

    expected = """\
void independent(int *A, int *B, int *C, int *D) {
    for (int c1 = 0; c1 <= 9; c1++) {
        B[c1] = A[c1];
    }
    for (int c1 = 0; c1 <= 7; c1++) {
        D[c1] = C[c1];
    }
}"""

    assert c_code == expected


def test_compile_2d_multi_compute():
    """2次元の複数Compute: B[i,j] = A[i,j], C[i,j] = B[i,j] + B[i-1,j]."""
    n = 4
    m = 4
    a = Tensor("A", (IntConst(n), IntConst(m)))
    b = Tensor("B", (IntConst(n), IntConst(m)))
    c = Tensor("C", (IntConst(n), IntConst(m)))

    func = PrimFunc(
        name="stencil_2d",
        params=(a, b, c),
        computes=(
            Compute(
                name="S1",
                domain=Domain(
                    params=(),
                    iterators=(Iterator("i"), Iterator("j")),
                    constraints=(
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=IntConst(n)),
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=IntConst(m)),
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
                    params=(),
                    iterators=(Iterator("i"), Iterator("j")),
                    constraints=(
                        Compare(lhs=IntConst(1), op="LE", rhs=Var("i")),
                        Compare(lhs=Var("i"), op="LT", rhs=IntConst(n)),
                        Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                        Compare(lhs=Var("j"), op="LT", rhs=IntConst(m)),
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
        schedule=Schedule(("i", "j")),
    )

    c_code = compile(func)

    expected = """\
void stencil_2d(int *A, int *B, int *C) {
    for (int c1 = 0; c1 <= 3; c1++) {
        for (int c2 = 0; c2 <= 3; c2++) {
            B[(c1*4 + c2)] = A[(c1*4 + c2)];
        }
    }
    for (int c1 = 1; c1 <= 3; c1++) {
        for (int c2 = 0; c2 <= 3; c2++) {
            C[(c1*4 + c2)] = B[(c1*4 + c2)] + B[((c1 - 1)*4 + c2)];
        }
    }
}"""

    assert c_code == expected
