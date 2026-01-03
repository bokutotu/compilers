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
