"""compilerモジュールのテスト."""

from compiler import compile
from ir_types import (
    AffineConstraint,
    AffineExpr,
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


def test_compile():
    """PrimFuncをコンパイルできることをテストする."""
    a = Tensor("A", (10,))
    b = Tensor("B", (10,))
    c = Tensor("C", (10,))

    domain = Domain((Axis("i", 10),))
    schedule = Schedule(("i",))

    compute = Compute(
        name="S",
        domain=domain,
        stmt=Store(
            target=c,
            index=("i",),
            value=BinaryOp(
                op="add",
                left=Load(tensor=a, index=("i",)),
                right=Load(tensor=b, index=("i",)),
            ),
        ),
    )

    func = PrimFunc(
        name="add_func",
        compute=compute,
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
    a = Tensor("A", (4, 4))
    b = Tensor("B", (4, 4))
    c = Tensor("C", (4, 4))

    # j <= i という制約
    constraint = AffineConstraint(
        lhs=AffineExpr.from_var("j"),
        op="LE",
        rhs=AffineExpr.from_var("i"),
    )

    domain = Domain(
        axis=(Axis("i", 4), Axis("j", 4)),
        constraints=(constraint,),
    )
    schedule = Schedule(("i", "j"))

    compute = Compute(
        name="S",
        domain=domain,
        stmt=Store(
            target=c,
            index=("i", "j"),
            value=BinaryOp(
                op="add",
                left=Load(tensor=a, index=("i", "j")),
                right=Load(tensor=b, index=("i", "j")),
            ),
        ),
    )

    func = PrimFunc(
        name="triangular",
        compute=compute,
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
    a = Tensor("A", (4, 4))
    b = Tensor("B", (4, 4))
    c = Tensor("C", (4, 4))

    # i + j < 4 という制約（上三角の一部）
    constraint = AffineConstraint(
        lhs=AffineExpr.from_var("i") + "j",
        op="LT",
        rhs=AffineExpr.from_const(4),
    )

    domain = Domain(
        axis=(Axis("i", 4), Axis("j", 4)),
        constraints=(constraint,),
    )
    schedule = Schedule(("i", "j"))

    compute = Compute(
        name="S",
        domain=domain,
        stmt=Store(
            target=c,
            index=("i", "j"),
            value=BinaryOp(
                op="add",
                left=Load(tensor=a, index=("i", "j")),
                right=Load(tensor=b, index=("i", "j")),
            ),
        ),
    )

    func = PrimFunc(
        name="upper_triangular",
        compute=compute,
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
