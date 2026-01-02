"""compilerモジュールのテスト."""

from src.compiler import compile
from src.ir_types import (
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
