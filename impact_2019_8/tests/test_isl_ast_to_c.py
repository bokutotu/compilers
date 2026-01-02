"""isl_ast_to_cモジュールのテスト."""

from src.ast_parser import parse_isl_ast
from src.codegen import isl_ast_to_c
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


def test_isl_ast_to_c():
    """ISL AST文字列からC言語コードが生成されることをテストする."""
    ast_str = (
        "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: "
        "[ { id: c0 }, { val: 9 } ] }, inc: { val: 1 }, body: { user: "
        "{ op: call, args: [ { id: S }, { id: A }, { id: B }, { id: C }, "
        "{ id: c0 } ] } } }"
    )

    ast = parse_isl_ast(ast_str)
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
        name="kernel",
        compute=compute,
        schedule=schedule,
        params=(a, b, c),
    )
    c_code = isl_ast_to_c(ast, func)

    expected = """\
void kernel(int *A, int *B, int *C) {
    for (int c0 = 0; c0 <= 9; c0++) {
        C[c0] = A[c0] + B[c0];
    }
}"""

    assert c_code == expected
