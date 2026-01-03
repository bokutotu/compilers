"""isl_ast_to_cモジュールのテスト."""

from codegen import isl_ast_to_c
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
from isl_ast_parser import parse_isl_ast


def test_isl_ast_to_c():
    """ISL AST文字列からC言語コードが生成されることをテストする."""
    ast_str = (
        "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: "
        "[ { id: c0 }, { val: 9 } ] }, inc: { val: 1 }, body: { user: "
        "{ op: call, args: [ { id: S }, { id: c0 } ] } } }"
    )

    ast = parse_isl_ast(ast_str)
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
        name="kernel",
        computes=(compute,),
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
