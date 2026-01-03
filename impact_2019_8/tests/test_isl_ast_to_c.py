"""isl_ast_to_cモジュールのテスト."""

import islpy as isl

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
from isl_ast_converter import convert_ast_node


def test_isl_ast_to_c():
    """ISL AST文字列からC言語コードが生成されることをテストする."""
    # ISL で AST を生成
    ctx = isl.Context()
    domain = isl.UnionSet("{ S[i] : 0 <= i <= 9 }")
    schedule = isl.UnionMap("{ S[i] -> [i] }")
    schedule = schedule.intersect_domain(domain)
    build = isl.AstBuild.alloc(ctx)
    isl_ast = build.node_from_schedule_map(schedule)

    ast = convert_ast_node(isl_ast)
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
