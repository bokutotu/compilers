"""reductionコード生成のテスト."""

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
    ReduceStore,
    Schedule,
    Tensor,
    Var,
)
from isl_ast_parser import parse_isl_ast


def test_codegen_gemm_like_reduction() -> None:
    ast_str = (
        "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: "
        "[ { id: c0 }, { val: 1 } ] }, inc: { val: 1 }, body: { iterator: "
        "{ id: c1 }, init: { val: 0 }, cond: { op: le, args: [ { id: c1 }, "
        "{ val: 2 } ] }, inc: { val: 1 }, body: { iterator: { id: c2 }, init: "
        "{ val: 0 }, cond: { op: le, args: [ { id: c2 }, { val: 3 } ] }, "
        "inc: { val: 1 }, body: { user: { op: call, args: [ { id: S }, { id: c0 }, "
        "{ id: c1 }, { id: c2 } ] } } } } } }"
    )

    ast = parse_isl_ast(ast_str)

    m, n, k = 2, 3, 4
    a = Tensor("A", (IntConst(m), IntConst(k)))
    b = Tensor("B", (IntConst(k), IntConst(n)))
    c = Tensor("C", (IntConst(m), IntConst(n)))

    compute = Compute(
        name="S",
        domain=Domain(
            params=(),
            iterators=(
                Iterator("i"),
                Iterator("j"),
                Iterator("k", kind="reduce"),
            ),
            constraints=(
                Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
                Compare(lhs=Var("i"), op="LT", rhs=IntConst(m)),
                Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
                Compare(lhs=Var("j"), op="LT", rhs=IntConst(n)),
                Compare(lhs=IntConst(0), op="LE", rhs=Var("k")),
                Compare(lhs=Var("k"), op="LT", rhs=IntConst(k)),
            ),
        ),
        body=ReduceStore(
            op="Sum",
            access=Access(tensor=c, index=(Var("i"), Var("j"))),
            value=BinaryOp(
                op="Mul",
                lhs=Load(access=Access(tensor=a, index=(Var("i"), Var("k")))),
                rhs=Load(access=Access(tensor=b, index=(Var("k"), Var("j")))),
            ),
            init=IntConst(0),
        ),
    )

    func = PrimFunc(
        name="gemm",
        computes=(compute,),
        schedule=Schedule(("i", "j", "k")),
        params=(a, b, c),
    )

    c_code = isl_ast_to_c(ast, func)

    expected = """\
void gemm(int *A, int *B, int *C) {
    for (int c0 = 0; c0 <= 1; c0++) {
        for (int c1 = 0; c1 <= 2; c1++) {
            for (int c2 = 0; c2 <= 3; c2++) {
                if (c2 == 0) C[(c0*3 + c1)] = 0;
                C[(c0*3 + c1)] += A[(c0*4 + c2)] * B[(c2*3 + c1)];
            }
        }
    }
}"""

    assert c_code == expected
