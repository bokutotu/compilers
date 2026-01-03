"""reductionコード生成のテスト."""

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
    ReduceStore,
    Schedule,
    Tensor,
    Var,
)
from isl_ast_converter import convert_ast_node


def test_codegen_gemm_like_reduction() -> None:
    # ISL で AST を生成 (3重ネストループ: i=0..1, j=0..2, k=0..3)
    ctx = isl.Context()
    domain_str = "{ S[i, j, k] : 0 <= i <= 1 and 0 <= j <= 2 and 0 <= k <= 3 }"
    domain = isl.UnionSet(domain_str)
    schedule = isl.UnionMap("{ S[i, j, k] -> [i, j, k] }")
    schedule = schedule.intersect_domain(domain)
    build = isl.AstBuild.alloc(ctx)
    isl_ast = build.node_from_schedule_map(schedule)

    ast = convert_ast_node(isl_ast)

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
