"""reductionコード生成のテスト."""

from codegen import isl_ast_to_c
from ir_types import (
    Axis,
    BinaryOp,
    Compute,
    Const,
    Domain,
    Load,
    PrimFunc,
    ReduceStore,
    Schedule,
    Tensor,
)
from isl_ast_parser import parse_isl_ast


def test_codegen_gemm_like_reduction() -> None:
    ast_str = (
        "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: "
        "[ { id: c0 }, { val: 1 } ] }, inc: { val: 1 }, body: { iterator: "
        "{ id: c1 }, init: { val: 0 }, cond: { op: le, args: [ { id: c1 }, "
        "{ val: 2 } ] }, inc: { val: 1 }, body: { iterator: { id: c2 }, init: "
        "{ val: 0 }, cond: { op: le, args: [ { id: c2 }, { val: 3 } ] }, inc: "
        "{ val: 1 }, body: { user: { op: call, args: [ { id: S }, { id: c0 }, "
        "{ id: c1 }, { id: c2 } ] } } } } } }"
    )

    ast = parse_isl_ast(ast_str)

    m, n, k = 2, 3, 4
    a = Tensor("A", (m, k))
    b = Tensor("B", (k, n))
    c = Tensor("C", (m, n))

    compute = Compute(
        name="S",
        domain=Domain(
            (
                Axis("i", m),
                Axis("j", n),
                Axis("k", k, kind="reduce"),
            )
        ),
        stmt=ReduceStore(
            op="add",
            target=c,
            index=("i", "j"),
            value=BinaryOp(
                op="mul",
                left=Load(a, ("i", "k")),
                right=Load(b, ("k", "j")),
            ),
            init=Const(0),
        ),
    )

    func = PrimFunc(
        name="gemm",
        compute=compute,
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
