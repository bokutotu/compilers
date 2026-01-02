"""isl_ast_to_cモジュールのテスト."""

from src.ast_parser import parse_isl_ast
from src.codegen import isl_ast_to_c
from src.ir_types import Axis, Compute, Domain, Tensor


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
    compute = Compute(
        name="S",
        op="add",
        a=a,
        b=b,
        out=c,
        domain=domain,
        a_index=("i",),
        b_index=("i",),
        out_index=("i",),
    )
    c_code = isl_ast_to_c(ast, domain_exprs={compute.name: compute})

    expected = """\
void kernel(int *A, int *B, int *C) {
    for (int c0 = 0; c0 <= 9; c0++) {
        C[c0] = A[c0] + B[c0];
    }
}"""

    assert c_code == expected
