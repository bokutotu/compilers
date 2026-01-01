"""isl_ast_to_cモジュールのテスト."""

from src.isl_ast_to_c import isl_ast_to_c


def test_isl_ast_to_c():
    """ISL AST文字列からC言語コードが生成されることをテストする."""
    ast_str = (
        "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: "
        "[ { id: c0 }, { val: 9 } ] }, inc: { val: 1 }, body: { user: "
        "{ op: call, args: [ { id: S }, { id: c0 } ] } } }"
    )

    c_code = isl_ast_to_c(ast_str)

    expected = """\
int main() {
    for (int i = 0; i <= 9; i += 1) {
    }
    return 0;
}"""

    assert c_code == expected
