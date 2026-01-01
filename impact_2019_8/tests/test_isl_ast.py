"""isl_astモジュールのテスト."""

from src.isl_ast import generate_simple_loop_ast


def test_generate_simple_loop_ast():
    """1重ループのAST生成をテストする."""
    ast = generate_simple_loop_ast(n=10)

    isl_str = str(ast)
    expected = (
        "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: "
        "[ { id: c0 }, { val: 9 } ] }, inc: { val: 1 }, body: { user: "
        "{ op: call, args: [ { id: S }, { id: c0 } ] } } }"
    )

    assert isl_str == expected
