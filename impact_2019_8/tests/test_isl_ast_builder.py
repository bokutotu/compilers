"""isl_astのビルダー関数のテスト."""

import islpy as isl

from isl_ast import build_ast_from_domain_and_schedule


def test_build_ast_from_domain_and_schedule():
    """ドメインとスケジュールからASTを生成できる."""
    ctx = isl.Context()
    domain = isl.UnionSet("{ S[i] : 0 <= i < 10 }", ctx)
    schedule = isl.UnionMap("{ S[i] -> [i] : 0 <= i < 10 }", ctx)

    ast = build_ast_from_domain_and_schedule(domain, schedule)
    isl_str = str(ast)

    expected = (
        "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: "
        "[ { id: c0 }, { val: 9 } ] }, inc: { val: 1 }, body: { user: "
        "{ op: call, args: [ { id: S }, { id: c0 } ] } } }"
    )

    assert isl_str == expected
