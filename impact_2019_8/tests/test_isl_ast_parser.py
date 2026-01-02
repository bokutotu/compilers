"""ast_parserのテスト."""

from ast_types import BinOp, Call, ForLoop, Id, User, Val
from isl_ast_parser import parse_isl_ast


def test_parse_simple_for_loop() -> None:
    """単純なforループをパースできる."""
    ast_str = "{ iterator: { id: c0 }, init: { val: 0 }, cond: { op: le, args: [ { id: c0 }, { val: 9 } ] }, inc: { val: 1 }, body: { user: { op: call, args: [ { id: S }, { id: c0 } ] } } }"  # noqa: E501

    result = parse_isl_ast(ast_str)

    expected = ForLoop(
        iterator=Id(name="c0"),
        init=Val(value=0),
        cond=BinOp(op="le", left=Id(name="c0"), right=Val(value=9)),
        inc=Val(value=1),
        body=User(expr=Call(args=[Id(name="S"), Id(name="c0")])),
    )

    assert result == expected
