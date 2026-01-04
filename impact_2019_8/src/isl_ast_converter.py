"""isl.AstNode から ast_types への変換モジュール."""

from __future__ import annotations

import islpy as isl

from ast_types import (
    BinOp,
    Block,
    Body,
    Call,
    Expr,
    ForLoop,
    Guard,
    Id,
    UnaryOp,
    User,
    Val,
)

AstResult = ForLoop | Block

# isl.ast_expr_op_type から文字列へのマッピング
_OP_TYPE_MAP: dict[isl.ast_expr_op_type, str] = {
    isl.ast_expr_op_type.le: "le",
    isl.ast_expr_op_type.lt: "lt",
    isl.ast_expr_op_type.ge: "ge",
    isl.ast_expr_op_type.gt: "gt",
    isl.ast_expr_op_type.eq: "eq",
    isl.ast_expr_op_type.add: "add",
    isl.ast_expr_op_type.sub: "sub",
    isl.ast_expr_op_type.mul: "mul",
    isl.ast_expr_op_type.div: "div",
    isl.ast_expr_op_type.pdiv_q: "pdiv_q",
    isl.ast_expr_op_type.pdiv_r: "pdiv_r",
    isl.ast_expr_op_type.fdiv_q: "fdiv_q",
    isl.ast_expr_op_type.zdiv_r: "zdiv_r",
    isl.ast_expr_op_type.min: "min",
    isl.ast_expr_op_type.max: "max",
    isl.ast_expr_op_type.minus: "minus",
    isl.ast_expr_op_type.and_: "and",
    isl.ast_expr_op_type.or_: "or",
    isl.ast_expr_op_type.and_then: "and_then",
    isl.ast_expr_op_type.or_else: "or_else",
    isl.ast_expr_op_type.cond: "cond",
    isl.ast_expr_op_type.select: "select",
    isl.ast_expr_op_type.access: "access",
    isl.ast_expr_op_type.member: "member",
    isl.ast_expr_op_type.address_of: "address_of",
    isl.ast_expr_op_type.call: "call",
}


def convert_ast_node(node: isl.AstNode) -> AstResult:
    """isl.AstNode を ast_types に変換する."""
    node_type = node.get_type()

    if node_type == isl.ast_node_type.for_:
        return _convert_for_loop(node)
    elif node_type == isl.ast_node_type.block:
        return _convert_block(node)
    elif node_type == isl.ast_node_type.user:
        user = _convert_user(node)
        return Block(stmts=(user,))
    elif node_type == isl.ast_node_type.if_:
        guard = _convert_guard(node)
        return Block(stmts=(guard,))
    else:
        raise ValueError(f"Unsupported node type: {node_type}")


def _convert_for_loop(node: isl.AstNode) -> ForLoop:
    """for ループノードを変換する."""
    iterator_expr = _convert_expr(node.for_get_iterator())
    if not isinstance(iterator_expr, Id):
        raise ValueError("iterator must be an Id")

    init_expr = _convert_expr(node.for_get_init())

    cond_expr = _convert_expr(node.for_get_cond())
    if not isinstance(cond_expr, BinOp):
        raise ValueError("cond must be a BinOp")

    inc_expr = _convert_expr(node.for_get_inc())

    body = _convert_body(node.for_get_body())

    return ForLoop(
        iterator=iterator_expr,
        init=init_expr,
        cond=cond_expr,
        inc=inc_expr,
        body=body,
    )


def _convert_body(node: isl.AstNode) -> Body:
    """body ノードを変換する."""
    node_type = node.get_type()

    if node_type == isl.ast_node_type.for_:
        return _convert_for_loop(node)
    elif node_type == isl.ast_node_type.user:
        return _convert_user(node)
    elif node_type == isl.ast_node_type.block:
        return _convert_block(node)
    elif node_type == isl.ast_node_type.if_:
        return _convert_guard(node)
    else:
        raise ValueError(f"Unsupported body node type: {node_type}")


def _convert_block(node: isl.AstNode) -> Block:
    """block ノードを変換する."""
    children = node.block_get_children()
    n = children.n_ast_node()
    stmts: list[Body] = []

    for i in range(n):
        child = children.get_at(i)
        stmts.append(_convert_body(child))

    return Block(stmts=tuple(stmts))


def _convert_user(node: isl.AstNode) -> User:
    """user ノードを変換する."""
    expr = _convert_expr(node.user_get_expr())
    if not isinstance(expr, Call):
        raise ValueError("user body must contain a Call expression")
    return User(expr=expr)


def _convert_guard(node: isl.AstNode) -> Guard:
    """if ノードを Guard として変換する."""
    cond_expr = _convert_expr(node.if_get_cond())
    if not isinstance(cond_expr, BinOp):
        raise ValueError("guard condition must be a BinOp")

    then_body = _convert_body(node.if_get_then_node())

    return Guard(cond=cond_expr, then=then_body)


def _convert_expr(expr: isl.AstExpr) -> Expr:
    """isl.AstExpr を ast_types.Expr に変換する."""
    expr_type = expr.get_type()

    if expr_type == isl.ast_expr_type.id:
        id_obj = expr.get_id()
        return Id(name=id_obj.get_name())

    elif expr_type == isl.ast_expr_type.int:
        val_obj = expr.get_val()
        return Val(value=val_obj.get_num_si())

    elif expr_type == isl.ast_expr_type.op:
        return _convert_op_expr(expr)

    else:
        raise ValueError(f"Unsupported expression type: {expr_type}")


def _convert_op_expr(expr: isl.AstExpr) -> Expr:
    """演算子式を変換する."""
    op_type = expr.get_op_type()
    n_arg = expr.get_op_n_arg()

    # 演算子名を取得
    if op_type not in _OP_TYPE_MAP:
        raise ValueError(f"Unsupported operator type: {op_type}")
    op_name = _OP_TYPE_MAP[op_type]

    # 引数を変換
    args = [_convert_expr(expr.get_op_arg(i)) for i in range(n_arg)]

    # call は特別扱い
    if op_type == isl.ast_expr_op_type.call:
        return Call(args=args)

    # 単項演算子
    if n_arg == 1:
        return UnaryOp(op=op_name, operand=args[0])

    # 二項演算子
    if n_arg == 2:
        return BinOp(op=op_name, left=args[0], right=args[1])

    # 多引数のmax/minはネストした二項演算に変換
    if n_arg > 2 and op_name in ("max", "min"):
        result = args[0]
        for arg in args[1:]:
            result = BinOp(op=op_name, left=result, right=arg)
        return result

    raise ValueError(f"Unexpected number of args: {n_arg} for op {op_name}")
