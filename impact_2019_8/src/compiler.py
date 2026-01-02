"""PrimFuncからCコードを生成するコンパイラ."""

from src.ast_parser import parse_isl_ast
from src.codegen import isl_ast_to_c
from src.ir_to_isl import build_domain_and_schedule
from src.ir_types import PrimFunc
from src.isl_ast import build_ast_from_domain_and_schedule


def compile(func: PrimFunc) -> str:
    """PrimFuncからCコードを生成する.

    Args:
        func: コンパイル対象のPrimFunc

    Returns:
        生成されたCコード文字列
    """
    # 1. ドメインとスケジュールからISL ASTを生成
    isl_domain, isl_schedule = build_domain_and_schedule(func)
    ast = build_ast_from_domain_and_schedule(isl_domain, isl_schedule)

    # 2. AST文字列をパースしてForLoopオブジェクトに変換
    parsed_ast = parse_isl_ast(str(ast))

    # 3. ForLoopからCコードを生成
    c_code = isl_ast_to_c(parsed_ast, func)

    return c_code
