from codegen import isl_ast_to_c
from ir_to_isl import build_domain_and_schedule
from ir_types import PrimFunc
from isl_ast import build_ast_from_domain_and_schedule
from isl_ast_parser import parse_isl_ast


def compile(func: PrimFunc) -> str:
    isl_domain, isl_schedule = build_domain_and_schedule(func)
    ast = build_ast_from_domain_and_schedule(isl_domain, isl_schedule)

    parsed_ast = parse_isl_ast(str(ast))

    c_code = isl_ast_to_c(parsed_ast, func)

    return c_code
