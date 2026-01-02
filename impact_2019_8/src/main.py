from ast_parser import parse_isl_ast
from ir_to_isl import build_domain_and_schedule
from ir_types import Dim, MatrixOp, MatrixPtr
from isl_ast import build_ast_from_domain_and_schedule
from isl_ast_to_c import isl_ast_to_c


def main():
    # 1. IRを定義
    op = MatrixOp(
        name="S",
        op="add",
        left=MatrixPtr("A", dims=[Dim(10)]),
        right=MatrixPtr("B", dims=[Dim(10)]),
        out=MatrixPtr("C", dims=[Dim(10)]),
    )

    # 2. ドメインとスケジュールからISL ASTを生成
    domain, schedule = build_domain_and_schedule(op)
    ast = build_ast_from_domain_and_schedule(domain, schedule)

    # 3. AST文字列をパースしてForLoopオブジェクトに変換
    parsed_ast = parse_isl_ast(str(ast))

    # 4. ForLoopからCコードを生成
    c_code = isl_ast_to_c(parsed_ast, domain_exprs={op.name: op})

    # 5. Cコードを出力
    print(c_code)


if __name__ == "__main__":
    main()
