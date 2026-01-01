from isl_ast import generate_simple_loop_ast
from ast_parser import parse_isl_ast
from isl_ast_to_c import isl_ast_to_c


def main():
    # 1. ISL ASTを生成
    ast = generate_simple_loop_ast(n=10)

    # 2. AST文字列をパースしてForLoopオブジェクトに変換
    parsed_ast = parse_isl_ast(str(ast))

    # 3. ForLoopからCコードを生成
    c_code = isl_ast_to_c(parsed_ast)

    # 4. Cコードを出力
    print(c_code)


if __name__ == "__main__":
    main()
