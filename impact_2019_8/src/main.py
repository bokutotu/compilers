from isl_ast import generate_simple_loop_ast
from isl_ast_to_c import isl_ast_to_c


def main():
    # ISL ASTを生成
    ast = generate_simple_loop_ast(n=10)

    # ASTからCコードを生成
    c_code = isl_ast_to_c(str(ast))

    # Cコードを出力
    print(c_code)


if __name__ == "__main__":
    main()
