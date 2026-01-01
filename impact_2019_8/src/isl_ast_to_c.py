"""ISL ASTからC言語コードを生成するモジュール."""

import re


def isl_ast_to_c(ast_str: str) -> str:
    """ISL AST文字列からC言語コードを生成する."""
    init_val = int(re.search(r"init: \{ val: (\d+) \}", ast_str).group(1))
    bound_val = int(re.search(r"cond:.*?args: \[.*?\{ val: (\d+) \}", ast_str).group(1))
    inc_val = int(re.search(r"inc: \{ val: (\d+) \}", ast_str).group(1))

    return f"""int main() {{
    for (int i = {init_val}; i <= {bound_val}; i += {inc_val}) {{
    }}
    return 0;
}}"""
