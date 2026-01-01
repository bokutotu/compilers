"""ISL ASTからC言語コードを生成するモジュール."""

from __future__ import annotations

from ast_types import BinOp, Body, Call, Expr, ForLoop, Id, User, Val


class CCodeGenerator:
    """ForLoopからC言語コードを生成するクラス."""

    def __init__(self) -> None:
        self._indent_level = 0

    def generate(self, ast: ForLoop) -> str:
        """ForLoopからC言語コードを生成する."""
        lines = [
            "int main() {",
        ]
        self._indent_level = 1
        lines.append(self._generate_for_loop(ast))
        lines.append("    return 0;")
        lines.append("}")
        return "\n".join(lines)

    def _indent(self) -> str:
        """現在のインデントを返す."""
        return "    " * self._indent_level

    def _generate_for_loop(self, loop: ForLoop) -> str:
        """ForLoopをC言語のforループに変換する."""
        iterator = loop.iterator.name
        init_val = loop.init.value
        cond_str = self._generate_cond(loop.cond, iterator)
        inc_val = loop.inc.value

        lines = []
        indent = self._indent()

        # forループのヘッダー
        if inc_val == 1:
            inc_str = f"{iterator}++"
        else:
            inc_str = f"{iterator} += {inc_val}"

        lines.append(f"{indent}for (int {iterator} = {init_val}; {cond_str}; {inc_str}) {{")

        # body
        self._indent_level += 1
        body_str = self._generate_body(loop.body)
        if body_str:
            lines.append(body_str)
        self._indent_level -= 1

        lines.append(f"{indent}}}")
        return "\n".join(lines)

    def _generate_cond(self, cond: BinOp, iterator: str) -> str:
        """条件式を生成する."""
        op_map = {
            "le": "<=",
            "lt": "<",
            "ge": ">=",
            "gt": ">",
            "eq": "==",
        }
        op = op_map.get(cond.op, cond.op)
        left = self._generate_expr(cond.left)
        right = self._generate_expr(cond.right)
        return f"{left} {op} {right}"

    def _generate_expr(self, expr: Expr) -> str:
        """式を生成する."""
        if isinstance(expr, Id):
            return expr.name
        elif isinstance(expr, Val):
            return str(expr.value)
        elif isinstance(expr, BinOp):
            left = self._generate_expr(expr.left)
            right = self._generate_expr(expr.right)
            op_map = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "le": "<=",
                "lt": "<",
                "ge": ">=",
                "gt": ">",
                "eq": "==",
            }
            op = op_map.get(expr.op, expr.op)
            return f"({left} {op} {right})"
        elif isinstance(expr, Call):
            # Sは全て足し算として処理
            return self._generate_call_as_sum(expr)
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    def _generate_call_as_sum(self, call: Call) -> str:
        """Call（S関数）を足し算として生成する."""
        if not call.args:
            return "0"
        args_str = [self._generate_expr(arg) for arg in call.args]
        return " + ".join(args_str)

    def _generate_body(self, body: Body) -> str:
        """bodyを生成する."""
        if isinstance(body, User):
            indent = self._indent()
            sum_expr = self._generate_call_as_sum(body.expr)
            return f"{indent}S({sum_expr});"
        elif isinstance(body, ForLoop):
            return self._generate_for_loop(body)
        else:
            raise ValueError(f"Unknown body type: {type(body)}")


def isl_ast_to_c(ast: ForLoop) -> str:
    """ForLoopからC言語コードを生成する."""
    generator = CCodeGenerator()
    return generator.generate(ast)
