"""ForLoopからC言語コードを生成するクラス."""

from __future__ import annotations

from collections.abc import Mapping

from ast_types import Body, ForLoop, User
from ir_types import MatrixOp

from .args import collect_function_args
from .expr import generate_cond
from .ops import generate_user_stmt


class CCodeGenerator:
    """ForLoopからC言語コードを生成するクラス."""

    def __init__(
        self, domain_exprs: Mapping[str, MatrixOp], function_name: str
    ) -> None:
        self._domain_exprs = dict(domain_exprs)
        self._function_name = function_name
        self._indent_level = 0

    def generate(self, ast: ForLoop) -> str:
        """ForLoopからC言語コードを生成する."""
        args = collect_function_args(ast, self._domain_exprs)
        args_str = ", ".join(f"int *{name}" for name in args) if args else "void"
        lines = [
            f"void {self._function_name}({args_str}) {{",
        ]
        self._indent_level = 1
        lines.append(self._generate_for_loop(ast))
        lines.append("}")
        return "\n".join(lines)

    def _indent(self) -> str:
        """現在のインデントを返す."""
        return "    " * self._indent_level

    def _generate_for_loop(self, loop: ForLoop) -> str:
        """ForLoopをC言語のforループに変換する."""
        iterator = loop.iterator.name
        init_val = loop.init.value
        cond_str = generate_cond(loop.cond)
        inc_val = loop.inc.value

        lines = []
        indent = self._indent()

        inc_str = f"{iterator}++" if inc_val == 1 else f"{iterator} += {inc_val}"

        lines.append(
            f"{indent}for (int {iterator} = {init_val}; {cond_str}; {inc_str}) {{"
        )

        self._indent_level += 1
        body_str = self._generate_body(loop.body)
        if body_str:
            lines.append(body_str)
        self._indent_level -= 1

        lines.append(f"{indent}}}")
        return "\n".join(lines)

    def _generate_body(self, body: Body) -> str:
        """bodyを生成する."""
        if isinstance(body, User):
            indent = self._indent()
            stmt = generate_user_stmt(body.expr, self._domain_exprs)
            return f"{indent}{stmt}"
        elif isinstance(body, ForLoop):
            return self._generate_for_loop(body)
        else:
            raise ValueError(f"Unknown body type: {type(body)}")


def isl_ast_to_c(ast: ForLoop, domain_exprs: Mapping[str, MatrixOp]) -> str:
    """ForLoopからC言語コードを生成する."""
    generator = CCodeGenerator(domain_exprs=domain_exprs, function_name="kernel")
    return generator.generate(ast)
