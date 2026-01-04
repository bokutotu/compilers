"""ForLoopからC言語コードを生成するクラス."""

from __future__ import annotations

from ast_types import Block, Body, Call, ForLoop, Guard, Id, User, Val
from ir_types import Compute, PrimFunc

from .expr import generate_cond, generate_expr
from .ops import generate_user_stmt

AstInput = ForLoop | Block


class CCodeGenerator:
    """ForLoopからC言語コードを生成するクラス."""

    def __init__(self, func: PrimFunc) -> None:
        self._func = func
        self._indent_level = 0
        # Compute名からComputeへのマッピングを作成
        self._computes: dict[str, Compute] = {c.name: c for c in func.computes}

    def generate(self, ast: AstInput) -> str:
        """ForLoopまたはBlockからC言語コードを生成する."""
        args = [tensor.name for tensor in self._func.params]
        args_str = ", ".join(f"int *{name}" for name in args) if args else "void"
        lines = [
            f"void {self._func.name}({args_str}) {{",
        ]
        self._indent_level = 1
        if isinstance(ast, Block):
            lines.append(self._generate_block(ast))
        else:
            lines.append(self._generate_for_loop(ast))
        lines.append("}")
        return "\n".join(lines)

    def _generate_block(self, block: Block) -> str:
        """Blockを複数の文に変換する（ForLoop, User, または入れ子Block）."""
        stmt_strs = [self._generate_body(stmt) for stmt in block.stmts]
        return "\n".join(s for s in stmt_strs if s)

    def _indent(self) -> str:
        """現在のインデントを返す."""
        return "    " * self._indent_level

    def _generate_for_loop(self, loop: ForLoop) -> str:
        """ForLoopをC言語のforループに変換する."""
        iterator = loop.iterator.name
        init_str = generate_expr(loop.init)
        cond_str = generate_cond(loop.cond)
        inc_expr = loop.inc

        lines = []
        indent = self._indent()

        if isinstance(inc_expr, Val) and inc_expr.value == 1:
            inc_str = f"{iterator}++"
        else:
            inc_str = f"{iterator} += {generate_expr(inc_expr)}"

        lines.append(
            f"{indent}for (int {iterator} = {init_str}; {cond_str}; {inc_str}) {{"
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
            # Call argsの最初の要素がCompute名
            call = body.expr
            if not isinstance(call, Call) or not call.args:
                raise ValueError("User body must contain a Call with args")
            first_arg = call.args[0]
            if not isinstance(first_arg, Id):
                raise ValueError("First arg of Call must be an Id (compute name)")
            compute_name = first_arg.name
            compute = self._computes.get(compute_name)
            if compute is None:
                raise ValueError(f"Unknown compute: {compute_name}")
            stmt = generate_user_stmt(call, compute)
            if not stmt:
                return ""
            lines = stmt.splitlines()
            return "\n".join(f"{indent}{line}" for line in lines)
        elif isinstance(body, ForLoop):
            return self._generate_for_loop(body)
        elif isinstance(body, Block):
            return self._generate_block(body)
        elif isinstance(body, Guard):
            return self._generate_guard(body)
        else:
            raise ValueError(f"Unknown body type: {type(body)}")

    def _generate_guard(self, guard: Guard) -> str:
        """Guard（条件付き実行）をif文に変換する."""
        indent = self._indent()
        cond_str = generate_cond(guard.cond)

        lines = [f"{indent}if ({cond_str}) {{"]
        self._indent_level += 1
        body_str = self._generate_body(guard.then)
        if body_str:
            lines.append(body_str)
        self._indent_level -= 1
        lines.append(f"{indent}}}")
        return "\n".join(lines)


def isl_ast_to_c(ast: AstInput, func: PrimFunc) -> str:
    """ForLoopまたはBlockからC言語コードを生成する."""
    generator = CCodeGenerator(func)
    return generator.generate(ast)
