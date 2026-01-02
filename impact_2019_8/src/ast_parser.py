"""ISL AST文字列をパースするモジュール."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ast_types import BinOp, Body, Call, Expr, ForLoop, Id, User, Val


@dataclass
class Parser:
    """再帰下降パーサー."""

    text: str
    pos: int = 0

    def parse(self) -> ForLoop:
        """AST文字列をパースしてForLoopを返す."""
        return self._parse_for_loop()

    def _skip_whitespace(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def _expect(self, char: str) -> None:
        self._skip_whitespace()
        if self.pos >= len(self.text) or self.text[self.pos] != char:
            raise ValueError(f"Expected '{char}' at position {self.pos}")
        self.pos += 1

    def _peek(self) -> str | None:
        self._skip_whitespace()
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def _parse_identifier(self) -> str:
        """識別子をパースする."""
        self._skip_whitespace()
        match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", self.text[self.pos :])
        if not match:
            raise ValueError(f"Expected identifier at position {self.pos}")
        self.pos += len(match.group())
        return match.group()

    def _parse_integer(self) -> int:
        """整数をパースする."""
        self._skip_whitespace()
        match = re.match(r"-?\d+", self.text[self.pos :])
        if not match:
            raise ValueError(f"Expected integer at position {self.pos}")
        self.pos += len(match.group())
        return int(match.group())

    def _parse_expr(self) -> Expr:
        """式をパースする."""
        self._expect("{")
        key = self._parse_identifier()
        self._expect(":")

        if key == "id":
            name = self._parse_identifier()
            self._expect("}")
            return Id(name=name)
        elif key == "val":
            value = self._parse_integer()
            self._expect("}")
            return Val(value=value)
        elif key == "op":
            op = self._parse_identifier()
            self._expect(",")
            args_key = self._parse_identifier()
            if args_key != "args":
                raise ValueError(f"Expected 'args', got '{args_key}'")
            self._expect(":")
            args = self._parse_expr_list()
            self._expect("}")

            if op == "call":
                return Call(args=args)
            else:
                if len(args) != 2:
                    raise ValueError(f"BinOp expects 2 args, got {len(args)}")
                return BinOp(op=op, left=args[0], right=args[1])
        else:
            raise ValueError(f"Unknown expression key: {key}")

    def _parse_expr_list(self) -> list[Expr]:
        """式のリストをパースする."""
        self._expect("[")
        exprs: list[Expr] = []

        if self._peek() != "]":
            exprs.append(self._parse_expr())
            while self._peek() == ",":
                self._expect(",")
                exprs.append(self._parse_expr())

        self._expect("]")
        return exprs

    def _parse_body(self) -> Body:
        """bodyをパースする."""
        # bodyの開始位置を保存（ネストしたForLoopの場合に巻き戻すため）
        self._skip_whitespace()
        body_start = self.pos

        self._expect("{")
        key = self._parse_identifier()
        self._expect(":")

        if key == "user":
            expr = self._parse_expr()
            if not isinstance(expr, Call):
                raise ValueError("user body must contain a Call expression")
            self._expect("}")
            return User(expr=expr)
        elif key == "iterator":
            # ネストされたForLoop - 開始位置に巻き戻し
            self.pos = body_start
            return self._parse_for_loop()
        else:
            raise ValueError(f"Unknown body key: {key}")

    def _parse_for_loop(self) -> ForLoop:
        """ForLoopをパースする."""
        self._expect("{")

        # iterator
        key = self._parse_identifier()
        if key != "iterator":
            raise ValueError(f"Expected 'iterator', got '{key}'")
        self._expect(":")
        iterator_expr = self._parse_expr()
        if not isinstance(iterator_expr, Id):
            raise ValueError("iterator must be an Id")
        self._expect(",")

        # init
        key = self._parse_identifier()
        if key != "init":
            raise ValueError(f"Expected 'init', got '{key}'")
        self._expect(":")
        init_expr = self._parse_expr()
        if not isinstance(init_expr, Val):
            raise ValueError("init must be a Val")
        self._expect(",")

        # cond
        key = self._parse_identifier()
        if key != "cond":
            raise ValueError(f"Expected 'cond', got '{key}'")
        self._expect(":")
        cond_expr = self._parse_expr()
        if not isinstance(cond_expr, BinOp):
            raise ValueError("cond must be a BinOp")
        self._expect(",")

        # inc
        key = self._parse_identifier()
        if key != "inc":
            raise ValueError(f"Expected 'inc', got '{key}'")
        self._expect(":")
        inc_expr = self._parse_expr()
        if not isinstance(inc_expr, Val):
            raise ValueError("inc must be a Val")
        self._expect(",")

        # body
        key = self._parse_identifier()
        if key != "body":
            raise ValueError(f"Expected 'body', got '{key}'")
        self._expect(":")
        body = self._parse_body()
        self._expect("}")

        return ForLoop(
            iterator=iterator_expr,
            init=init_expr,
            cond=cond_expr,
            inc=inc_expr,
            body=body,
        )


def parse_isl_ast(ast_str: str) -> ForLoop:
    """ISL AST文字列をパースする."""
    parser = Parser(text=ast_str)
    return parser.parse()
