from __future__ import annotations

from dataclasses import dataclass

from ast_types import BinOp, Body, Call, Expr, ForLoop, Id, User, Val
from isl_ast_lexer import Token, TokenType, tokenize


@dataclass
class AstParser:
    """再帰下降パーサー."""

    tokens: list[Token]
    pos: int = 0

    def parse(self) -> ForLoop:
        """トークン列をパースしてForLoopを返す."""
        return self._parse_for_loop()

    def _current(self) -> Token:
        """現在のトークンを取得する."""
        return self.tokens[self.pos]

    def _peek(self, token_type: TokenType) -> bool:
        """現在のトークンが指定した型かどうかを確認する."""
        return self._current().type == token_type

    def _expect(self, token_type: TokenType) -> Token:
        """指定した型のトークンを消費する."""
        token = self._current()
        if token.type != token_type:
            raise ValueError(
                f"Expected {token_type}, got {token.type} at position {token.pos}"
            )
        self.pos += 1
        return token

    def _expect_ident(self, expected: str | None = None) -> str:
        """識別子トークンを消費する."""
        token = self._expect(TokenType.IDENT)
        name = str(token.value)
        if expected is not None and name != expected:
            raise ValueError(
                f"Expected '{expected}', got '{name}' at position {token.pos}"
            )
        return name

    def _parse_expr(self) -> Expr:
        """式をパースする."""
        self._expect(TokenType.LBRACE)
        key = self._expect_ident()
        self._expect(TokenType.COLON)

        if key == "id":
            name = self._expect_ident()
            self._expect(TokenType.RBRACE)
            return Id(name=name)
        elif key == "val":
            token = self._expect(TokenType.INT)
            value = int(token.value)  # type: ignore[arg-type]
            self._expect(TokenType.RBRACE)
            return Val(value=value)
        elif key == "op":
            op = self._expect_ident()
            self._expect(TokenType.COMMA)
            self._expect_ident("args")
            self._expect(TokenType.COLON)
            args = self._parse_expr_list()
            self._expect(TokenType.RBRACE)

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
        self._expect(TokenType.LBRACKET)
        exprs: list[Expr] = []

        if not self._peek(TokenType.RBRACKET):
            exprs.append(self._parse_expr())
            while self._peek(TokenType.COMMA):
                self._expect(TokenType.COMMA)
                exprs.append(self._parse_expr())

        self._expect(TokenType.RBRACKET)
        return exprs

    def _parse_body(self) -> Body:
        """bodyをパースする."""
        # bodyの開始位置を保存（ネストしたForLoopの場合に巻き戻すため）
        body_start = self.pos

        self._expect(TokenType.LBRACE)
        key = self._expect_ident()
        self._expect(TokenType.COLON)

        if key == "user":
            expr = self._parse_expr()
            if not isinstance(expr, Call):
                raise ValueError("user body must contain a Call expression")
            self._expect(TokenType.RBRACE)
            return User(expr=expr)
        elif key == "iterator":
            # ネストされたForLoop - 開始位置に巻き戻し
            self.pos = body_start
            return self._parse_for_loop()
        else:
            raise ValueError(f"Unknown body key: {key}")

    def _parse_for_loop(self) -> ForLoop:
        """ForLoopをパースする."""
        self._expect(TokenType.LBRACE)

        # iterator
        self._expect_ident("iterator")
        self._expect(TokenType.COLON)
        iterator_expr = self._parse_expr()
        if not isinstance(iterator_expr, Id):
            raise ValueError("iterator must be an Id")
        self._expect(TokenType.COMMA)

        # init
        self._expect_ident("init")
        self._expect(TokenType.COLON)
        init_expr = self._parse_expr()
        if not isinstance(init_expr, Val):
            raise ValueError("init must be a Val")
        self._expect(TokenType.COMMA)

        # cond
        self._expect_ident("cond")
        self._expect(TokenType.COLON)
        cond_expr = self._parse_expr()
        if not isinstance(cond_expr, BinOp):
            raise ValueError("cond must be a BinOp")
        self._expect(TokenType.COMMA)

        # inc
        self._expect_ident("inc")
        self._expect(TokenType.COLON)
        inc_expr = self._parse_expr()
        if not isinstance(inc_expr, Val):
            raise ValueError("inc must be a Val")
        self._expect(TokenType.COMMA)

        # body
        self._expect_ident("body")
        self._expect(TokenType.COLON)
        body = self._parse_body()
        self._expect(TokenType.RBRACE)

        return ForLoop(
            iterator=iterator_expr,
            init=init_expr,
            cond=cond_expr,
            inc=inc_expr,
            body=body,
        )


def parse_isl_ast(ast_str: str) -> ForLoop:
    """ISL AST文字列をパースする."""
    tokens = tokenize(ast_str)
    parser = AstParser(tokens=tokens)
    return parser.parse()
