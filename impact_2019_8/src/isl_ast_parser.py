from __future__ import annotations

from dataclasses import dataclass

from ast_types import BinOp, Block, Body, Call, Expr, ForLoop, Guard, Id, UnaryOp, User, Val
from isl_ast_lexer import Token, TokenType, tokenize

AstResult = ForLoop | Block


@dataclass
class AstParser:
    """再帰下降パーサー."""

    tokens: list[Token]
    pos: int = 0

    def parse(self) -> AstResult:
        """トークン列をパースしてForLoopまたはBlockを返す."""
        # トップレベルがリスト [ ... ] の場合はBlock
        if self._peek(TokenType.LBRACKET):
            return self._parse_block()
        return self._parse_for_loop()

    def _parse_block(self) -> Block:
        """ForLoopのリストをBlockとしてパースする（入れ子も処理）."""
        self._expect(TokenType.LBRACKET)
        loops: list[ForLoop] = []

        if not self._peek(TokenType.RBRACKET):
            loops.extend(self._parse_block_element())
            while self._peek(TokenType.COMMA):
                self._expect(TokenType.COMMA)
                loops.extend(self._parse_block_element())

        self._expect(TokenType.RBRACKET)
        return Block(stmts=tuple(loops))

    def _parse_block_element(self) -> list[ForLoop]:
        """ブロック要素をパースする（ForLoopまたは入れ子Block）."""
        if self._peek(TokenType.LBRACKET):
            # 入れ子のBlock - 展開してリストに追加
            nested_block = self._parse_block()
            return list(nested_block.stmts)
        else:
            return [self._parse_for_loop()]

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
            elif len(args) == 1:
                # 単項演算子（minus など）
                return UnaryOp(op=op, operand=args[0])
            elif len(args) == 2:
                # 二項演算子
                return BinOp(op=op, left=args[0], right=args[1])
            else:
                raise ValueError(f"Unexpected number of args: {len(args)} for op {op}")
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
        # bodyが配列の場合（融合されたループで複数ステートメントがある場合）
        if self._peek(TokenType.LBRACKET):
            return self._parse_body_list()

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
        elif key == "guard":
            # 条件付き実行（if文）
            cond_expr = self._parse_expr()
            if not isinstance(cond_expr, BinOp):
                raise ValueError("guard condition must be a BinOp")
            self._expect(TokenType.COMMA)
            self._expect_ident("then")
            self._expect(TokenType.COLON)
            then_body = self._parse_body()
            self._expect(TokenType.RBRACE)
            return Guard(cond=cond_expr, then=then_body)
        else:
            raise ValueError(f"Unknown body key: {key}")

    def _parse_body_list(self) -> Block:
        """body配列をBlockとしてパースする（融合されたループ用）."""
        self._expect(TokenType.LBRACKET)
        bodies: list[Body] = []

        if not self._peek(TokenType.RBRACKET):
            bodies.append(self._parse_body())
            while self._peek(TokenType.COMMA):
                self._expect(TokenType.COMMA)
                bodies.append(self._parse_body())

        self._expect(TokenType.RBRACKET)
        return Block(stmts=tuple(bodies))

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


def parse_isl_ast(ast_str: str) -> AstResult:
    """ISL AST文字列をパースする."""
    tokens = tokenize(ast_str)
    parser = AstParser(tokens=tokens)
    return parser.parse()
