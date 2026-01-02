"""ISL AST文字列の字句解析モジュール."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """トークンの種類."""

    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COLON = auto()  # :
    COMMA = auto()  # ,
    IDENT = auto()  # 識別子
    INT = auto()  # 整数
    EOF = auto()  # 終端


@dataclass(frozen=True)
class Token:
    """トークン."""

    type: TokenType
    value: str | int | None = None
    pos: int = 0


class AstLexer:
    """字句解析器."""

    def __init__(self, text: str) -> None:
        """初期化."""
        self.text = text
        self.pos = 0

    def _skip_whitespace(self) -> None:
        """空白をスキップする."""
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def tokenize(self) -> list[Token]:
        """テキストをトークン列に変換する."""
        tokens: list[Token] = []

        while self.pos < len(self.text):
            self._skip_whitespace()
            if self.pos >= len(self.text):
                break

            token = self._next_token()
            tokens.append(token)

        tokens.append(Token(type=TokenType.EOF, pos=self.pos))
        return tokens

    def _next_token(self) -> Token:
        """次のトークンを取得する."""
        char = self.text[self.pos]
        start_pos = self.pos

        # 単一文字トークン
        if char == "{":
            self.pos += 1
            return Token(type=TokenType.LBRACE, pos=start_pos)
        elif char == "}":
            self.pos += 1
            return Token(type=TokenType.RBRACE, pos=start_pos)
        elif char == "[":
            self.pos += 1
            return Token(type=TokenType.LBRACKET, pos=start_pos)
        elif char == "]":
            self.pos += 1
            return Token(type=TokenType.RBRACKET, pos=start_pos)
        elif char == ":":
            self.pos += 1
            return Token(type=TokenType.COLON, pos=start_pos)
        elif char == ",":
            self.pos += 1
            return Token(type=TokenType.COMMA, pos=start_pos)

        # 整数（負の数を含む）
        if char == "-" or char.isdigit():
            match = re.match(r"-?\d+", self.text[self.pos :])
            if match:
                self.pos += len(match.group())
                return Token(
                    type=TokenType.INT, value=int(match.group()), pos=start_pos
                )

        # 識別子
        if char.isalpha() or char == "_":
            match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", self.text[self.pos :])
            if match:
                self.pos += len(match.group())
                return Token(type=TokenType.IDENT, value=match.group(), pos=start_pos)

        raise ValueError(f"Unexpected character '{char}' at position {self.pos}")


def tokenize(text: str) -> list[Token]:
    """テキストをトークン列に変換する."""
    lexer = AstLexer(text)
    return lexer.tokenize()
