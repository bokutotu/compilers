"""ISL ASTの代数的データ構造."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# 基本的な式の型
@dataclass(frozen=True)
class Id:
    """識別子."""

    name: str


@dataclass(frozen=True)
class Val:
    """整数値."""

    value: int


# 式の型（再帰的に定義）
Expr = Union["Id", "Val", "BinOp", "Call"]


@dataclass(frozen=True)
class BinOp:
    """二項演算（比較演算子など）."""

    op: str  # "le", "lt", "ge", "gt", "eq" など
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Call:
    """関数呼び出し."""

    args: list[Expr]


# 文の型
@dataclass(frozen=True)
class User:
    """ユーザー定義の文."""

    expr: Call


Body = Union["User", "ForLoop"]


@dataclass(frozen=True)
class ForLoop:
    """forループ."""

    iterator: Id
    init: Val
    cond: BinOp
    inc: Val
    body: Body
