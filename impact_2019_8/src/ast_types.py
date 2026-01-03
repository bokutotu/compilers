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
Expr = Union["Id", "Val", "UnaryOp", "BinOp", "Call"]


@dataclass(frozen=True)
class UnaryOp:
    """単項演算（マイナスなど）."""

    op: str  # "minus" など
    operand: Expr


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


@dataclass(frozen=True)
class Guard:
    """条件付き実行（if文）."""

    cond: BinOp  # 条件式
    then: Body  # 条件が真の場合に実行される本体


Body = Union["User", "ForLoop", "Block", "Guard"]


@dataclass(frozen=True)
class ForLoop:
    """forループ."""

    iterator: Id
    init: Val
    cond: BinOp
    inc: Val
    body: Body


@dataclass(frozen=True)
class Block:
    """複数の文のシーケンス（ForLoop, User, または入れ子のBlockを含む）."""

    stmts: tuple[Body, ...]
