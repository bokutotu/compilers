"""Minimal IR dataclasses for matrix operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OpKind = Literal["add", "sub", "mul", "div"]
Extent = int | str


@dataclass(frozen=True)
class Dim:
    """Dimension extent."""

    extent: Extent


@dataclass(frozen=True)
class MatrixPtr:
    """Matrix pointer with shape."""

    name: str
    dims: list[Dim]


@dataclass(frozen=True)
class MatrixOp:
    """Binary matrix op: out = left (op) right with a statement name."""

    name: str
    op: OpKind
    left: MatrixPtr
    right: MatrixPtr
    out: MatrixPtr
