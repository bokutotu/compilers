from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Extent: TypeAlias = int | str
Shape: TypeAlias = tuple[Extent, ...]
Index: TypeAlias = tuple[str, ...]

AxisKind: TypeAlias = Literal["spatial", "reduce"]

BinOpKind: TypeAlias = Literal["add", "sub", "mul", "div"]
ReduceOpKind: TypeAlias = Literal["add", "mul", "max", "min"]

OpKind: TypeAlias = BinOpKind


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: Shape


@dataclass(frozen=True)
class Axis:
    name: str
    extent: Extent
    lower: int | str = 0
    kind: AxisKind = "spatial"


@dataclass(frozen=True)
class Domain:
    axis: tuple[Axis, ...]


@dataclass(frozen=True)
class Schedule:
    loop_order: tuple[str, ...]


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Load:
    tensor: Tensor
    index: Index


@dataclass(frozen=True)
class BinaryOp:
    op: BinOpKind
    left: Expr
    right: Expr


Expr: TypeAlias = Const | Load | BinaryOp


@dataclass(frozen=True)
class Store:
    target: Tensor
    index: Index
    value: Expr


@dataclass(frozen=True)
class ReduceStore:
    op: ReduceOpKind
    target: Tensor
    index: Index
    value: Expr
    init: Expr | None = None


Stmt: TypeAlias = Store | ReduceStore


@dataclass(frozen=True)
class Compute:
    name: str
    domain: Domain
    stmt: Stmt


@dataclass(frozen=True)
class PrimFunc:
    name: str
    compute: Compute
    schedule: Schedule
    params: tuple[Tensor, ...]
