from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OpKind = Literal["add", "sub", "mul", "div"]
Extent = int | str
Shape = tuple[Extent, ...]
Index = tuple[str, ...]


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: Shape


@dataclass(frozen=True)
class Axis:
    name: str
    extent: Extent
    lower: int = 0


@dataclass(frozen=True)
class Domain:
    axis: tuple[Axis, ...]


@dataclass(frozen=True)
class Schedule:
    loop_order: tuple[str, ...]


@dataclass(frozen=True)
class Compute:
    name: str
    op: OpKind
    a: Tensor
    b: Tensor
    out: Tensor
    domain: Domain
    a_index: Index
    b_index: Index
    out_index: Index


@dataclass(frozen=True)
class PrimFunc:
    name: str
    compute: Compute
    schedule: Schedule
    params: tuple[Tensor, ...]
