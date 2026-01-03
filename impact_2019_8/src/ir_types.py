from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Extent: TypeAlias = int | str
Shape: TypeAlias = tuple[Extent, ...]
Index: TypeAlias = tuple[str, ...]

AxisKind: TypeAlias = Literal["spatial", "reduce"]
ConstraintOp: TypeAlias = Literal["LE", "LT", "GE", "GT", "EQ", "NE"]

# 演算子からISL文字列へのマッピング
CONSTRAINT_OP_TO_ISL: dict[ConstraintOp, str] = {
    "LE": "<=",
    "LT": "<",
    "GE": ">=",
    "GT": ">",
    "EQ": "=",
    "NE": "!=",
}


@dataclass(frozen=True)
class AffineTerm:
    """アフィン項: coeff * var (varが空文字列の場合は定数項)"""

    coeff: int
    var: str = ""  # 変数名。空の場合は定数項として扱う

    def __neg__(self) -> AffineTerm:
        return AffineTerm(-self.coeff, self.var)

    def to_isl(self) -> str:
        if not self.var:
            return str(self.coeff)
        if self.coeff == 1:
            return self.var
        if self.coeff == -1:
            return f"-{self.var}"
        return f"{self.coeff}*{self.var}"


@dataclass(frozen=True)
class AffineExpr:
    """アフィン式: Σ(coeff * var) の形式"""

    terms: tuple[AffineTerm, ...]

    def to_isl(self) -> str:
        if not self.terms:
            return "0"
        parts: list[str] = []
        for i, term in enumerate(self.terms):
            s = term.to_isl()
            if i == 0:
                parts.append(s)
            elif s.startswith("-"):
                parts.append(f" - {s[1:]}")
            else:
                parts.append(f" + {s}")
        return "".join(parts)

    @staticmethod
    def from_var(var: str, coeff: int = 1) -> AffineExpr:
        """変数からアフィン式を作成"""
        return AffineExpr((AffineTerm(coeff, var),))

    @staticmethod
    def from_const(value: int) -> AffineExpr:
        """定数からアフィン式を作成"""
        return AffineExpr((AffineTerm(value, ""),))

    def __add__(self, other: AffineExpr | int | str) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.from_const(other)
        elif isinstance(other, str):
            other = AffineExpr.from_var(other)
        return AffineExpr(self.terms + other.terms)

    def __radd__(self, other: int | str) -> AffineExpr:
        if isinstance(other, int):
            return AffineExpr.from_const(other) + self
        return AffineExpr.from_var(other) + self

    def __sub__(self, other: AffineExpr | int | str) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.from_const(other)
        elif isinstance(other, str):
            other = AffineExpr.from_var(other)
        neg_terms = tuple(-t for t in other.terms)
        return AffineExpr(self.terms + neg_terms)

    def __rsub__(self, other: int | str) -> AffineExpr:
        if isinstance(other, int):
            return AffineExpr.from_const(other) - self
        return AffineExpr.from_var(other) - self

    def __neg__(self) -> AffineExpr:
        return AffineExpr(tuple(-t for t in self.terms))

    def __mul__(self, coeff: int) -> AffineExpr:
        return AffineExpr(tuple(AffineTerm(t.coeff * coeff, t.var) for t in self.terms))

    def __rmul__(self, coeff: int) -> AffineExpr:
        return self * coeff


@dataclass(frozen=True)
class AffineConstraint:
    """アフィン制約: lhs op rhs (例: i + j <= N)"""

    lhs: AffineExpr
    op: ConstraintOp
    rhs: AffineExpr

    def to_isl(self) -> str:
        op_str = CONSTRAINT_OP_TO_ISL[self.op]
        return f"{self.lhs.to_isl()} {op_str} {self.rhs.to_isl()}"

    def collect_params(self) -> list[str]:
        """シンボリックパラメータを収集（大文字の変数はパラメータとして扱う）"""
        params: list[str] = []
        for term in self.lhs.terms + self.rhs.terms:
            if term.var and term.var.isupper() and term.var not in params:
                params.append(term.var)
        return params

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
    constraints: tuple[AffineConstraint, ...] = ()  # 追加のアフィン制約


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
