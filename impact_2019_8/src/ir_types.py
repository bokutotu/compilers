from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias, Union

# 算術・準アフィン演算
# ISLで重要な floor, mod, min, max を含める
BinOpKind: TypeAlias = Literal[
    "Add",
    "Sub",
    "Mul",
    "Div",  # +, -, *, /
    "FloorDiv",
    "Mod",  # //, %  (ISL: floor(x/y), x%y)
    "Max",
    "Min",  # max, min
]

UnaryOpKind: TypeAlias = Literal["Neg", "Not"]

# 比較演算
CompareOpKind: TypeAlias = Literal["LT", "LE", "GT", "GE", "EQ", "NE"]

# 論理演算（制約の結合用）
LogicOpKind: TypeAlias = Literal["And", "Or"]

AxisKind: TypeAlias = Literal["spatial", "reduce"]
ReduceOpKind: TypeAlias = Literal["Sum", "Prod", "Max", "Min"]


@dataclass(frozen=True)
class Expr:
    """式の基底クラス"""

    pass


@dataclass(frozen=True)
class IntConst(Expr):
    """整数定数"""

    value: int


@dataclass(frozen=True)
class FloatConst(Expr):
    """浮動小数点定数 (計算本体のRHS等で使用)"""

    value: float


@dataclass(frozen=True)
class Var(Expr):
    """変数参照 (イテレータ変数 または パラメータ)"""

    name: str


@dataclass(frozen=True)
class BinaryOp(Expr):
    """二項演算: lhs op rhs
    例: i + 1, i % 2, min(N, M)
    """

    op: BinOpKind
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    """単項演算: op operand"""

    op: UnaryOpKind
    operand: Expr


@dataclass(frozen=True)
class Call(Expr):
    """関数呼び出し
    ISLの特殊関数 (floor, ceil) や外部関数呼び出し用
    """

    name: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Constraint:
    """制約の基底クラス"""

    pass


@dataclass(frozen=True)
class Compare(Constraint):
    """比較制約: lhs op rhs
    例: i < N
    """

    lhs: Expr
    op: CompareOpKind
    rhs: Expr


@dataclass(frozen=True)
class Logical(Constraint):
    """論理結合: lhs op rhs
    例: (0 <= i) and (i < N)
    """

    op: LogicOpKind
    lhs: Constraint
    rhs: Constraint


@dataclass(frozen=True)
class Iterator:
    """ループ変数の定義"""

    name: str
    kind: AxisKind = "spatial"


@dataclass(frozen=True)
class Domain:
    """
    反復空間定義
    """

    # シンボリックパラメータ (例: N, M, K)
    params: tuple[str, ...]

    # イテレータ変数 (例: i, j)
    # ここで定義された順序がSetの次元順序になります
    iterators: tuple[Iterator, ...]

    # 制約条件のリスト
    # 複数の制約は暗黙的に AND で結合されますが、
    # Logicalクラスを使って複雑な条件(ORなど)も記述可能です
    constraints: tuple[Constraint, ...]


# ==========================================
# 5. テンソルとアクセス (Access)
# ==========================================

# 形状やインデックスも「式」で表現する
# これにより A[i + 1, 2 * j] のようなアクセスが可能になる
Shape: TypeAlias = tuple[Expr, ...]
Index: TypeAlias = tuple[Expr, ...]


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: Shape
    dtype: str = "float32"


@dataclass(frozen=True)
class Access:
    """テンソルアクセス共通構造"""

    tensor: Tensor
    index: Index


Stmt: TypeAlias = Union["Store", "ReduceStore", "Block"]


@dataclass(frozen=True)
class Load(Expr):
    """ロード式 (式の一部として埋め込まれる)"""

    access: Access


@dataclass(frozen=True)
class Store:
    """ストア文"""

    access: Access
    value: Expr
    # 条件付き実行 (if文)
    predicate: Constraint | None = None


@dataclass(frozen=True)
class ReduceStore:
    """リダクション文: A[i] += value"""

    op: ReduceOpKind
    access: Access
    value: Expr
    init: Expr | None = None


@dataclass(frozen=True)
class Block:
    """文のブロック"""

    stmts: tuple[Stmt, ...]


@dataclass(frozen=True)
class Compute:
    name: str
    domain: Domain
    body: Stmt


@dataclass(frozen=True)
class Schedule:
    """スケジューリング情報"""

    loop_order: tuple[str, ...]


@dataclass(frozen=True)
class PrimFunc:
    name: str
    params: tuple[Tensor, ...]  # 入出力引数
    computes: tuple[Compute, ...]  # 計算ステージ
    schedule: Schedule
