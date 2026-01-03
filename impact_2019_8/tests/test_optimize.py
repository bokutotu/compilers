"""optimizeモジュールのテスト."""

import islpy as isl
import pytest

from ir_types import (
    Access,
    BinaryOp,
    Compare,
    Compute,
    Domain,
    IntConst,
    Iterator,
    Load,
    PrimFunc,
    Schedule,
    Store,
    Tensor,
    Var,
)
from optimization_types import IllegalTilingError, Tile
from optimize import apply_tiling, check_tiling_legality


def test_apply_tiling_specific_axis():
    """指定した軸のみがタイル化されることを確認する."""
    ctx = isl.Context()
    a = Tensor("A", (IntConst(8), IntConst(8)))
    c = Tensor("C", (IntConst(8), IntConst(8)))

    domain = Domain(
        params=(),
        iterators=(Iterator("i"), Iterator("j")),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(8)),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=IntConst(8)),
        ),
    )
    schedule = Schedule(("i", "j"))

    compute = Compute(
        name="S",
        domain=domain,
        body=Store(
            access=Access(tensor=c, index=(Var("i"), Var("j"))),
            value=Load(access=Access(tensor=a, index=(Var("i"), Var("j")))),
        ),
    )

    func = PrimFunc(
        name="copy",
        params=(a, c),
        computes=(compute,),
        schedule=schedule,
    )

    tiled_schedule = apply_tiling(func, [Tile("j", 4)], ctx=ctx)

    expected_schedule = isl.Schedule.read_from_str(
        ctx,
        (
            '{ domain: "{ S[i, j] : 0 <= i <= 7 and 0 <= j <= 7 }", '
            'child: { schedule: "[{ S[i, j] -> [(i)] : 0 <= i <= 7 and 0 <= j <= 7 }, '
            '{ S[i, j] -> [(j - (j) mod 4)] : 0 <= i <= 7 and 0 <= j <= 7 }]", '
            'child: { schedule: "[{ S[i, j] -> [(0)] : 0 <= i <= 7 and 0 <= j <= 7 }, '
            '{ S[i, j] -> [((j) mod 4)] : 0 <= i <= 7 and 0 <= j <= 7 }]" } } }'
        ),
    )

    assert tiled_schedule.to_str() == expected_schedule.to_str()


def test_illegal_tiling_rejected():
    """負の依存距離がある軸のタイル化が拒否されることを確認する."""
    a = Tensor("A", (IntConst(4), IntConst(4)))

    domain = Domain(
        params=(),
        iterators=(Iterator("i"), Iterator("j")),
        constraints=(
            Compare(lhs=IntConst(1), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(4)),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=IntConst(3)),
        ),
    )
    schedule = Schedule(("i", "j"))

    compute = Compute(
        name="S",
        domain=domain,
        body=Store(
            access=Access(tensor=a, index=(Var("i"), Var("j"))),
            value=Load(
                access=Access(
                    tensor=a,
                    index=(
                        BinaryOp(op="Sub", lhs=Var("i"), rhs=IntConst(1)),
                        BinaryOp(op="Add", lhs=Var("j"), rhs=IntConst(1)),
                    ),
                )
            ),
        ),
    )

    func = PrimFunc(
        name="shift",
        params=(a,),
        computes=(compute,),
        schedule=schedule,
    )

    tiles = [Tile("j", 2)]
    is_legal, violations = check_tiling_legality(func, tiles)

    assert not is_legal
    assert any("Axis 'j'" in v for v in violations)

    with pytest.raises(IllegalTilingError):
        apply_tiling(func, tiles, check_legality=True)
