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
from optimize import (
    _apply_tile_to_band,
    apply_tiling,
    apply_tiling_to_schedule,
    check_tiling_legality,
    compute_optimized_schedule,
)


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

    tiled_schedule = apply_tiling(func, [Tile(1, 4)], ctx=ctx)  # j軸（インデックス1）

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


def test_apply_tiling_on_skewed_schedule():
    """skew済みスケジュールでもタイル化できることを確認する."""
    ctx = isl.Context()
    a = Tensor("A", (IntConst(4), IntConst(4)))
    c = Tensor("C", (IntConst(4), IntConst(4)))

    domain = Domain(
        params=(),
        iterators=(Iterator("i"), Iterator("j")),
        constraints=(
            Compare(lhs=IntConst(0), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(4)),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=IntConst(4)),
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
        name="skewed_copy",
        params=(a, c),
        computes=(compute,),
        schedule=schedule,
    )

    domain_isl = isl.UnionSet(
        "{ S[i, j] : 0 <= i < 4 and 0 <= j < 4 }",
        ctx,
    )
    schedule_map = isl.UnionMap(
        "{ S[i, j] -> [i, i + j] : 0 <= i < 4 and 0 <= j < 4 }",
        ctx,
    )
    mupa = isl.MultiUnionPwAff.from_union_map(schedule_map)

    schedule_tree = isl.Schedule.from_domain(domain_isl)
    node = schedule_tree.get_root().child(0)
    node = node.insert_partial_schedule(mupa)

    tiled_node = _apply_tile_to_band(func, node, [Tile(0, 2), Tile(1, 3)], ctx)  # i軸, j軸
    tiled_schedule = tiled_node.get_schedule()

    expected_schedule = isl.Schedule.read_from_str(
        ctx,
        (
            '{ domain: "{ S[i, j] : 0 <= i <= 3 and 0 <= j <= 3 }", '
            'child: { schedule: "[{ S[i, j] -> [(i - (i) mod 2)] : 0 <= i <= 3 and '
            "0 <= j <= 3 }, { S[i, j] -> [(i + j - (i + j) mod 3)] : 0 <= i <= 3 and "
            '0 <= j <= 3 }]", child: { schedule: "[{ S[i, j] -> [((i) mod 2)] : '
            "0 <= i <= 3 and 0 <= j <= 3 }, { S[i, j] -> [((i + j) mod 3)] : "
            '0 <= i <= 3 and 0 <= j <= 3 }]" } } }'
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

    tiles = [Tile(1, 2)]  # j軸（インデックス1）
    is_legal, violations = check_tiling_legality(func, tiles)

    assert not is_legal
    assert any("Axis 1" in v for v in violations)

    with pytest.raises(IllegalTilingError):
        apply_tiling(func, tiles, check_legality=True)


def test_compute_optimized_schedule_enables_tiling():
    """skewingが必要なケースでcompute_optimized_scheduleによりタイリング可能になることを確認."""
    ctx = isl.Context()
    a = Tensor("A", (IntConst(8), IntConst(8)))

    # A[i,j] = A[i-1, j+1] の依存パターン
    # j方向に負の依存距離(-1)があるため、元のスケジュールではj軸のタイリングが不可能
    domain = Domain(
        params=(),
        iterators=(Iterator("i"), Iterator("j")),
        constraints=(
            Compare(lhs=IntConst(1), op="LE", rhs=Var("i")),
            Compare(lhs=Var("i"), op="LT", rhs=IntConst(8)),
            Compare(lhs=IntConst(0), op="LE", rhs=Var("j")),
            Compare(lhs=Var("j"), op="LT", rhs=IntConst(7)),
        ),
    )

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
        name="skewed",
        params=(a,),
        computes=(compute,),
        schedule=Schedule(("i", "j")),
    )

    # 元のスケジュールではj軸のタイリングが不可能
    is_legal, _ = check_tiling_legality(func, [Tile(1, 2)])
    assert not is_legal

    # compute_optimized_scheduleでpermutableなスケジュールを取得
    optimized = compute_optimized_schedule(func, ctx)

    # タイリングを適用
    tiled = apply_tiling_to_schedule(optimized, [Tile(0, 2), Tile(1, 2)])

    # 期待されるタイル済みスケジュールをISL APIで構築
    # ISLはskewing後のスケジュールとして(i+j, i)を選択する
    domain_isl = isl.UnionSet("{ S[i, j] : 0 < i <= 7 and 0 <= j <= 6 }", ctx)
    schedule_map = isl.UnionMap("{ S[i, j] -> [i + j, i] }", ctx)
    mupa = isl.MultiUnionPwAff.from_union_map(schedule_map)
    expected_sched = isl.Schedule.from_domain(domain_isl)
    node = expected_sched.get_root().child(0)
    node = node.insert_partial_schedule(mupa)
    node = node.band_set_permutable(1)
    node = node.band_member_set_coincident(0, True)
    space = node.band_get_space()
    tile_sizes = isl.MultiVal.zero(space)
    tile_sizes = tile_sizes.set_val(0, isl.Val.int_from_si(ctx, 2))
    tile_sizes = tile_sizes.set_val(1, isl.Val.int_from_si(ctx, 2))
    node = node.band_tile(tile_sizes)
    expected_tiled = node.get_schedule()

    assert tiled.plain_is_equal(expected_tiled)
