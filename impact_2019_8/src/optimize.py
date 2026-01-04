from __future__ import annotations

import islpy as isl

from ir_to_isl import (
    build_domain,
    build_read_access,
    build_schedule,
    build_write_access,
    compute_raw_dependence,
    compute_war_dependence,
    compute_waw_dependence,
)
from ir_types import PrimFunc
from optimization_types import IllegalTilingError, Tile


def check_tiling_legality(
    func: PrimFunc,
    tiles: list[Tile],
    ctx: isl.Context | None = None,
) -> tuple[bool, list[str]]:
    """
    タイル化が合法かどうかをチェック

    タイル化が合法な条件:
    - タイル化する軸において、すべての依存距離が非負であること

    Args:
        func: 対象の関数
        tiles: タイル化の指定リスト
        ctx: ISLコンテキスト

    Returns:
        (is_legal, violations): 合法かどうかと、違反している依存関係のリスト
    """
    ctx = ctx or isl.Context()

    # スケジュールの軸数を取得
    n_axes = len(func.schedule.loop_order)

    # 軸インデックスの検証を先に行う
    tile_axes_indices = []
    for tile in tiles:
        if tile.axis >= n_axes:
            raise ValueError(f"Invalid axis index: {tile.axis}. Valid range: 0-{n_axes - 1}")
        tile_axes_indices.append(tile.axis)

    # スケジュールとアクセスを構築
    domain = build_domain(func, ctx)
    schedule_map = build_schedule(func, ctx).intersect_domain(domain)
    read_access = build_read_access(func, ctx)
    write_access = build_write_access(func, ctx)

    # すべての依存関係を計算
    raw_dep = compute_raw_dependence(schedule_map, write_access, read_access)
    war_dep = compute_war_dependence(schedule_map, write_access, read_access)
    waw_dep = compute_waw_dependence(schedule_map, write_access)

    # すべての依存関係を結合
    all_deps = raw_dep.union(war_dep).union(waw_dep)

    if all_deps.is_empty():
        return True, []

    violations = []

    # 依存距離を計算してチェック
    # 依存関係マップに対してスケジュールを適用し、時間空間での依存を得る
    # { S[i] -> S[i'] } を { [t] -> [t'] } に変換
    scheduled_deps = all_deps.apply_domain(schedule_map).apply_range(schedule_map)

    # 依存距離（delta）を計算: { [t' - t] }
    deltas = scheduled_deps.deltas()

    if deltas.is_empty():
        return True, []

    # 各タイル軸について、依存距離が非負かチェック
    for axis_idx in tile_axes_indices:
        if not _check_axis_non_negative(deltas, axis_idx):
            violations.append(f"Axis {axis_idx} has negative dependence distance")

    return len(violations) == 0, violations


def _check_axis_non_negative(deltas: isl.UnionSet, axis_idx: int) -> bool:
    """
    指定された軸の依存距離が非負かチェック

    Args:
        deltas: 依存距離の集合
        axis_idx: チェックする軸のインデックス

    Returns:
        すべての依存距離が非負ならTrue
    """
    # UnionSetの各Setについてチェック
    set_list = deltas.get_set_list()
    for i in range(set_list.n_set()):
        basic_set = set_list.get_at(i)
        n_dims = basic_set.dim(isl.dim_type.set)
        if axis_idx >= n_dims:
            continue

        # d[axis_idx] < 0 となる点が存在するかチェック
        # 存在すれば負の依存距離がある
        dim_name = basic_set.get_dim_name(isl.dim_type.set, axis_idx)
        if dim_name is None:
            dim_name = f"__delta{axis_idx}__"
            basic_set = basic_set.set_dim_name(isl.dim_type.set, axis_idx, dim_name)

        # 制約を追加: d[axis_idx] < 0 (つまり -d[axis_idx] - 1 >= 0)
        space = basic_set.get_space()
        c = isl.Constraint.ineq_from_names(space, {dim_name: -1, 1: -1})
        negative_set = basic_set.add_constraint(c)
        if not negative_set.is_empty():
            return False

    return True


def apply_tiling(
    func: PrimFunc,
    tiles: list[Tile],
    ctx: isl.Context | None = None,
    check_legality: bool = True,
) -> isl.Schedule:
    """
    指定された軸をタイル化したスケジュールを返す

    Args:
        func: 対象の関数
        tiles: タイル化の指定リスト
        ctx: ISLコンテキスト
        check_legality: 合法性をチェックするかどうか

    Returns:
        タイル化されたISL Schedule

    Raises:
        IllegalTilingError: タイル化が依存関係を違反する場合
    """
    ctx = ctx or isl.Context()

    # 合法性チェック
    if check_legality:
        is_legal, violations = check_tiling_legality(func, tiles, ctx)
        if not is_legal:
            raise IllegalTilingError(
                f"Tiling violates dependences: {', '.join(violations)}"
            )

    # ドメインとスケジュールマップを構築
    domain = build_domain(func, ctx)
    schedule_map = build_schedule(func, ctx)

    # スケジュールツリーを構築
    mupa = isl.MultiUnionPwAff.from_union_map(schedule_map)
    schedule = isl.Schedule.from_domain(domain)
    node = schedule.get_root()
    node = node.child(0)
    node = node.insert_partial_schedule(mupa)

    # タイル化を適用
    if node.get_type() == isl.schedule_node_type.band:
        node = _apply_tile_to_band(func, node, tiles, ctx)

    return node.get_schedule()


def _apply_tile_to_band(
    func: PrimFunc,
    node: isl.ScheduleNode,
    tiles: list[Tile],
    ctx: isl.Context,
) -> isl.ScheduleNode:
    """
    バンドノードにタイル化を適用

    Args:
        func: 対象の関数
        node: バンドノード
        tiles: タイル化の指定リスト
        ctx: ISLコンテキスト

    Returns:
        タイル化されたバンドノード
    """
    # タイル指定を軸インデックスからサイズへのマップに変換
    tile_map: dict[int, int] = {}
    for tile in tiles:
        tile_map[tile.axis] = tile.tile_size

    # タイルサイズのMultiValを構築
    space = node.band_get_space()
    n_dims = space.dim(isl.dim_type.set)

    tile_sizes = isl.MultiVal.zero(space)
    for i in range(n_dims):
        if i in tile_map:
            tile_sizes = tile_sizes.set_val(
                i, isl.Val.int_from_si(ctx, tile_map[i])
            )
        else:
            # タイル化しない軸はサイズ1
            tile_sizes = tile_sizes.set_val(i, isl.Val.int_from_si(ctx, 1))

    # タイル化を実行
    node = node.band_tile(tile_sizes)

    return node


def compute_optimized_schedule(
    func: PrimFunc,
    ctx: isl.Context | None = None,
) -> isl.Schedule:
    """
    依存関係を解析し、Skewing等を含む最適なスケジュールを自動計算する。
    これにより、タイル化可能なPermutable Bandが生成される。
    """
    ctx = ctx or isl.Context()

    domain = build_domain(func, ctx)
    schedule_map = build_schedule(func, ctx).intersect_domain(domain)
    read_access = build_read_access(func, ctx)
    write_access = build_write_access(func, ctx)

    raw_dep = compute_raw_dependence(schedule_map, write_access, read_access)
    war_dep = compute_war_dependence(schedule_map, write_access, read_access)
    waw_dep = compute_waw_dependence(schedule_map, write_access)

    sc = isl.ScheduleConstraints.on_domain(domain)

    all_deps = raw_dep.union(war_dep).union(waw_dep)
    sc = sc.set_validity(all_deps)
    sc = sc.set_coincidence(all_deps)
    sc = sc.set_proximity(all_deps)

    return sc.compute_schedule()


def apply_tiling_to_schedule(
    schedule: isl.Schedule,
    tiles: list[Tile],
) -> isl.Schedule:
    """
    最適化されたスケジュールにタイリングを適用する。
    """
    ctx = schedule.get_ctx()
    root = schedule.get_root()
    node = root.child(0)

    if node.get_type() != isl.schedule_node_type.band:
        return schedule

    tile_map: dict[int, int] = {tile.axis: tile.tile_size for tile in tiles}

    space = node.band_get_space()
    n_dims = space.dim(isl.dim_type.set)

    tile_sizes = isl.MultiVal.zero(space)
    for i in range(n_dims):
        size = tile_map.get(i, 1)
        tile_sizes = tile_sizes.set_val(i, isl.Val.int_from_si(ctx, size))

    node = node.band_tile(tile_sizes)

    return node.get_schedule()
