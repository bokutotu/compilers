from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    axis: int
    tile_size: int


class IllegalTilingError(ValueError):
    """依存関係に違反するタイル化が指定された場合の例外."""
