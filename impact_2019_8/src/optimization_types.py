from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    """タイル化の指定."""

    axis: str
    tile_size: int

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")


class IllegalTilingError(ValueError):
    """依存関係に違反するタイル化が指定された場合の例外."""
