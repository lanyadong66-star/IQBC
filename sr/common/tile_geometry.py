"""Shared tile coordinates for SR inference and scaled post-processing."""

from __future__ import annotations

import math
from typing import NamedTuple


class TileRegion(NamedTuple):
    input_start_x: int
    input_end_x: int
    input_start_y: int
    input_end_y: int
    input_start_x_pad: int
    input_end_x_pad: int
    input_start_y_pad: int
    input_end_y_pad: int
    output_start_x: int
    output_end_x: int
    output_start_y: int
    output_end_y: int
    tile_output_start_x: int
    tile_output_end_x: int
    tile_output_start_y: int
    tile_output_end_y: int


def count_tile_regions(height, width, tile_size):
    """Return the number of core tiles needed to cover an image."""
    if height < 1 or width < 1:
        raise ValueError('Tile image dimensions must be positive.')
    if tile_size < 1:
        raise ValueError('tile_size must be positive.')
    return math.ceil(width / tile_size) * math.ceil(height / tile_size)


def iter_tile_regions(height, width, tile_size, tile_pad, scale):
    """Yield matching input, padded-input, output, and tile-output regions."""
    if height < 1 or width < 1:
        raise ValueError('Tile image dimensions must be positive.')
    if tile_size < 1 or scale < 1:
        raise ValueError('tile_size and scale must be positive.')
    if tile_pad < 0:
        raise ValueError('tile_pad cannot be negative.')

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            input_start_x = tile_x * tile_size
            input_end_x = min(input_start_x + tile_size, width)
            input_start_y = tile_y * tile_size
            input_end_y = min(input_start_y + tile_size, height)

            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            tile_output_start_x = (input_start_x - input_start_x_pad) * scale
            tile_output_start_y = (input_start_y - input_start_y_pad) * scale
            tile_output_end_x = tile_output_start_x + (input_end_x - input_start_x) * scale
            tile_output_end_y = tile_output_start_y + (input_end_y - input_start_y) * scale

            yield TileRegion(
                input_start_x=input_start_x,
                input_end_x=input_end_x,
                input_start_y=input_start_y,
                input_end_y=input_end_y,
                input_start_x_pad=input_start_x_pad,
                input_end_x_pad=input_end_x_pad,
                input_start_y_pad=input_start_y_pad,
                input_end_y_pad=input_end_y_pad,
                output_start_x=output_start_x,
                output_end_x=output_end_x,
                output_start_y=output_start_y,
                output_end_y=output_end_y,
                tile_output_start_x=tile_output_start_x,
                tile_output_end_x=tile_output_end_x,
                tile_output_start_y=tile_output_start_y,
                tile_output_end_y=tile_output_end_y,
            )
