# Tile Regions

`Region` is a plain `(x, y, width, height)` crop area in level-0 coordinates.
`TileSpec` gives `target_mpp` an explicit contract: `size` is the final output pixel
size, `mpp` is the target physical resolution (or `None` for native resolution).

::: histoslice.tiles.Region

::: histoslice.tiles.TileSpec

::: histoslice.tiles.tile_regions

::: histoslice.tiles.filter_by_tissue

::: histoslice.tiles.level0_tile_size

::: histoslice.tiles.spot_regions
