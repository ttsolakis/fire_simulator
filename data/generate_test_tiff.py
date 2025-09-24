#!/usr/bin/env python3
"""
generate_test_tiff.py

Creates a single-band GeoTIFF grid for fire simulation testing,
always writing the output into the same data/ folder this script lives in.
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

# ─────── User-configurable settings ───────
ROWS            = 100      # number of rows (height in pixels)
COLS            = 100      # number of columns (width in pixels)
CELL_SIZE       = 1.0      # size of each cell, in metres
LANDCOVER_VALUE = 10        # integer code to fill every pixel
CRS             = "EPSG:3857"  # projected CRS so units are metres
# ─────────────────────────────────────────

# Compute data/ folder (where this script lives)
BASE_DIR    = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(BASE_DIR, "test_grid.tif")

def main():
    # 1) create a uniform landcover array
    data = np.full((ROWS, COLS), LANDCOVER_VALUE, dtype=np.int32)

    # 2) define an affine transform: top-left at (0, ROWS*CELL_SIZE)
    transform = from_origin(
        west=0,
        north=ROWS * CELL_SIZE,
        xsize=CELL_SIZE,
        ysize=CELL_SIZE
    )

    # 3) write the GeoTIFF into the data/ folder
    with rasterio.open(
        OUTPUT_PATH, "w",
        driver="GTiff",
        height=ROWS,
        width=COLS,
        count=1,
        dtype=data.dtype,
        crs=CRS,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    print(f"Generated {OUTPUT_PATH}:")
    print(f"  rows × cols: {ROWS} × {COLS}")
    print(f"  cell size: {CELL_SIZE} m")
    print(f"  landcover code: {LANDCOVER_VALUE}")
    print(f"  CRS: {CRS}")

if __name__ == "__main__":
    main()
