import math
import numpy as np
import rasterio
from rasterio.enums import Resampling

class FireGrid:
    # Cell‐state constants
    UNBURNED = 0
    BURNING  = 1
    BURNED   = 2

    def __init__(self, rows: int, cols: int, cell_size: float):
        """
        Initialize a fire simulation grid.

        Args:
          rows, cols: number of grid cells in Y and X.
          cell_size: size of each square cell in metres.
        """
        self.rows      = rows
        self.cols      = cols
        self.cell_size = cell_size

        # Grid data per cell
        self.grid_state        = np.zeros((rows, cols), dtype=np.uint8)  #TODO: change self.grid to self.grid_state
        self.burn_progress     = np.zeros((rows, cols), dtype=np.float32)
        self.ignition_progress = np.zeros((rows, cols), dtype=np.float32)
        self.landcover         = None
        self.flammability      = np.ones((rows, cols), dtype=np.float32)

    @classmethod
    def from_raster(cls, path: str, class_to_value: dict):
        """
        Build a FireGrid from any single-band GeoTIFF.

        - Reads rows, cols, transform, CRS, and resolution.
        - If CRS is projected, uses metres directly.
        - If CRS is geographic, converts degrees → metres at raster centre.
        - Loads raw landcover codes and maps to flammability.
        """
        with rasterio.open(path) as src:
            raw        = src.read(1)
            transform  = src.transform
            rows, cols = raw.shape
            res_x, res_y = src.res  # pixel size in CRS units
            crs = src.crs

        # Determine cell size in metres
        if crs.is_geographic:
            # convert degrees → metres at the raster centroid latitude
            lon0, lat0 = transform * (cols / 2, rows / 2)
            m_per_deg_lon = 111_320 * math.cos(math.radians(lat0))
            m_per_deg_lat = 110_574
            width_m  = abs(res_x) * m_per_deg_lon
            height_m = abs(res_y) * m_per_deg_lat
            cell_size = (width_m + height_m) / 2.0
        else:
            # assume linear units (metres) in projected CRS
            cell_size = (abs(res_x) + abs(res_y)) / 2.0

        print(f"Raster: {rows}×{cols}, cell ≈ {cell_size:.2f} m")

        # Instantiate and populate
        grid = cls(rows, cols, cell_size)
        grid.landcover = raw.astype(np.int32)
        grid.map_flammability(class_to_value)
        return grid

    def map_flammability(self, class_to_value: dict):
        """
        Map integer landcover codes → flammability [0–1].
        """
        if self.landcover is None:
            raise RuntimeError("Load landcover first.")
        f = np.zeros_like(self.landcover, dtype=np.float32)
        for code, val in class_to_value.items():
            f[self.landcover == code] = val
        self.flammability = f

    def ignite(self, i: int, j: int):
        """Ignite cell (i,j) immediately."""
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.grid_state[i, j]     = self.BURNING
            self.burn_progress[i, j]     = 0.0
            self.ignition_progress[i, j] = 0.0

    def get_state(self, i: int, j: int) -> int:
        """Return the fire state at (i,j)."""
        if 0 <= i < self.rows and 0 <= j < self.cols:
            return int(self.grid_state[i, j])
        return None

    def set_state(self, i: int, j: int, state: int):
        """Manually override a cell’s fire state."""
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.grid_state[i, j] = state
