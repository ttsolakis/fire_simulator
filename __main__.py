# fire_simulator/__main__.py

import os
import numpy as np
from .fire_grid       import FireGrid
from .fire_simulation import FireSimulation
from .visualize       import save_static_map, visualize_fire

# ── Land-cover → flammability ─────────────────────────────────────
CLASS_TO_FLAMM = {
    10: 1.0,   # broadleaf forest
    20: 1.0,   # needleleaf forest
    30: 0.8,   # mixed forest
    40: 0.6,   # shrubland
    50: 0.5,   # grassland
    60: 0.2,   # sparse vegetation
    70: 0.0,   # bare soil
    80: 0.0,   # water
    90: 0.0,   # buildings
}

# ── Class-code → display color & name ─────────────────────────────
CLASS_COLOR_MAP = {
    10: (0.0, 0.4, 0.0),
    20: (0.0, 0.6, 0.3),
    30: (0.2, 0.7, 0.2),
    40: (0.6, 0.5, 0.2),
    50: (0.5, 0.8, 0.5),
    60: (0.9, 0.8, 0.5),
    70: (0.7, 0.5, 0.3),
    80: (0.0, 0.0, 0.5),
    90: (0.3, 0.3, 0.3),
}
CLASS_NAME_MAP = {
    10: "Broadleaf forest",
    20: "Needleleaf forest",
    30: "Mixed forest",
    40: "Shrubland",
    50: "Grassland",
    60: "Sparse vegetation",
    70: "Bare soil",
    80: "Water",
    90: "Buildings",
}

def main():
    # ── Simulation parameters ────────────────────────────────────────────
    dt                 = 1.0     # seconds per step
    wind_speed_mu      = 10.0     # m/s
    wind_speed_sigma   = 1.0     # m/s
    wind_dir_mu_rad    = -np.pi/4     # radians (0 = east)
    wind_dir_sigma_rad = 0.05     # radians
    steps_per_frame    = 60      # number of steps to take until next frame
    delay_sec          = 0.1     # pause between frames in seconds

    # ── Load GeoTIFF into grid ──────────────────────────────────────────
    pkg_dir = os.path.dirname(__file__)
    # tif_path = os.path.join(pkg_dir, "data", "test_grid.tif")
    # tif_path = os.path.join(pkg_dir, "data", "test_lcm.tiff")
    tif_path = os.path.join(pkg_dir, "data", "patras_lcm10_2500x1250.tif")
    grid = FireGrid.from_raster(tif_path, CLASS_TO_FLAMM)

    # ── Save landcover + flammability maps side-by-side ─────────────────────
    save_static_map(
        grid,
        class_color_map = CLASS_COLOR_MAP,
        class_name_map  = CLASS_NAME_MAP,
        unknown_color   = (0.6, 0.8, 1.0),
        unknown_label   = "Sea",
        output_path     = "landcover_and_flammability.png"
    )

    # ── Set up fire simulation ─────────────────────────────────
    sim = FireSimulation(
        grid               = grid,
        dt                 = dt,
        wind_speed_mu      = wind_speed_mu,
        wind_speed_sigma   = wind_speed_sigma,
        wind_dir_mu_rad    = wind_dir_mu_rad,
        wind_dir_sigma_rad = wind_dir_sigma_rad
    )

    # ── Show interactive fire animation ───────────────────────────────────
    # visualize_fire(sim, class_color_map = CLASS_COLOR_MAP, delay_sec = delay_sec, steps_per_frame = steps_per_frame)

    visualize_fire(sim, CLASS_COLOR_MAP,
               delay_sec,
               steps_per_frame,
               record=True,
               record_path="my_fire.mp4",
               record_fps=1)

if __name__ == "__main__":
    main()
