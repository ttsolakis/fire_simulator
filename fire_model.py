# fire_simulator/fire_model.py

import numpy as np
from fire_simulator.fire_grid import FireGrid

# 8-connected neighbor offsets, plus their (dx,dy) for wind alignment
# Note: dx is +x step (cols), dy is –y step (rows), matching your old convention.
_NEIGHBORS = [
    {'shift': (-1, -1), 'dx': -1, 'dy': +1},
    {'shift': (-1,  0), 'dx':  0, 'dy': +1},
    {'shift': (-1, +1), 'dx': +1, 'dy': +1},
    {'shift': ( 0, -1), 'dx': -1, 'dy':  0},
    {'shift': ( 0, +1), 'dx': +1, 'dy':  0},
    {'shift': (+1, -1), 'dx': -1, 'dy': -1},
    {'shift': (+1,  0), 'dx':  0, 'dy': -1},
    {'shift': (+1, +1), 'dx': +1, 'dy': -1},
]

def update_fire(grid: FireGrid, dt: float, wind_speed: float, wind_dir_rad: float):
    """
    Vectorized update of burn + ignition progress.
    """
    gs  = grid.grid_state
    bp  = grid.burn_progress
    ip  = grid.ignition_progress
    fl  = grid.flammability
    cs  = grid.cell_size

    # Burn‐out step (vectorized)
    burning = (gs == FireGrid.BURNING)
    # compute burn durations for ALL burning cells at once
    bt = burn_duration(cs, fl[burning], wind_speed)          # 1D array for burning positions
    # advance their progress
    bp_flat = bp[burning] + (dt / bt)
    # mark those that finish
    done     = bp_flat >= 1.0
    # update state & clamp
    idxs     = np.nonzero(burning)
    gs[idxs[0][done], idxs[1][done]] = FireGrid.BURNED
    bp[idxs[0][done], idxs[1][done]] = 1.0
    # leave others burning
    bp[burning] = bp_flat
    # now 'burning' mask should exclude newly‐burned
    burning[idxs[0][done], idxs[1][done]] = False

    # Spread/ignition step (vectorized over neighbors)
    # Only unburned cells can ignite:
    unburned = (gs == FireGrid.UNBURNED)

    for nb in _NEIGHBORS:
        # roll the burning mask into neighbor positions
        shifted_burning = np.roll(burning, shift=nb['shift'], axis=(0,1))

        # mask off the wrap-around edges
        si, sj = nb['shift']
        if si == -1:   shifted_burning[-1,:] = False
        if si == +1:   shifted_burning[ 0,:] = False
        if sj == -1:   shifted_burning[:, -1] = False
        if sj == +1:   shifted_burning[:,  0] = False

        # candidate ignition cells: currently unburned and adjacent to burning
        candidates = shifted_burning & unburned

        if not np.any(candidates):
            continue

        # directional dx, dy for these spread distances
        dx = nb['dx']; dy = nb['dy']

        # compute ignition times and rates for all candidates
        f_nb     = fl[candidates]
        it       = ignition_duration(cs, f_nb, wind_speed, wind_dir_rad, dx=dx, dy=dy)
        rate_flat= dt / it

        # advance their ignition progress
        ip_flat = ip[candidates] + rate_flat
        ip[candidates] = ip_flat

        # see which cross 1.0 this round
        fires = ip_flat >= 1.0
        if not np.any(fires):
            continue

        # stochastic trial on those that crossed
        pos_i, pos_j = np.nonzero(candidates)
        pos_i = pos_i[fires]; pos_j = pos_j[fires]

        # draw uniform and compare to ignition_probability
        probs = ignition_probability(fl[candidates][fires])
        randv = np.random.rand(len(probs))
        ignite_mask = randv < probs

        # ignite selected subset
        ign_i = pos_i[ignite_mask]; ign_j = pos_j[ignite_mask]
        gs[ign_i, ign_j] = FireGrid.BURNING
        bp[ign_i, ign_j] = 0.0
        ip[ign_i, ign_j] = 0.0

    # commit back
    grid.grid_state        = gs
    grid.burn_progress     = bp
    grid.ignition_progress = ip
    
# probability of ignition based on flammability
def ignition_probability(flammability, base_prob: float = 1.0):
    """
    Vectorized probability [0–1] of igniting.  Works on scalars or arrays.
    """
    return np.minimum(1.0, base_prob * np.asarray(flammability, dtype=float))

def fire_travel_time(cell_size: float,
                     flammability,
                     wind_speed: float,
                     wind_dir_rad: float,
                     dx: float,
                     dy: float) -> np.ndarray:
    """
    Returns an array of travel times.  Zero‐flammability → infinite time.
    """
    # --- compute distance & direction once (scalars) ---
    if dx or dy:
        distance      = np.hypot(dx, dy) * cell_size
        cell_direction = np.arctan2(dy, dx)
    else:
        distance       = cell_size
        cell_direction = 0.0

    # wind‐alignment (scalar)
    delta     = cell_direction - wind_dir_rad
    sharpness = 1.5
    align     = ((1 + np.cos(delta)) / 2) ** sharpness
    wind_factor = 1.0 / (1.0 + wind_speed * align)

    # force flammability into a float array
    f = np.asarray(flammability, dtype=float)

    # compute (vectorized) but avoid warnings:
    with np.errstate(divide='ignore', invalid='ignore'):
        flame_factor = 1.0 / f
        result = (distance / 0.2) * flame_factor * wind_factor
        # wherever f <= 0, force to inf
        result = np.where(f <= 0, np.inf, result)

    return result

# delay before neighbor ignites
def ignition_duration(cell_size: float, flammability: float, wind_speed: float, wind_dir_rad: float, dx: float, dy: float) -> float:
    "Delay before neighbor ignites"
    return fire_travel_time(cell_size, flammability, wind_speed, wind_dir_rad, dx, dy)

# time for a cell to burn out in place
def burn_duration(cell_size: float, flammability: float, wind_speed: float) -> float:
    "Time for a cell to burn out in place"
    min_slowdown = 5
    max_slowdown = 40
    max_windspeed = 20.0  # m/s
    burn_slowdown = min_slowdown + (max_slowdown - min_slowdown) * (wind_speed/max_windspeed)  # linear slowdown
    burn_slowdown = 5
    return burn_slowdown * fire_travel_time(cell_size, flammability, wind_speed, 0.0, dx=0, dy=0)
