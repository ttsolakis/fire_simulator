# fire_simulator/fire_simulation.py

import numpy as np
from fire_simulator.fire_grid import FireGrid
from fire_simulator.fire_model import update_fire

class FireSimulation:
    """ High‐level wrapper to run a FireGrid through time with stochastic wind. """
    def __init__(self, grid: FireGrid, dt: float, wind_speed_mu: float, wind_speed_sigma: float, wind_dir_mu_rad: float, wind_dir_sigma_rad: float):
        self.grid               = grid
        self.dt                 = dt
        self.wind_speed_mu      = wind_speed_mu
        self.wind_speed_sigma   = wind_speed_sigma
        self.wind_dir_mu_rad    = wind_dir_mu_rad
        self.wind_dir_sigma_rad = wind_dir_sigma_rad
        self.step_count         = 0
        self.wind_speed         = 0.0
        self.wind_direction     = 0.0

    def ignite(self, i: int, j: int):
        """Ignite the cell at (i,j)."""
        self.grid.ignite(i, j)

    def step(self):
        """ Advance the simulation by one timestep:
            - draw a new wind speed & direction,
            - update the FireGrid accordingly,
            - increment step counter.              """
        
        self.wind_speed = max(0.0, np.random.normal(self.wind_speed_mu, self.wind_speed_sigma))  # wind speed sampling
        self.wind_direction = np.random.normal(self.wind_dir_mu_rad, self.wind_dir_sigma_rad)    # wind direction sampling

        update_fire(self.grid, dt=self.dt, wind_speed=self.wind_speed, wind_dir_rad=self.wind_direction)
        self.step_count += 1

    def run(self, steps: int, callback=None):
        """
        Run the simulation for a given number of steps.

        After each step, if `callback` is provided, it is called as:
            callback(grid, step_count)

        where `grid` is the FireGrid and `step_count` is the current step.
        """
        for _ in range(steps):
            self.step()
            if callback:
                callback(self.grid, self.step_count)
