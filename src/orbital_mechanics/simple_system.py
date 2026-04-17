import numpy as np
import pandas as pd
import pykep

from pathlib import Path

from common.constants import ALTAIRA_MU as MU

from orbital_mechanics.solar_system import SolarSystem, GTOC13Body


class SimpleSystem(SolarSystem):

    def __init__(self):
        super().__init__()

        # filter bodies
        self.filter_bodies()

        # simplify indices
        self.fix_indices()

    
    def fix_indices(self):
        self.bodies_df = self.bodies_df.reset_index(drop=True)

        df = self.bodies_df

        planets_mask = (df['name'].isin(['Jotunn', 'Bespin', 'Beyonc', 'Hoth']))
        asteroids_mask = ~planets_mask

        self.planets_idx = df[planets_mask].index.to_numpy()
        self.asteroids_idx = df[asteroids_mask].index.to_numpy()

        # empty indices
        self.massive_planets_idx = np.empty(shape=(0,), dtype=int)
        self.massless_planets_idx = np.empty(shape=(0,), dtype=int)
        self.comets_idx = np.empty(shape=(0,), dtype=int)


    def filter_bodies(self):
        filtered_bodies = [pl for pl in self.bodies if self.eligible_bodies(pl)]
        filtered_bodies_mask = self.bodies_df.apply(self.eligible_bodies, axis=1)
        filtered_bodies_df = self.bodies_df[filtered_bodies_mask].copy()

        self.bodies = filtered_bodies
        self.bodies_df = filtered_bodies_df
    

    @staticmethod
    def eligible_bodies(body):
        if isinstance(body, GTOC13Body):
            name = body.name
            type_ = body.gtoc13_btype
        
        elif isinstance(body, pd.Series):
            name = body['name']
            type_ = body['type']
        
        else:
            raise ValueError()
        
        return (name in ['Jotunn', 'Bespin', 'Beyonc', 'Yandi', 'Hoth']) or \
                (type_ == "asteroid") or (type_ == "comet")


if __name__ == "__main__":
    ss = SimpleSystem()
    ss.get_state_at_t(100.0, ss.planets_idx)
    ss.get_orbit_points(ss.planets_idx)

