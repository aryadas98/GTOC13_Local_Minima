import numpy as np
import pandas as pd

from pathlib import Path

from constants import ALTAIRA_MU
from orbit_utils import mean2ecc, ecc2true, kep2rv

class SolarSystem:

    def __init__(self):
        # hardcoded filenames
        PLANET_FILE = "gtoc13_planets.csv"
        ASTEROID_FILE = "gtoc13_asteroids.csv"
        COMET_FILE = "gtoc13_comets.csv"

        planet_path = Path(__file__).with_name(PLANET_FILE)
        asteroid_path = Path(__file__).with_name(ASTEROID_FILE)
        comet_path = Path(__file__).with_name(COMET_FILE)

        planet_df = pd.read_csv(planet_path,
                                     header=None,
                                     skiprows=1,
                                     index_col=None,
                                     names=['id','name','mu','R','A','e','i','RAAN','AOP','MA','Wgt'],
                                     encoding_errors='ignore')
        
        planet_df[['type', 'massless']] = ['planet', False]
        planet_df.loc[np.isclose(planet_df['mu'],0.0), 'massless'] = True 

        asteroid_df = pd.read_csv(asteroid_path,
                                       header=None,
                                       skiprows=1,
                                       index_col=None,
                                       names=['id','A','e','i','RAAN','AOP','MA','Wgt'],
                                       encoding_errors='ignore')
        
        asteroid_df[['name','mu','R','type','massless']] = ['',0.0,0.0,'asteroid',True]
        
        comet_df = pd.read_csv(comet_path,
                                    header=None,
                                    skiprows=1,
                                    index_col=None,
                                    names=['id','A','e','i','RAAN','AOP','MA','Wgt'],
                                    encoding_errors='ignore')
        
        comet_df[['name','mu','R','type','massless']] = ['',0.0,0.0,'comet',True]

        self.init_bodies = pd.concat([planet_df, comet_df, asteroid_df], ignore_index=True)

        # add epoch information
        self.init_bodies.attrs['epoch'] = 0.0

        # convert orbital angles to radians
        self.init_bodies[['i','RAAN','AOP','MA']] = np.deg2rad(self.init_bodies[['i','RAAN','AOP','MA']])

        # calculate the orbital angular velocity and time period
        self.init_bodies['n'] = np.sqrt(ALTAIRA_MU / self.init_bodies['A'] ** 3)
        self.init_bodies['T'] = 2*np.pi / self.init_bodies['n']

        # collect the indices for different types of bodies
        self.planets_idx = self.init_bodies['type'] == 'planet'
        self.massive_planets_idx = (self.init_bodies['type'] == 'planet') & \
                                   (self.init_bodies['massless'] == False)
        self.massless_planets_idx = (self.init_bodies['type'] == 'planet') & \
                                                  (self.init_bodies['massless'] == True)
        
        self.comets_idx = self.init_bodies['type'] == 'comet'
        self.asteroids_idx = self.init_bodies['type'] == 'asteroid'


    @staticmethod
    def _mean2ecc(s):
        # mean to eccentric anomaly utility function
        # for applying in numba.apply
        return mean2ecc(s[0], s[1])
    
    @staticmethod
    def _ecc2true(s):
        # eccentric to true anomaly utility function
        # for applying in numba.apply
        return ecc2true(s[0], s[1])

    @staticmethod
    def _kep2rv(s):
        return kep2rv(s, ALTAIRA_MU)

    def get_state_at_t(self, t, idx:pd.Series = None):
        # returns the orbital states of the bodies at idx indices, propagated to time t (seconds)
        # if no indices are provided, all the bodies states are returned
        if idx is None:
            idx = pd.Series(np.arange(len(self.init_bodies)))
        
        df = self.init_bodies.loc[idx].copy()
        df.attrs['epoch'] = t


        # calculate the mean anomaly at time t
        t0 = self.init_bodies.attrs['epoch']
        df['MA'] += df['n'] * (t - t0)
        df['MA'] = (df['MA'] + np.pi) % (2 * np.pi) - np.pi

        # calculate the eccentric anomaly efficiently
        df['EA'] = df[['e','MA']].apply(self._mean2ecc, axis=1, raw=True, engine='numba',
                                       engine_kwargs={"nopython": True, "nogil": True, "parallel": True})
        
        # calculate the true anomaly efficiently
        df['TA'] = df[['e','EA']].apply(self._ecc2true, axis=1, raw=True, engine='numba',
                                       engine_kwargs={"nopython": True, "nogil": True, "parallel": True})
        

        # calculate the cartesian position and velocity
        df[['rx','ry','rz','vx','vy','vz']] = df[['A','e','i','RAAN','AOP','TA']].apply(self._kep2rv, axis=1,
                                                        raw=True, result_type='expand', engine='numba',
                                                        engine_kwargs={"nopython": True, "nogil": True, "parallel": True})


if __name__ == "__main__":
    ss = SolarSystem()
    ss.get_state_at_t(100.0, ss.planets_idx)