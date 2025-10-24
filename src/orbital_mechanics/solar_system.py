import numpy as np
import pandas as pd

from pathlib import Path
from numba import njit

from .constants import ALTAIRA_MU
from .orbit_utils import mean2ecc, ecc2true, kep2rv, orbit_points

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
    @njit
    def _mean2ecc(ecc:np.ndarray, MA:np.ndarray) -> np.ndarray:
        # mean to eccentric anomaly utility function
        N = ecc.shape[0]
        EA = np.empty((N,), dtype=ecc.dtype)
        for i in range(N):
            EA[i] = mean2ecc(ecc[i], MA[i])
        return EA

    
    @staticmethod
    @njit
    def _ecc2true(ecc:np.ndarray, EA:np.ndarray) -> np.ndarray:
        # eccentric to true anomaly utility function
        N = ecc.shape[0]
        TA = np.empty((N,), dtype=ecc.dtype)
        for i in range(N):
            TA[i] = ecc2true(ecc[i], EA[i])
        return TA


    @staticmethod
    @njit
    def _kep2rv(kep:np.ndarray) -> np.ndarray:
        # keplerian elements to r,v vectors utility function
        N = kep.shape[0]
        rv = np.empty((N,6), dtype=kep.dtype)
        for i in range(N):
            rv[i] = kep2rv(kep[i], ALTAIRA_MU)
        return rv


    @staticmethod
    def _orbit_points(kep:np.ndarray, ta_arr:np.ndarray) -> list:
        # get the orbit points for all the given bodies
        # mainly used to plot the orbits
        N = kep.shape[0]
        pts = np.empty((N,3,ta_arr.shape[0]), dtype=kep.dtype)
        for i in range(N):
            pts[i] = orbit_points(kep[i], ta_arr)
        return pts


    def get_state_at_t(self, t, idx:pd.Series = None) -> pd.DataFrame:
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

        # calculate the eccentric and true anomaly efficiently
        ecc_arr = df['e'].to_numpy(dtype=float)
        ma_arr = df['MA'].to_numpy(dtype=float)

        ea_arr = self._mean2ecc(ecc_arr, ma_arr)
        ta_arr = self._ecc2true(ecc_arr, ea_arr)

        df['EA'] = ea_arr
        df['TA'] = ta_arr

        # calculate the cartesian vectors
        kep_arr = df[['A','e','i','RAAN','AOP','TA']].to_numpy(dtype=float)
        rv_arr = self._kep2rv(kep_arr)

        df[['rx','ry','rz','vx','vy','vz']] = rv_arr

        return df


    def get_orbit_points(self, idx:pd.Series = None, num_points:int = 20) -> pd.DataFrame:
        # returns x,y,z arrays around the orbit of the bodies denoted by idx
        # this function is useful for plotting the orbit

        if idx is None:
            idx = pd.Series(np.arange(len(self.init_bodies)))
        
        df = self.init_bodies.loc[idx].copy()
        ta_arr = np.linspace(-np.pi, np.pi, num_points)
        df.attrs['TA_arr'] = ta_arr
        kep_arr = df[['A','e','i','RAAN','AOP']].to_numpy(dtype=float)

        # calculate the x,y,z points for the orbit trajectory
        out = self._orbit_points(kep_arr, ta_arr)
        df['orbit'] = [out[i] for i in range(out.shape[0])]

        return df

if __name__ == "__main__":
    ss = SolarSystem()
    ss.get_state_at_t(100.0, ss.planets_idx)
    ss.get_orbit_points(ss.planets_idx)