import numpy as np
import pandas as pd
import pykep

from pathlib import Path

from common.constants import ALTAIRA_MU as MU


class GTOC13Body(pykep.planet.keplerian):
    def __init__(self, id, body_type, weight, *args, **kwargs):
        self.gtoc13_id = int(id)
        self.gtoc13_btype = str(body_type)
        self.gtoc13_weight = float(weight)
        super().__init__(*args, **kwargs)


    def __repr__(self):
        mystr = f"Id: {self.gtoc13_id}\n" \
                f"Type: {self.gtoc13_btype}\n" \
                f"Weight: {self.gtoc13_weight}\n"
        def_str = super().__repr__()

        return mystr + def_str


class SolarSystem:

    def __init__(self):
        # read the csv files
        df = self.read_and_preprocess_files()

        # make the indices
        self.make_indices(df)

        # now convert the data frame into an array of pykep planets
        self.t0 = 0.0
        self.bodies = np.empty(len(df), dtype=GTOC13Body)
        
        for i in range(len(df)):
            row = df.iloc[i]
            self.bodies[i] = GTOC13Body(
                row['id'],
                row['type'],
                row['Wgt'],
                pykep.epoch(self.t0),
                row[['A','e','i','RAAN','AOP','MA']],
                MU,
                row['mu'],
                row['R'],
                1.1*row['R'],
                row['name']
            )
        
        # also store the original dataframe (for human readability)
        df.attrs['epoch'] = self.t0
        self.bodies_df = df


    def read_and_preprocess_files(self):
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
        
        planet_df['type'] = 'planet'

        asteroid_df = pd.read_csv(asteroid_path,
                                       header=None,
                                       skiprows=1,
                                       index_col=None,
                                       names=['id','A','e','i','RAAN','AOP','MA','Wgt'],
                                       encoding_errors='ignore')
        
        asteroid_df[['name','mu','R','type']] = ['',0.0,0.0,'asteroid']
        
        comet_df = pd.read_csv(comet_path,
                                    header=None,
                                    skiprows=1,
                                    index_col=None,
                                    names=['id','A','e','i','RAAN','AOP','MA','Wgt'],
                                    encoding_errors='ignore')
        
        comet_df[['name','mu','R','type']] = ['',0.0,0.0,'comet']

        combined_df = pd.concat([planet_df, comet_df, asteroid_df], ignore_index=True)

        # convert degrees to radians
        combined_df[['i','RAAN','AOP','MA']] *= pykep.DEG2RAD

        return combined_df


    def make_indices(self, df):
        # collect the indices for different types of bodies
        # run this function after reading and preprocessing the files
        planets_mask = df['type'] == 'planet'
        massive_planets_mask = (df['type'] == 'planet') & (df['mu'] > 0)
        massless_planets_mask = (df['type'] == 'planet') & (df['mu'] == 0.0)
        comets_mask = df['type'] == 'comet'
        asteroids_mask = df['type'] == 'asteroid'

        self.planets_idx = df[planets_mask].index.to_numpy()
        self.massive_planets_idx = df[massive_planets_mask].index.to_numpy()
        self.massless_planets_idx = df[massless_planets_mask].index.to_numpy()
        self.comets_idx = df[comets_mask].index.to_numpy()
        self.asteroids_idx = df[asteroids_mask].index.to_numpy()


    def get_state_at_t(self, t:float, idx:np.ndarray = None) -> pd.DataFrame:
        # returns the orbital states of the bodies at idx indices, propagated to time t (seconds)
        # if no indices are provided, all the bodies states are returned
        if idx is None:
            idx = np.arange(len(self.bodies))
        
        t *= pykep.SEC2DAY

        out_df = self.bodies_df.iloc[idx].copy()
        out_df.attrs['epoch'] = t
        out_df[['MA','rx','ry','rz','vx','vy','vz']] = None

        # propagate the planets and set their positions in the output df
        for i in idx:
            pl = self.bodies[i]
            ma = pl.osculating_elements(pykep.epoch(t))
            ma = ma[5]       # mean anomaly
            r,v = pl.eph(pykep.epoch(t))  # position, velocity
            out_df.loc[i, ['MA','rx','ry','rz','vx','vy','vz']] = [ma, *r, *v]

        return out_df


    def get_orbit_points(self, idx:np.ndarray = None, num_points:int = 20) -> pd.DataFrame:
        # returns x,y,z arrays around the orbit of the bodies denoted by idx
        # this function is useful for plotting the orbit
        if idx is None:
            idx = np.arange(len(self.bodies))

        out_df = self.bodies_df.iloc[idx].copy()
        xyz_points = np.empty((out_df.shape[0], 3, num_points))

        # calculate the x,y,z points for each orbit one by one
        for i in range(len(idx)):
            pl = self.bodies[idx[i]]
            pl_T = pl.compute_period(pykep.epoch(0))
            t_arr = np.linspace(self.t0, self.t0 + pl_T, num_points)

            for j in range(len(t_arr)):
                _t = t_arr[j] * pykep.SEC2DAY
                r,_ = pl.eph(pykep.epoch(_t))
                xyz_points[i,:,j] = r
        
        out_df['orbit'] = [xyz_points[i] for i in range(xyz_points.shape[0])]

        return out_df

if __name__ == "__main__":
    ss = SolarSystem()
    ss.get_state_at_t(100.0, ss.planets_idx)
    ss.get_orbit_points(ss.planets_idx)