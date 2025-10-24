import math
import numpy as np

from numba import njit


@njit
def mean2ecc(e, M, tol=1e-10, g_clip=0.5, max_iter=10) -> float:
    f = lambda e,M,E: E-e*math.sin(E)-M
    df = lambda e,E: 1-e*math.cos(E)
    clip = lambda value,low,high: min(max(low, value), high)

    E = M + e*math.sin(M)
    n = 0
    while abs(f(e,M,E)) > tol and n < max_iter:
        dE = -f(e,M,E)/df(e,E)
        dE = clip(dE, -g_clip, g_clip)

        E = E + dE
        n += 1

    return E


@njit
def ecc2true(e, E) -> float:
    v = math.atan2(math.sqrt(1-e**2)*math.sin(E), math.cos(E)-e)
    return v


@njit
def Rx(th) -> np.ndarray:
    R = np.array([[1.0,            0.0,             0.0],
                  [0.0,   math.cos(th),   -math.sin(th)],
                  [0.0,   math.sin(th),    math.cos(th)]])
    
    return R


@njit
def Ry(th) -> np.ndarray:
    R = np.array([[ math.cos(th),   0.0,   math.sin(th)],
                  [          0.0,   1.0,            0.0],
                  [-math.sin(th),   0.0,   math.cos(th)]])


@njit
def Rz(th) -> np.ndarray:
    R = np.array([[math.cos(th),   -math.sin(th),   0.0],
                  [math.sin(th),    math.cos(th),   0.0],
                  [         0.0,             0.0,   1.0]])
    
    return R


@njit
def pf2i_rot(inc, RAAN, AOP) -> np.ndarray:
    R1 = Rz(RAAN)
    R2 = Rx(inc)
    R3 = Rz(AOP)

    R = R1 @ R2 @ R3

    return R


@njit
def r_pf(a,e,th) -> np.ndarray:
    p = a*(1-e**2)
    r = p/(1 + e*math.cos(th))

    rpf = r*np.array([math.cos(th), math.sin(th), 0])

    return rpf


@njit
def v_pf(a,e,th,mu) -> np.ndarray:
    p = a*(1-e**2)
    v = math.sqrt(mu/p)

    vpf = v*np.array([-math.sin(th), e+math.cos(th), 0])

    return vpf


@njit
def kep2rv(kep:np.ndarray, mu:float) -> np.ndarray:
    a, e, i, raan, aop, th = kep

    rpf = r_pf(a,e,th).reshape((3,1))
    vpf = v_pf(a,e,th,mu).reshape((3,1))
    R_pf2i = pf2i_rot(i, raan, aop)

    r_i = R_pf2i @ rpf
    v_i = R_pf2i @ vpf

    rv_i = np.empty((6,), dtype=r_i.dtype)
    rv_i[0:3] = r_i.ravel()
    rv_i[3:6] = v_i.ravel()

    return rv_i


@njit
def orbit_points(kep:np.ndarray, ta_arr:np.ndarray) -> np.ndarray:
    kep_ag = np.empty((6,), dtype=kep.dtype)
    kep_ag[0:5] = kep

    N = ta_arr.shape[0]

    orbit_pts = np.empty((3, N), dtype=kep.dtype)

    for i in range(N):
        kep_ag[5] = ta_arr[i]

        # mu can be set to 0 because we aren't interested in v
        r_i = kep2rv(kep_ag, 0)[0:3]
        orbit_pts[:,i] = r_i
    
    return orbit_pts