import math
import numpy as np

from numba import njit


@njit
def mean2ecc(e, M, tol=1e-10, g_clip=0.5, max_iter=10):
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
def ecc2true(e, E):
    v = math.atan2(math.sqrt(1-e**2)*math.sin(E), math.cos(E)-e)
    return v


@njit
def Rx(th):
    R = np.array([[1.0,            0.0,             0.0],
                  [0.0,   math.cos(th),   -math.sin(th)],
                  [0.0,   math.sin(th),    math.cos(th)]])
    
    return R


@njit
def Ry(th):
    R = np.array([[ math.cos(th),   0.0,   math.sin(th)],
                  [          0.0,   1.0,            0.0],
                  [-math.sin(th),   0.0,   math.cos(th)]])


@njit
def Rz(th):
    R = np.array([[math.cos(th),   -math.sin(th),   0.0],
                  [math.sin(th),    math.cos(th),   0.0],
                  [         0.0,             0.0,   1.0]])
    
    return R

@njit
def pf2i_rot(inc, RAAN, AOP):
    R1 = Rz(RAAN)
    R2 = Rx(inc)
    R3 = Rz(AOP)

    R = R1 @ R2 @ R3

    return R


@njit
def r_pf(a,e,th):
    p = a*(1-e**2)
    r = p/(1 + e*math.cos(th))

    rpf = r*np.array([math.cos(th), math.sin(th), 0])

    return rpf

@njit
def v_pf(a,e,th,mu):
    p = a*(1-e**2)
    v = math.sqrt(mu/p)

    vpf = v*np.array([-math.sin(th), e+math.cos(th), 0])

    return vpf

@njit
def kep2rv(kep, mu):
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