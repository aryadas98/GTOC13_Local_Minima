"""Validator utilities for GTOC13 submissions.

This module implements the physics & geometry checks described in the
problem statement: Kepler propagation, patched-conic flyby checks,
hyperbolic excess computations, RK4 re-propagation for integrated arcs,
and a set of helper checks. It intentionally does NOT parse a submission
format; adapt `parse_solution()` (or call `Validator` methods) to feed the
validator the expected data structures.

Units: kilometers (km), kilometers-per-second (km/s), seconds (s), AU
conversion constant provided below. Times may be given in seconds or years
— be consistent; the validator expects times in seconds in the API.

Usage (example):

    from rng.validator import Validator, propagate_kepler

    # build ephemerides dict: body_id -> {'a':..., 'e':..., 'i':..., 'Omega':..., 'omega':..., 'M0':..., 't0':..., 'mu':..., 'radius':..., 'period':...}
    val = Validator(ephemerides)
    # call specific helpers or val.validate_solution(parsed_solution)

"""
from __future__ import annotations

import math
from typing import Dict, Tuple, Sequence, Callable, Any, List

import numpy as np

# Constants
KM_PER_AU = 149597870.691
POSITION_TOL_KM = 0.1       # 100 m
VEL_TOL_KM_S = 1e-7         # 0.1 mm/s = 1e-7 km/s
FLYBY_ALT_TOL_KM = 0.1      # 100 m
PERIHELION_TOL_KM = 1.0     # 1 km
REL_INT_TOL = 1e-4
REPORT_INTERVAL_MIN_S = 60.0


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _solve_kepler_eccentric_anomaly(M: float, e: float, tol: float = 1e-12, maxiter: int = 100) -> float:
    """Solve Kepler's equation for eccentric anomaly E when e < 1.

    M in radians, returns E in radians.
    """
    if e < 0.8:
        E = M
    else:
        E = math.pi
    for _ in range(maxiter):
        f = E - e * math.sin(E) - M
        fp = 1 - e * math.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            return E
    raise RuntimeError("Kepler solver (elliptic) did not converge")


def _solve_kepler_hyperbolic_anomaly(M: float, e: float, tol: float = 1e-12, maxiter: int = 100) -> float:
    """Solve M = e*sinh(F) - F for hyperbolic eccentric anomaly F (e>1).

    M may be negative; returns F (real) such that M = e sinh F - F.
    """
    # initial guess
    if M >= 0:
        F = math.log(2 * M / e + 1.8)
    else:
        F = -math.log(-2 * M / e + 1.8)
    for _ in range(maxiter):
        f = e * math.sinh(F) - F - M
        fp = e * math.cosh(F) - 1
        dF = -f / fp
        F += dF
        if abs(dF) < tol:
            return F
    raise RuntimeError("Kepler solver (hyperbolic) did not converge")


def _rot_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def propagate_kepler(a: float,
                     e: float,
                     i_deg: float,
                     Omega_deg: float,
                     omega_deg: float,
                     M0_deg: float,
                     t: float,
                     mu: float,
                     t0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate classical Keplerian orbit to time t and return r (km), v (km/s).

    Parameters
    - a: semi-major axis in km (positive for elliptic, negative for hyperbolic)
    - e: eccentricity
    - i_deg, Omega_deg, omega_deg, M0_deg: orbital angles in degrees
    - t, t0: times in seconds (t0 is epoch of M0)
    - mu: gravitational parameter (km^3/s^2) of central body (e.g. Sun)

    Returns (r, v) in the inertial frame.
    """
    # mean motion and mean anomaly
    if a == 0.0:
        raise ValueError("Semi-major axis a cannot be zero")
    # convert angles
    i = math.radians(i_deg)
    Omega = math.radians(Omega_deg)
    omega = math.radians(omega_deg)
    M0 = math.radians(M0_deg)

    if e < 1.0:
        n = math.sqrt(mu / (a ** 3))
        M = M0 + n * (t - t0)
        # reduce M to [-pi, pi]
        M = (M + math.pi) % (2 * math.pi) - math.pi
        E = _solve_kepler_eccentric_anomaly(M, e)
        # true anomaly
        cosE = math.cos(E)
        sinE = math.sin(E)
        sqrt_1_e2 = math.sqrt(1 - e * e)
        cosf = (cosE - e) / (1 - e * cosE)
        sinf = (sqrt_1_e2 * sinE) / (1 - e * cosE)
        f = math.atan2(sinf, cosf)
        # distance
        r_p = a * (1 - e * cosE)
        # position in perifocal
        r_pf = np.array([r_p * math.cos(f), r_p * math.sin(f), 0.0])
        # velocity in perifocal
        h = math.sqrt(mu * a * (1 - e * e))
        v_pf = np.array([-mu / h * math.sin(f), mu / h * (e + math.cos(f)), 0.0])
    else:
        # hyperbolic
        a_h = -abs(a)
        n = math.sqrt(mu / (abs(a_h) ** 3))
        M = M0 + n * (t - t0)
        # hyperbolic anomaly F
        F = _solve_kepler_hyperbolic_anomaly(M, e)
        coshF = math.cosh(F)
        sinhF = math.sinh(F)
        # true anomaly
        sqrt_e2_1 = math.sqrt(e * e - 1)
        tanhf2 = sqrt_e2_1 * sinhF / (e - coshF)
        f = 2 * math.atan(tanhf2)
        r_p = a_h * (1 - e * coshF)
        # position in perifocal
        r_pf = np.array([r_p * math.cos(f), r_p * math.sin(f), 0.0])
        # velocity in perifocal
        h = math.sqrt(mu * abs(a_h) * (e * e - 1))
        v_pf = np.array([-mu / h * math.sin(f), mu / h * (e + math.cos(f)), 0.0])

    # rotate from perifocal to inertial frame: R = Rz(Omega) * Rx(i) * Rz(omega)
    R = _rot_z(Omega) @ _rot_x(i) @ _rot_z(omega)
    r = R @ r_pf
    v = R @ v_pf
    return r, v


def compute_v_inf(v_sc: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    return v_sc - v_body


def check_position_equality(r_sc: np.ndarray, r_body: np.ndarray, tol_km: float = POSITION_TOL_KM) -> Tuple[bool, float]:
    err = norm(r_sc - r_body)
    return err <= tol_km, err


def check_vinf_magnitude_equal(vinf_minus: np.ndarray, vinf_plus: np.ndarray, tol_km_s: float = VEL_TOL_KM_S) -> Tuple[bool, float]:
    m1 = norm(vinf_minus)
    m2 = norm(vinf_plus)
    return abs(m1 - m2) <= tol_km_s, abs(m1 - m2)


def compute_turn_angle(vinf_minus: np.ndarray, vinf_plus: np.ndarray) -> float:
    # use average magnitude for denominator as suggested
    m1 = norm(vinf_minus)
    m2 = norm(vinf_plus)
    vinf = 0.5 * (m1 + m2)
    if vinf <= 0.0:
        return 0.0
    cosd = float(np.dot(vinf_minus, vinf_plus) / (vinf * vinf))
    cosd = max(-1.0, min(1.0, cosd))
    return math.acos(cosd)


def compute_flyby_periapsis_altitude(delta: float, vinf: float, muP: float, Rp: float) -> float:
    """Compute hp (altitude above planet surface) from turn angle delta (rad), vinf (km/s).

    Uses rearranged patched-conic formula in the problem statement.
    Returns hp in km.
    """
    s = math.sin(delta / 2.0)
    if s <= 0.0 or vinf <= 0.0 or s >= 1.0:
        raise ValueError("Invalid input for flyby periapsis computation")
    mu_over_r = (s * vinf * vinf) / (1.0 - s)
    RP_plus_hp = muP / mu_over_r
    hp = RP_plus_hp - Rp
    return hp


def rk4_propagate(accel_func: Callable[[float, np.ndarray], np.ndarray],
                  x0: np.ndarray,
                  t0: float,
                  tf: float,
                  dt: float) -> np.ndarray:
    """Simple RK4 integrator for state x = [r(3), v(3)]. accel_func(t, x) -> a(3).

    Returns final state x(tf).
    """
    x = x0.copy()
    t = t0
    nsteps = max(1, int(math.ceil((tf - t0) / dt)))
    h = (tf - t0) / nsteps
    for _ in range(nsteps):
        r = x[:3]
        v = x[3:]
        a1 = accel_func(t, x)
        k1_r = v
        k1_v = a1

        a2 = accel_func(t + 0.5 * h, np.concatenate([r + 0.5 * h * k1_r, v + 0.5 * h * k1_v]))
        k2_r = v + 0.5 * h * k1_v
        k2_v = a2

        a3 = accel_func(t + 0.5 * h, np.concatenate([r + 0.5 * h * k2_r, v + 0.5 * h * k2_v]))
        k3_r = v + 0.5 * h * k2_v
        k3_v = a3

        a4 = accel_func(t + h, np.concatenate([r + h * k3_r, v + h * k3_v]))
        k4_r = v + h * k3_v
        k4_v = a4

        x[:3] = r + (h / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        x[3:] = v + (h / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        t += h
    return x


class Validator:
    """High level helper container for validation checks.

    ephemerides: mapping body_id -> dict with keys a,e,i,Omega,omega,M0,t0,mu,radius,period
    Times must be seconds in the interfaces below. Radii in km, mu in km^3/s^2.
    """

    def __init__(self, ephemerides: Dict[int, Dict[str, Any]]):
        self.ephem = ephemerides

    def body_state(self, body_id: int, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (r, v) of body at time t (seconds) using Kepler orbital elements.

        Expects ephemerides[body_id] to contain: a,e,i,Omega,omega,M0,t0,mu
        """
        if body_id not in self.ephem:
            raise KeyError(f"No ephemeris for body {body_id}")
        e = self.ephem[body_id]
        r, v = propagate_kepler(e['a'], e['e'], e['i'], e['Omega'], e['omega'], e['M0'], t, e['mu'], t0=e.get('t0', 0.0))
        return r, v

    def check_flyby(self,
                    tG: float,
                    r_sc: np.ndarray,
                    v_sc_minus: np.ndarray,
                    v_sc_plus: np.ndarray,
                    body_id: int) -> Tuple[bool, Dict[str, Any]]:
        """Run the suite of flyby checks for a single reported flyby event.

        Returns (pass:bool, info:dict) where info contains numeric diagnostics.
        """
        info: Dict[str, Any] = {}
        r_body, v_body = self.body_state(body_id, tG)

        ok_pos, pos_err = check_position_equality(r_sc, r_body)
        info['pos_err_km'] = pos_err
        if not ok_pos:
            info['error'] = f'position mismatch {pos_err*1000:.1f} m'
            return False, info

        vinf_minus = compute_v_inf(v_sc_minus, v_body)
        vinf_plus = compute_v_inf(v_sc_plus, v_body)
        info['vinf_minus_km_s'] = norm(vinf_minus)
        info['vinf_plus_km_s'] = norm(vinf_plus)

        # mass classification: bodies 1..10 are massive planets per statement
        massive = 1 <= body_id <= 10
        if massive:
            ok_vmag, vmag_diff = check_vinf_magnitude_equal(vinf_minus, vinf_plus)
            info['vinf_mag_diff_km_s'] = vmag_diff
            if not ok_vmag:
                info['error'] = f'vinf magnitude mismatch {vmag_diff*1000:.3f} m/s'
                return False, info
        else:
            # massless: vectors must be equal (direction and mag) within small tol
            vec_diff = norm(vinf_plus - vinf_minus)
            info['vinf_vec_diff_km_s'] = vec_diff
            if vec_diff > VEL_TOL_KM_S:
                info['error'] = f'vinf vector discontinuity {vec_diff*1000:.3f} m/s'
                return False, info

        # turn angle and hp (use planet mu and radius if available)
        delta = compute_turn_angle(vinf_minus, vinf_plus)
        info['turn_angle_rad'] = delta
        muP = self.ephem[body_id].get('mu_body', None)
        Rp = self.ephem[body_id].get('radius', None)
        if muP is not None and Rp is not None:
            vinf_mag = 0.5 * (norm(vinf_minus) + norm(vinf_plus))
            try:
                hp = compute_flyby_periapsis_altitude(delta, vinf_mag, muP, Rp)
            except Exception as exc:
                info['error'] = f'cannot compute hp: {exc}'
                return False, info
            info['hp_km'] = hp
            # check altitude band for massive planets only
            if massive:
                if not (0.1 * Rp - FLYBY_ALT_TOL_KM <= hp <= 100.0 * Rp + FLYBY_ALT_TOL_KM):
                    info['error'] = f'hp out of bounds: {hp:.3f} km (Rp={Rp:.3f})'
                    return False, info

        return True, info

    def check_successive_flyby_timing(self, t1: float, t2: float, body_id: int) -> Tuple[bool, float]:
        """Check if two successive flybys of same body are separated by >= 1/3 of orbital period.

        t1,t2 in seconds. Returns (ok, dt_days)
        """
        if body_id not in self.ephem:
            raise KeyError(body_id)
        period = self.ephem[body_id].get('period', None)
        if period is None:
            # cannot check without period
            return True, abs(t2 - t1)
        dt = abs(t2 - t1)
        return dt >= (period / 3.0), dt

    def check_reporting_interval(self, times: Sequence[float]) -> Tuple[bool, float]:
        diffs = np.diff(np.asarray(times, dtype=float))
        min_dt = float(np.min(diffs)) if len(diffs) > 0 else float('inf')
        return min_dt > REPORT_INTERVAL_MIN_S, min_dt

    def check_integration_rel_error(self,
                                    x0: np.ndarray,
                                    xf_reported: np.ndarray,
                                    accel_func: Callable[[float, np.ndarray], np.ndarray],
                                    t0: float,
                                    tf: float,
                                    dt: float) -> Tuple[bool, float]:
        x_rk4 = rk4_propagate(accel_func, x0, t0, tf, dt)
        numer = norm(x_rk4 - xf_reported)
        denom = max(1e-12, norm(xf_reported - x0))
        rel = numer / denom
        return rel < REL_INT_TOL, rel


def find_perihelia_in_trajectory(states: Sequence[np.ndarray]) -> List[Tuple[int, float]]:
    """Find local minima in radial distance along a state sequence.

    states: sequence of state vectors [r(3), v(3)]. Returns list of (index, r_km) for minima.
    """
    rs = np.array([norm(s[:3]) for s in states])
    minima = []
    for i in range(1, len(rs) - 1):
        if rs[i] < rs[i - 1] and rs[i] < rs[i + 1]:
            minima.append((i, float(rs[i])))
    return minima


__all__ = [
    'propagate_kepler', 'compute_v_inf', 'check_position_equality', 'compute_turn_angle',
    'compute_flyby_periapsis_altitude', 'rk4_propagate', 'Validator', 'find_perihelia_in_trajectory'
]
