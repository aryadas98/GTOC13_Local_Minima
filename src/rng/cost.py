from __future__ import annotations

import numpy as np
from typing import Iterable


def seasonal_penalty_sequence(rhats: Iterable[Iterable[float]]) -> np.ndarray:
    """
    Compute the seasonal penalty S for a sequence of unit heliocentric
    position vectors of a single body.

    The formula (for visit i>1) is:
      S(rhat_{k,i}) = 0.1 + 0.9 / (1 + 10 * sum_{j=1}^{i-1} exp(- (acosd(rhat_i·rhat_j)^2) / 50))

    and S(rhat_{k,1}) = 1.

    Parameters
    - rhats: sequence-like of shape (N, 3) containing the (approx.) unit
      heliocentric position vectors for successive flybys. The function will
      normalize vectors if they are not exact unit length.

    Returns
    - numpy array of length N with the seasonal penalty S for each visit.
    """
    rhats_arr = np.asarray(list(rhats), dtype=float)
    if rhats_arr.size == 0:
        return np.array([], dtype=float)

    # Ensure a 2D array with shape (N, 3)
    if rhats_arr.ndim == 1:
        # single vector given
        rhats_arr = rhats_arr.reshape(1, -1)

    if rhats_arr.shape[1] != 3:
        raise ValueError("Each position vector must have 3 components (x,y,z)")

    # Normalize to unit vectors to be robust to slight deviations
    norms = np.linalg.norm(rhats_arr, axis=1)
    # avoid division by zero
    norms = np.where(norms == 0.0, 1.0, norms)
    rhats = (rhats_arr.T / norms).T

    N = rhats.shape[0]
    S = np.empty(N, dtype=float)

    for i in range(N):
        if i == 0:
            S[i] = 1.0
            continue

        # dot products with previous visit vectors
        dots = rhats[i].dot(rhats[:i].T)
        # numerical safety: clamp into [-1, 1]
        dots = np.clip(dots, -1.0, 1.0)

        # acos returns radians; convert to degrees as required
        angles_deg = np.degrees(np.arccos(dots))

        # compute exponential terms and sum
        exp_terms = np.exp(-(angles_deg ** 2) / 50.0)
        summ = float(np.sum(exp_terms))

        S[i] = 0.1 + (0.9 / (1.0 + 10.0 * summ))

    return S


def seasonal_penalty_single(rhat: Iterable[float], previous_rhats: Iterable[Iterable[float]]) -> float:
    """Compute S for a single visit given previous visits.

    - rhat: 3-element iterable for the current unit heliocentric vector.
    - previous_rhats: sequence of prior vectors (each 3 elements).

    Returns the scalar S for the current visit. If there are no previous
    visits (len(previous_rhats) == 0) this returns 1.0.
    """
    prev = list(previous_rhats)
    if len(prev) == 0:
        return 1.0
    seq = prev + [list(rhat)]
    S = seasonal_penalty_sequence(seq)
    return float(S[-1])


def cost_function(b: float, c: float, w, n) -> float:
    """
    Compute the objective J described in the problem statement:

        J = b * c * sum_k w_k * sum_{i=1..N_k} ( S(rhat_{k,i}) * F(Vinf_{k,i}) )

    Expected input shapes / types (flexible):
    - b: scalar grand tour bonus
    - c: scalar time bonus
    - w: mapping (dict) from body_id -> weight, or a sequence (list/array) indexable by
         integer body id. If a body id is missing its weight defaults to 0.
    - n: mapping (dict) from body_id -> list-of-visits. Each visit may be either:
         * a dict with keys 'rhat' (3-vector), 'v_inf' (float) and optional 'is_science' (bool),
         * a sequence/tuple (rhat, v_inf) or (rhat, v_inf, is_science).

    Only visits flagged as scientific (default True) are included in the sums. The
    function is intentionally permissive to work with several common data layouts.

    Returns
    - scalar float J
    """

    def _get_weight(body_id):
        # support dict-like or sequence-like weights
        if isinstance(w, dict):
            return float(w.get(body_id, 0.0))
        try:
            # sequence/array access (body_id must be integer-like)
            return float(w[int(body_id)])
        except Exception:
            return 0.0

    total_score = 0.0

    # iterate bodies in n
    for body_id, visits in n.items():
        weight = _get_weight(body_id)
        if weight == 0.0:
            # skip bodies with zero weight early
            continue

        # collect scientific visits in chronological order
        rhats = []
        vins = []

        for vis in visits:
            # normalize allowed visit representations
            if isinstance(vis, dict):
                rhat = vis.get("rhat")
                v_inf = vis.get("v_inf")
                is_science = vis.get("is_science", True)
            else:
                # assume sequence-like
                try:
                    rhat = vis[0]
                    v_inf = vis[1]
                    is_science = vis[2] if len(vis) > 2 else True
                except Exception:
                    # malformed entry: skip
                    continue

            if not is_science:
                continue

            rhats.append(rhat)
            vins.append(v_inf)

        if len(rhats) == 0:
            continue

        # compute seasonal penalties and velocity penalties for this body's scientific visits
        S_arr = seasonal_penalty_sequence(rhats)
        F_arr = flyby_velocity_penalty(vins)

        subtotal = float(np.sum(S_arr * F_arr))
        total_score += weight * subtotal

    J = float(b * c * total_score)
    return J


def flyby_velocity_penalty(v_inf: float | Iterable[float]) -> np.ndarray:
        """
        Flyby velocity penalty term F(V_inf).

        Implements:
            F(V_inf) = 0.2 + exp(-V_inf/13) / (1 + exp(-5*(V_inf - 1.5)))

        where V_inf is given in km/s. The function accepts a scalar or an
        iterable/array of V_inf values and returns a numpy array of the same
        shape with the penalty values.

        Parameters
        - v_inf: float or iterable of floats (km/s)

        Returns
        - numpy.ndarray of penalty values (dtype float)
        """
        # Use numpy for vectorized handling
        v = np.asarray(v_inf, dtype=float)

        # Compute numerator and denominator separately for clarity
        num = np.exp(-v / 13.0)
        den = 1.0 + np.exp(-5.0 * (v - 1.5))

        F = 0.2 + (num / den)

        # Ensure we always return an ndarray (even for scalar input)
        return np.array(F, dtype=float)