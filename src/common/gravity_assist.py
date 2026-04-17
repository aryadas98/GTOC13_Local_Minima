import math


def turn_angle(mu, R, v_inf, hp):
    if mu == 0: return 0

    sind2 = mu/((v_inf**2)*(R+hp)+mu)
    d = 2*math.asin(sind2)
    return d

def minmax_turn_angle(mu, R, v_inf, h_min, h_max):
    return turn_angle(mu, R, v_inf, h_max), turn_angle(mu, R, v_inf, h_min)