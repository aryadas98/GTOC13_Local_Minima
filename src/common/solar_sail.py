import numpy as np


def calc_opt_un(ur:np.ndarray, ud:np.ndarray) -> np.ndarray:
    eps = 1e-3    # tolerance below which to consider ur and ud parallel

    ur = ur.reshape(3); ud = ud.reshape(3)
    ur = ur/np.linalg.norm(ur)
    ud = ud/np.linalg.norm(ud)

    # invert ud
    ud = -ud

    # calculate the angle between ur and ud
    alpha = np.arccos(np.dot(ur, ud))

    if alpha < eps:   # parallel
        return ur.copy()   # just return the vector to the sun
    
    # normal vector of the plane containing ur and ud
    k = np.cross(ur, ud)
    k = k / np.linalg.norm(k)

    # this constant has been calculated using mathematical analysis
    # to get the highest component of force along ud
    theta = 0.5 * alpha

    # rotate the vector ur about k by angle theta using rodrigues formula
    un = ur * np.cos(theta) + np.cross(k, ur)*np.sin(theta)

    return un
