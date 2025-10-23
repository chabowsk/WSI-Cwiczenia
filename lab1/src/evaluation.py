import numpy as np

def calc_path(control: np.ndarray, time: int = 200, dt: float = 0.1):
    pathx = []
    pathz = []
    posx = np.zeros(control.shape[:-1])
    posz = np.zeros(control.shape[:-1])
    velx = np.zeros(control.shape[:-1])
    velz = np.zeros(control.shape[:-1])
    control = control.reshape(*control.shape[:-1], time, 2)
    t = 0
    pathx.append(posx)
    pathz.append(posz)
    while (posz >= 0).any():
        if t < time:
            cx = control[..., t, 0]
            cz = control[..., t, 1]
        else:
            cx = 0
            cz = 0
        velx = velx + (cx * 15 - 0.5 * velx) * dt
        velz = velz + (cz * 15 - 9.8 - 0.5 * velz) * dt
        velx = velx * (posz >= 0)
        velz = velz * (posz >= 0)
        posx = posx + velx * dt
        posz = posz + velz * dt
        pathx.append(posx)
        pathz.append(posz)
        t += 1
    return pathx, pathz


def calc_target(control: np.ndarray):
    pathx, pathz = calc_path(control)
    return -(pathx[-1] - 350) ** 2