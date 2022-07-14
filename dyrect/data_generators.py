import numpy as np
from math import pi
from scipy.integrate import RK45
import sympy as sym


def generate_points(system, dimension, starting_point, n_points=1000, int_step=0.02):
    integrator = RK45(system, 0, starting_point, 10000, first_step=int_step, max_step=5 * int_step)
    n = n_points
    points = np.empty((n, dimension))

    for _ in range(1000):
        integrator.step()

    for i in range(n):
        points[i] = integrator.y
        integrator.step()
    return points

# EQUATIONS
# ##############


def lorentz_eq(_, args):
    l_sigma = 10.
    l_beta = 8. / 3
    l_rho = 28.
    return [l_sigma * (args[1] - args[0]),
            args[0] * (l_rho - args[2]) - args[1],
            args[0] * args[1] - l_beta * args[2]]


# TODO: add noise to t step
def lemniscate_eq(t, params=None):
    if params is None:
        params = [1.]
    c = np.sqrt(2) * np.cos(t)
    s2 = np.sin(t) ** 2
    x = params[0] * c / (1 + s2)
    return np.transpose(np.vstack(([x], [x * np.sin(t)])))


# 1D TIME SERIES
# ##############

def logistic_map(npoints, r=4, starting_point=1 / 3., symb=False):
    """
    @param npoints: length of the generated sequence
    @param r: parameter of the logistic equation
    @param starting_point: a starting point for generating a sequence
    @param symb: compute sequence symbolically?
    @return:
    """
    def log_map(x):
        return r * x * (1 - x)
    if symb:
        points = [sym.sympify(starting_point)]
    else:
        points = [starting_point]
    for i in range(1, npoints):
        points.append(log_map(points[-1]))
    return np.array([p for p in points])


def tent_map(npoints, starting_point=1 / 2):
    """
    :param npoints:
    :param starting_point: because of the numeric unstability of tent map use sympy starting points
    :return:
    """

    def t(x):
        # print(sympy.N(x,40), x.evalf(), x.evalf() <= 1/2, -x.evalf() <= 0)

        if x.evalf() <= 1 / 2:
            r = 2 * x
        else:
            r = 2 * (1 - x)
        # if r.evalf() < 0:
        #     return -r
        # else:
        return r

    # print(starting_point)
    points = [starting_point]
    # points = np.empty((npoints,))
    # points[0] = starting_point
    for i in range(1, npoints):
        points.append(t(points[-1]))
        # points[i] = t(points[i-1])
        # print(points[-1])
    return np.array([p.evalf() for p in points])


# 2D TIME SERIES
# ##############

def lemniscate(npoints, step=0.2, scale=1., tnoise=0.02, noise=0.05):
    times = np.empty((npoints,))
    times[0] = 0.
    for i in range(npoints - 1):
        times[i + 1] = times[i] + step + np.random.normal(0, tnoise)
    return lemniscate_eq(times, [scale]) + np.random.normal(0, noise, (npoints, 2))
    # return lemniscate_eq(np.arange(0., npoints * step * np.pi, step * np.pi), [scale]) + \
    #        np.random.normal(0, noise, (npoints, 2))
    # np.random.random_sample((npoints, 2))*noise


def circle_rotation(npoints, step=pi / 10, scale=1., starting_point=np.array([1, 0])):
    """
    :param npoints: number of generated points
    :param step: rotation in radians
    :param scale: radius of the circle
    :param starting_point:
    :return:
    """
    points = np.empty((npoints, 2))
    for i in range(npoints):
        c = np.cos(i * step)
        s = np.sin(i * step)
        points[i] = np.dot(np.array([[c, -s], [s, c]]), starting_point)
    return points * scale


# 3D TIME SERIES
# ##############

def lorenz_attractor(npoints, step=0.02, adaptive_step=False, starting_point=None, skip=100):
    if starting_point is None:
        starting_point = [1., 1., 1.]
    points = np.empty((npoints, 3))
    # starting_point = [1., 1., 1.]
    integrator = RK45(lorentz_eq, 0, starting_point, 10000,
                      first_step=step, max_step=(4 * step if adaptive_step else step))

    # get closer to the attractor first
    for _ in range(skip):
        integrator.step()

    for i in range(npoints):
        points[i] = integrator.y
        if integrator.status == 'running':
            integrator.step()
        else:
            print('reloading integrator at ' + str(i))
            integrator = RK45(lorentz_eq, 0, points[i], 10000,
                              first_step=step, max_step=(4 * step if adaptive_step else step))
    return points


def torus_rotation(npoints, ang_step=0.02, rotation=0.1, radius=.25):
    angles_phi = np.arange(0., npoints * pi * ang_step, pi * ang_step)
    angles_theta = angles_phi * rotation

    torus_points = np.array([[
        (1 + radius * np.cos(p)) * np.cos(t),
        (1 + radius * np.cos(p)) * np.sin(t),
        radius * np.sin(p)] for p, t in zip(angles_phi, angles_theta)])
    return torus_points


# 2D POINT CLOUDS
# ##############

def unit_circle_sample(npoints, noise=0.0):
    rpoints = 2 * np.pi * np.random.random_sample(npoints)
    x = np.cos(rpoints) + (np.random.random_sample(npoints) - 0.5) * noise
    y = np.sin(rpoints) + (np.random.random_sample(npoints) - 0.5) * noise
    return np.transpose(np.stack((x, y)))
