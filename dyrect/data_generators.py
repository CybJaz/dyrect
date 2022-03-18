import numpy as np
from scipy.integrate import RK45


def generate_points(system, dimension, starting_point, n_points, int_step=0.02):
    integrator = RK45(system, 0, starting_point, 10000, first_step=int_step, max_step=5 * int_step)
    n = 1000
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
    l_sigma = 10.;
    l_beta = 8. / 3;
    l_rho = 28.;
    return [l_sigma * (args[1] - args[0]),
            args[0] * (l_rho - args[2]) - args[1],
            args[0] * args[1] - l_beta * args[2]]


# TODO: add noise to t step
def lemniscate_eq(t, params=[1.]):
    c = np.sqrt(2) * np.cos(t)
    s2 = np.sin(t) ** 2
    x = params[0] * c / (1 + s2)
    return np.transpose(np.vstack(([x], [x * np.sin(t)])))


# 2D TIME SERIES
# ##############

def lemniscate(npoints, step=0.2, scale=1., tnoise=0.02, noise=0.05):
    times = np.empty((npoints,))
    times[0] = 0.
    for i in range(npoints-1):
        times[i+1] = times[i] + step + np.random.normal(0, tnoise)
    return lemniscate_eq(times, [scale]) + np.random.normal(0, noise, (npoints, 2))
    # return lemniscate_eq(np.arange(0., npoints * step * np.pi, step * np.pi), [scale]) + \
    #        np.random.normal(0, noise, (npoints, 2))
    # np.random.random_sample((npoints, 2))*noise


# 3D TIME SERIES
# ##############

def lorenz_attractor(npoints, step=0.02, adaptive_step=False, starting_point=[1., 1., 1.], skip=100):
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


# 2D POINT CLOUDS
# ##############

def unit_circle_sample(npoints, noise=0.0):
    rpoints = 2 * np.pi * np.random.random_sample((npoints))
    x = np.cos(rpoints) + (np.random.random_sample((npoints)) - 0.5) * noise
    y = np.sin(rpoints) + (np.random.random_sample((npoints)) - 0.5) * noise
    return np.transpose(np.stack((x, y)))
