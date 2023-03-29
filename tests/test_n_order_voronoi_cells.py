import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('macosx')
import numpy as np
from collections import Counter
from dyrect import EpsilonNet, draw_complex, WitnessComplex
import dyrect as dy

def draw_voronoi_cells_2d(lms, order=0, areax=[0, 1], areay=[0,1], resolution=10):
    fwidth = 20
    fig = plt.figure(figsize=(fwidth, fwidth*0.4))
    rows = 1
    cols = 2

    xpoints = np.linspace(areax[0], areax[1], resolution+1)
    ypoints = np.linspace(areay[0], areay[1], resolution+1)
    X, Y = np.meshgrid(xpoints, ypoints)

    print(X.shape)

    plt.show()

def unit_square_norder_voronoi_cells():
    stats = []
    npoints = 5000

    seeds = [0]
    for seed in seeds:
        print("Seed: " + str(seed))
        np.random.seed(seed)
        points = np.random.random((npoints, 2))
        eps=0.1
        EN = EpsilonNet(eps, 0)
        EN.fit(points)
        lms = EN.landmarks
        print("EN done")
        wc = WitnessComplex(lms, points, 2)
        # pwc = PatchedWitnessComplex(lms, points, 2, patching_level=4)
        draw_voronoi_cells_2d(lms, order=0, areax = [0, 1], areay=[0,1], resolution=100)
        # draw_example(lms, points, eps, pwc)



unit_square_norder_voronoi_cells()