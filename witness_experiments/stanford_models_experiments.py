import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

import dyrect as dy
from dyrect import draw_complex, EpsilonNet, PatchedWitnessComplex, WitnessComplex

out_path = 'meshes/'


def stanford_bunny():
    plydata = PlyData.read('models/bunny/reconstruction/bun_zipper.ply')
    px = plydata.elements[0].data['x']
    py = plydata.elements[0].data['y']
    pz = plydata.elements[0].data['z']
    points = np.transpose(np.array([px, py, pz]))
    print(points.shape)
    eps = 0.002
    model_name = "bunny_" + str(int(eps * 10000))
    patching_level = 3

    data_aspect = (np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2]))
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    print("Number of points: ", len(points))
    print("Number of landmarks: ", len(lms))

    # fwidth = 12
    # fig = plt.figure(figsize=(fwidth, fwidth * 0.4))
    #
    # rows = 1
    # cols = 2
    # ax = plt.subplot(rows, cols, 1, projection='3d')
    # # ax = fig.add_subplot(projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5)
    # ax.scatter(lms[:, 0], lms[:, 1], lms[:, 2], s=10.)
    # ax.set_box_aspect(data_aspect)
    #
    # ax = plt.subplot(rows, cols, 2, projection='3d')
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=patching_level, max_patched_dimensions=2)

    # pwc._dim = 3
    # draw_complex(pwc, fig=fig, ax=ax, vlabels=True)
    # ax.set_box_aspect(data_aspect)
    # plt.show()
    dy.save_as_plyfile(lms, pwc.simplices[1], pwc.simplices[2],
                       model_name + '_patching_level_' + str(patching_level), out_path)

def stanford_armadillo():
    plydata = PlyData.read('models/armadillo/Armadillo.ply')
    px = plydata.elements[0].data['x']
    py = plydata.elements[0].data['y']
    pz = plydata.elements[0].data['z']
    points = np.transpose(np.array([px, py, pz]))
    print(points.shape)

    eps = 4.5
    model_name = "armadillo_" + str(int(eps * 100))
    patching_level = 4

    data_aspect = (np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2]))

    np.random.seed(0)
    EN = EpsilonNet(eps, 0)
    EN.fit(points)
    lms = EN.landmarks
    print("Number of points: ", len(points))
    print("Number of landmarks: ", len(lms))

    fwidth = 12
    fig = plt.figure(figsize=(fwidth, fwidth * 0.4))

    rows = 1
    cols = 2
    ax = plt.subplot(rows, cols, 1, projection='3d')
    # ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5)
    ax.scatter(lms[:, 0], lms[:, 1], lms[:, 2], s=10.)
    ax.set_box_aspect(data_aspect)

    ax = plt.subplot(rows, cols, 2, projection='3d')
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=patching_level, max_patched_dimensions=2)
    print("patched_witness_computed")
    print("Betti numbers: ", pwc.betti_numbers)

    # pwc._dim = 3
    draw_complex(pwc, fig=fig, ax=ax, vlabels=True)
    ax.set_box_aspect(data_aspect)
    dy.save_as_plyfile(lms, pwc.simplices[1], pwc.simplices[2],
                    model_name + '_patching_level_' + str(patching_level), out_path)
    plt.show()

def stanford_buddha():
    plydata = PlyData.read('models/buddha/happy_recon/happy_vrip.ply')
    px = plydata.elements[0].data['x']
    py = plydata.elements[0].data['y']
    pz = plydata.elements[0].data['z']
    points = np.transpose(np.array([px, py, pz]))
    print(points.shape)

    data_aspect = (np.ptp(points[:, 0]), np.ptp(points[:, 1]), np.ptp(points[:, 2]))
    print(data_aspect)

    eps = .0025
    model_name = "buddha_" + str(int(eps * 10000))
    patching_level = 2

    np.random.seed(1)
    EN = EpsilonNet(eps, 0, method='furthest_point')
    EN.fit(points)
    lms = EN.landmarks
    print("Number of points: ", len(points))
    print("Number of landmarks: ", len(lms))

    # fwidth = 12
    # fig = plt.figure(figsize=(fwidth, fwidth * 0.4))
    #
    # rows = 1
    # cols = 2
    # ax = plt.subplot(rows, cols, 1, projection='3d')
    # # ax = fig.add_subplot(projection='3d')
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5)
    # ax.scatter(lms[:, 0], lms[:, 1], lms[:, 2], s=10.)
    # ax.set_box_aspect(data_aspect)
    # print("Scatter printed")

    # ax = plt.subplot(rows, cols, 2, projection='3d')
    pwc = PatchedWitnessComplex(lms, points, 2, patching_level=patching_level, max_patched_dimensions=2)
    print("patched_witness_computed")
    print("Betti numbers: ", pwc.betti_numbers)

    # pwc._dim = 3
    # draw_complex(pwc, fig=fig, ax=ax, vlabels=True)
    # ax.set_box_aspect(data_aspect)
    dy.save_as_plyfile(lms, pwc.simplices[1], pwc.simplices[2],
                       model_name + '_patching_level_' + str(patching_level), out_path)
    # plt.show()

def check_model():
    # datafile = "/Users/cybjaz/workspace/dyrect/witness_experiments/meshes/armadillo_900_patching_level_4.ply"
    # datafile = "/Users/cybjaz/workspace/dyrect/witness_experiments/meshes/armadillo_450_patching_level_4.ply"
    # datafile = "/Users/cybjaz/workspace/dyrect/witness_experiments/meshes/buddha_2_patching_level_4.ply"
    datafile = "/Users/cybjaz/workspace/dyrect/witness_experiments/meshes/buddha_40_patching_level_2.ply"
    lms, simplices = dy.load_plyfile(datafile)
    print(lms.shape)
    print(len(simplices[0]))
    model = dy.Complex(simplices=simplices, coords=lms, max_dim=2, ambient_dim=3)
    dy.all_triangles_intersection_test_3D(model)

stanford_bunny()
# stanford_armadillo()
# stanford_buddha()
# check_model()