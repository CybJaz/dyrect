import numpy as np
from plyfile import PlyData, PlyElement
import time

class Timer():
    def __init__(self):
        self._tick = time.time()

    def tick(self):
        self._tick = time.time()

    def tock(self, msg=None):
        tock = time.time()
        if msg is not None:
            print("Elapsed time [", msg, "]: ", tock - self._tick)
        else:
            print("Elapsed time: ", tock - self._tick)
        self._tick = tock

def save_as_plyfile(pts, lines, triangles, mesh_name, out_path):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    pts = list(zip(x, y, z))

    # the vertex are required to a 1-d list
    vertex = np.array(pts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    faces = np.array([(list(tr), 122, 122, 122) for tr in triangles],
                     dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    edges = np.array([(list(ln), 50, 50, 50) for ln in lines],
                     dtype=[('vertex_indices', 'i4', (2,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el0 = PlyElement.describe(vertex, 'vertex')
    el1 = PlyElement.describe(edges, 'edges')
    el2 = PlyElement.describe(faces, 'face')
    PlyData([el0, el1, el2], text=mesh_name).write(out_path + mesh_name + '.ply')

def load_plyfile(filename):
    plydata = PlyData.read(filename)
    coords = np.array([list(c) for c in plydata.elements[0].data], dtype=np.float)
    simplices = {}
    simplices[0] = [(v,) for v in range(len(coords))]
    simplices[1] = [tuple(e) for [e,_,_,_] in plydata.elements[1].data]
    simplices[2] = [tuple(t) for [t,_,_,_] in plydata.elements[2].data]
    return coords, simplices