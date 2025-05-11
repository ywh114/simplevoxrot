#!/usr/bin/env python3
import matplotlib.pyplot as plt

from util import voxel

IPYNB = False

if __name__ == '__main__':
    # Initialize voxel space
    extent = 14
    vs = voxel.VoxelSpace(extent)

    diamond = voxel.Voxel(vs)
    diamond.make_cone(4, 8, (8, 8, 8), invert=True, color='lightblue')
    diamond.make_cone(4, 8, (8, 8, 8), color='lightblue')
    diamond.make_cylinder(4.2, 2, (8, 8, 8), color='gold')

    f = diamond
    x_c, y_c, z_c = f.center_of_mass

    # Set up plot centered on COM
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x_c - extent, x_c + extent))
    ax.set_ylim((y_c - extent, y_c + extent))
    ax.set_zlim((z_c - extent, z_c + extent))  # type: ignore
    ax.set_box_aspect([1, 1, 1])  # type: ignore
    ax.voxels(f.filled, facecolors=f.facecolors, edgecolor='k')  # type: ignore
    plt.axis('off')
    plt.show()
