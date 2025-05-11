#!/usr/bin/env python3
import matplotlib.pyplot as plt

from util import voxel

IPYNB = False

if __name__ == '__main__':
    # Initialize voxel space
    extent = 14
    vs = voxel.VoxelSpace(extent)

    cross = voxel.Voxel(vs)
    anchor = voxel.Voxel(vs)

    cross.make_rect_prism((5, 1, 1), (5, 5, 5), color='red')
    cross.make_rect_prism((1, 1, 5), (7, 5, 3), color='red')
    cross.make_rect_prism((1, 5, 1), (7, 5, 5), color='red')

    anchor.make_point((7, 5, 5), rho=1e5, color='gold')

    f = anchor | cross
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
