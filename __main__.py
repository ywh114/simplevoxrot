#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from util import voxel, simulate

IPYNB = False

if __name__ == '__main__':
    # Initialize voxel space
    extent = 16
    vs = voxel.VoxelSpace(extent)

    cross = voxel.Voxel(vs)
    book = voxel.Voxel(vs)
    tee = voxel.Voxel(vs)
    anchor = voxel.Voxel(vs)

    # Create book components
    # book.make_rect_prism((9, 9, 1), (0, 0, 3), color='green')  # Front cover
    # book.make_rect_prism((9, 9, 1), (0, 0, 0), color='green')  # Back cover
    # book.make_rect_prism((9, 1, 4), (0, 0, 0), color='green')  # Spine
    # book.make_rect_prism((9, 7, 2), (0, 1, 1), rho=0.5, color='white')  # Pages

    # Create cross componenets
    # cross.make_rect_prism((5, 1, 1), (5, 5, 5), color='red')
    # cross.make_rect_prism((1, 1, 5), (7, 5, 3), color='red')
    # cross.make_rect_prism((1, 5, 1), (7, 5, 5), color='red')

    # Create tee componenets
    tee.make_rect_prism((1, 1, 5), (7, 5, 3), rho=1, color='gray')
    tee.make_rect_prism((1, 5, 1), (7, 5, 5), rho=10, color='red')

    # Testing purposes
    # anchor.make_point((0, 0, 0), rho=0, color='gold')

    f = anchor | book | cross | tee

    # Initial conditions
    T = 10.0
    dt = 0.01
    w0 = np.array((20, 0, 5))

    def gamma(
        t: float,
        dt: float,
        ws: np.ndarray,
        I: np.ndarray,  # noqa: E741
    ) -> np.ndarray:
        return np.zeros(3)

    ts, ws_p, aers_l, qs_l, ani = simulate.simulate(
        f, w0, T, dt, gamma, True, IPYNB
    )

    plt.close()

    ws_x, ws_y, ws_z = ws_p.T
    azim, elev, roll = aers_l.T
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('Principal frame $\\omega$')
    ax1.plot(ts, ws_x, label='$\\omega_x$')
    ax1.plot(ts, ws_y, label='$\\omega_y$')
    ax1.plot(ts, ws_z, label='$\\omega_z$')
    ax2.set_title('Lab frame angles')
    ax2.plot(ts, azim, label='Azimuth')
    ax2.plot(ts, elev, label='Elevation')
    ax2.plot(ts, roll, label='Roll')
    plt.show()
