#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from util import voxel, simulate

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

    # Get inertia properties
    I_p, R0 = f.principal_axes
    x_c, y_c, z_c = com = f.center_of_mass

    # Set up plot centered on COM
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x_c - extent, x_c + extent))
    ax.set_ylim((y_c - extent, y_c + extent))
    ax.set_zlim((z_c - extent, z_c + extent))
    ax.set_box_aspect([1, 1, 1])
    ax.voxels(f.filled, facecolors=f.facecolors, edgecolor='k')
    plt.axis('off')

    # Initial conditions
    T = 10.0
    dt = 0.01
    steps = int(T / dt)
    w_p0 = np.array((20, 0, 5))
    q0 = simulate.mat2q(R0)

    def gamma(t: float, dt: float, ws: np.ndarray, I: np.ndarray) -> np.ndarray:
        return np.zeros(3)

    ts = np.zeros(steps)
    ws_p = np.zeros(shape=(steps, 3))
    qs_l = np.zeros(shape=(steps, 4))
    aers_l = np.zeros(shape=(steps, 3))
    ws_p[0] = w_p0
    qs_l[0] = q0
    aers_l[0] = np.zeros(3)
    for i in range(1, steps):
        ts[i] = i * dt
        ws_p[i], qs_l[i] = simulate.principal_step_rk4(
            ts[i], dt, ws_p[i - 1], qs_l[i - 1], I_p, gamma
        )

        elev, azim, roll = aers_l[i] = np.degrees(simulate.q2aer(qs_l[i]))
        print(
            f'Time: {ts[i]:.2f}s | '
            f'Elevation: {elev:.2f}° | '
            f'Azimuth: {azim:.2f}° | '
            f'Roll: {roll:.2f}° | '
        )

        ax.view_init(-azim, -elev, -roll)
        plt.title(
            f'Elevation: {elev:.2f}°, Azimuth: {azim:.2f}°, Roll: {roll:.2f}°'
        )

        plt.draw()
        plt.pause(0.001)

    plt.close()

    print(ts.size)
    print(ws_p.size)

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
