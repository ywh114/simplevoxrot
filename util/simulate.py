#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable
from util.voxel import Voxel
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation


def mat2q(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix into its quaterion representation.

    Arguments:
        R: The rotation matrix.
    Returns:
        The quaternion representation.
    """
    qw = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qw, qx, qy, qz])


def q2aer(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to (azim, elev, roll) angles.

    Arguments:
        q: The quaternion.
    Returns:
        The rotation angles.
    """
    qw, qx, qy, qz = q

    # Roll (around view axis)
    roll = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    # Elevation (pitch)
    sin_pitch = 2 * (qw * qy - qz * qx)
    sin_pitch = np.clip(sin_pitch, -1, 1)  # Avoid numerical errors
    elev = np.arcsin(sin_pitch)
    # Azimuth (yaw)
    azim = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))

    return np.array((azim, elev, roll))


def q2euler(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to Euler angles (θ, φ, ψ) using the 3-1-3 convention.

    Arguments:
        q: The quaternion.
    Returns:
        The Euler angles.
    """
    qw, qx, qy, qz = q

    # theta (nutation)
    theta = np.arccos(2 * (qw**2 + qx**2) - 1)
    # phi (precession)
    phi = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    # psi (spin)
    psi = np.arctan2(2 * (qw * qy - qx * qz), 1 - 2 * (qy**2 + qz**2))

    return np.array((theta, phi, psi))


def principal_step_rk4(
    t: float,
    dt: float,
    w: np.ndarray,
    q: np.ndarray,
    I: np.ndarray,  # noqa: E741
    gamma: Callable,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform RK4 integration for angular velocity (body frame) and orientation (rotation matrix).

    Args:
        t: Current time.
        dt: Time step.
        w: Angular velocity in body frame.
        q: Quaternion in lab frame.
        I: Inertia tensor (shape: (3, 3)).
        gamma: Torque function gamma(t, dt, ws, I) -> torque in body frame (shape: (3,)).

    Returns:
        ws_new: Updated angular velocity (body frame).
        R_new: Updated rotation matrix.
    """

    def dwdt_p(
        t: float,
        dt: float,
        w: np.ndarray,
        I: np.ndarray,  # noqa: E741
        gamma: Callable,
    ) -> np.ndarray:
        """Compute dwdt in the principal frame."""
        torque = gamma(t, dt, w, I)
        dw = np.array(
            (
                (torque[0] - (I[2, 2] - I[1, 1]) * w[1] * w[2]) / I[0, 0],
                (torque[1] - (I[0, 0] - I[2, 2]) * w[0] * w[2]) / I[1, 1],
                (torque[2] - (I[1, 1] - I[0, 0]) * w[0] * w[1]) / I[2, 2],
            )
        )
        return dw * dt

    def dqdt(q: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Compute dqdt."""
        wx, wy, wz = w
        qw, qx, qy, qz = q
        return 0.5 * np.array(
            (
                (-wx * qx - wy * qy - wz * qz),
                (wx * qw + wz * qy - wy * qz),
                (wy * qw - wz * qx + wx * qz),
                (wz * qw + wy * qx - wx * qy),
            )
        )

    k1 = dwdt_p(t, dt, w, I, gamma)
    k2 = dwdt_p(t + dt / 2, dt, w + k1 / 2, I, gamma)
    k3 = dwdt_p(t + dt / 2, dt, w + k2 / 2, I, gamma)
    k4 = dwdt_p(t + dt, dt, w + k3, I, gamma)
    w_new = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    l1 = dqdt(q, w)
    l2 = dqdt(q + 0.5 * dt * l1, w)
    l3 = dqdt(q + 0.5 * dt * l2, w)
    l4 = dqdt(q + dt * l3, w)
    q_new = q + (dt / 6) * (l1 + 2 * l2 + 2 * l3 + l4)

    return w_new, q_new / np.linalg.norm(q_new)


def simulate(
    f: Voxel,
    w0: np.ndarray,
    T: float,
    dt: float,
    gamma: Callable,
    animate: bool = True,
    _ipynb: bool = False,
    _ipynb_downsample_ratio: int = 1,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, FuncAnimation | None
]:
    """
    Simulate the rotational dynamics of a voxel object and animate its motion.

    Args:
        f: The voxel object to simulate.
        w0: Initial angular velocity (3D vector) in the body frame.
        T: Total simulation time (seconds).
        dt: Time step (seconds) for numerical integration.
        gamma: Torque function with signature
            `gamma(t, dt, w, I) -> np.ndarray`, where:
                - `t`: Current time.
                - `dt`: Time step.
                - `w`: Current angular velocity.
                - `I`: Inertia tensor.
        animate: Whether or not to animate.
        _ipynb: If True, returns an animation for display in a Jupyter notebook.
        _ipynb_downsample_ratio: Downsampling ratio for the animation frames
            (skips frames to improve performance).
    Returns:
        A tuple containing:
        - ts: Array of time steps (shape: (steps,)).
        - ws_p: Angular velocities in the principal frame (shape: (steps, 3)).
        - aers_l: Azimuth, elevation, and roll angles in the lab frame
            (shape: (steps, 3)).
        - qs_l: Quaternions representing orientation in the lab frame
            (shape: (steps, 4)).
        - ani: Animation object (if `animate and _ipynb`), otherwise None.
    Notes:
        - For non-notebook use (`_ipynb=False`), the simulation updates the
            plot interactively.
        - For notebook use (`_ipynb=True`), the animation is rendered using
            `FuncAnimation`.
        - Downsampling (`_ipynb_downsample_ratio > 1`) reduces the number of
            frames rendered, improving performance.
    """
    # Get inertia properties
    I_p, R0 = f.principal_axes
    x_c, y_c, z_c = f.center_of_mass

    extent = f.vs.extent
    steps = int(T / dt)
    q0 = mat2q(R0)

    ts = np.zeros(steps)
    ws_p = np.zeros(shape=(steps, 3))
    qs_l = np.zeros(shape=(steps, 4))
    aers_l = np.zeros(shape=(steps, 3))

    ws_p[0] = w0
    qs_l[0] = q0
    aers_l[0] = np.zeros(3)

    # Set up plot centered on COM
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x_c - extent, x_c + extent))
    ax.set_ylim((y_c - extent, y_c + extent))
    ax.set_zlim((z_c - extent, z_c + extent))  # type: ignore
    ax.set_box_aspect([1, 1, 1])  # type: ignore
    ax.voxels(f.filled, facecolors=f.facecolors, edgecolor='k')  # type: ignore
    plt.axis('off')

    def update(azim: float, elev: float, roll: float) -> Axes:
        ax.view_init(-azim, -elev, -roll)  # type: ignore
        ax.set_title(
            f'Elevation: {elev:.2f}°, Azimuth: {azim:.2f}°, Roll: {roll:.2f}°'
        )
        return ax

    # Simulation loop
    for i in range(1, steps):
        ts[i] = i * dt
        ws_p[i], qs_l[i] = principal_step_rk4(
            ts[i], dt, ws_p[i - 1], qs_l[i - 1], I_p, gamma
        )

        azim, elev, roll = aers_l[i] = np.degrees(q2aer(qs_l[i]))

        if animate and not _ipynb:
            _ = update(azim, elev, roll)

            plt.draw()
            plt.pause(0.001)

    plt.close(fig)

    ani = None
    if animate and _ipynb:

        def _update(i: int) -> Axes:
            azim, elev, roll = aers_l[i]
            return update(azim, elev, roll)

        ani = FuncAnimation(
            fig,
            _update,  # type: ignore
            frames=range(0, steps, _ipynb_downsample_ratio),
            interval=50,
            blit=False,
            repeat=True,
        )

    return ts, ws_p, aers_l, qs_l, ani
