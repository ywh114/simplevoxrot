#!/usr/bin/env python3
import numpy as np
import itertools
import matplotlib.pyplot as plt

from typing import Callable
from util.voxel import Voxel
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation


def q2mat(q: np.ndarray) -> np.ndarray:
    """
    https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
    Covert a quaternion into a full three-dimensional rotation matrix.

    Arguments:
        q: The quaternion.
    Returns:
        The rotation matrix.
    """
    # Extract the values from Q
    q0, q1, q2, q3 = q

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])


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
    fig.set_size_inches(8, 4.5, True)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim((x_c - extent, x_c + extent))
    ax.set_ylim((y_c - extent, y_c + extent))
    ax.set_zlim((z_c - extent, z_c + extent))  # type: ignore
    ax.set_box_aspect([1, 1, 1])  # type: ignore
    ax.voxels(f.filled, facecolors=f.facecolors, edgecolor='k')  # type: ignore
    plt.axis('off')

    def update(azim: float, elev: float, roll: float, t: float) -> Axes:
        ax.view_init(-azim, -elev, -roll)  # type: ignore
        ax.set_title(
            f'Elevation: {elev:.2f}°, Azimuth: {azim:.2f}°, Roll: {roll:.2f}°\n'
            f'Time: {t:.2f}/{T}'
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
            _ = update(azim, elev, roll, i * dt)

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


def angular_momentum(
    f: Voxel, ws_p: np.ndarray, qs_l: np.ndarray
) -> np.ndarray:
    """
    Compute the angular momentum of a body.

    Arguments:
        f: The `Voxel` object.
        ws_p: Angular velocities in the principal frame.
        qs_l: Quaternions in the lab frame.
    Returns:
        An array of the angular momentums.
    """
    I_p, _ = f.principal_axes
    Is = np.array(
        tuple(
            # q2mat converts a quaternion (used internally for simulation) to
            # a rotation matrix.
            q2mat(q) @ I_p  # Apply each rotation to I_p to get I in lab frame.
            for q in qs_l
        )
    )
    return np.array(
        tuple(
            I @ w  # Compute L = I @ omega at each omega.
            for I, w in zip(Is, ws_p)  # noqa: E741
        )
    )


def generalized_momenta(
    f: Voxel, qs_l: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute generalized momenta p_phi and p_psi of an axisymmetric object.

    Arguments:
        f: The `Voxel` object.
        qs_l: Quaternions in the lab frame.
    Returns:
        An array of generalized momenta.
    """

    def get_axisymmetric(I_p: np.ndarray) -> tuple[float, float] | None:
        diag = np.diag(I_p.round(4))
        au = np.array(tuple(p == q for p, q in itertools.combinations(diag, 2)))
        if not any(au):
            return None

        m = au.argmax()
        return diag[2 - m], diag[1 - m]

    thetas, phis, psis = q2euler(qs_l.T)

    cos_theta = np.cos(thetas)
    phi_dot = np.gradient(phis)
    psi_dot = np.gradient(psis)

    I_p, _ = f.principal_axes
    I_a, I_t = at if (at := get_axisymmetric(I_p)) is not None else (None, None)

    if I_a is None:
        raise ValueError(I_p, 'is not axisymmetric.')

    omega = psi_dot + cos_theta * phi_dot
    p_psi = I_a * omega
    p_phi = I_a * omega * cos_theta + I_t * np.sin(thetas) ** 2 * phi_dot

    # Filter out gimbal lock discontinuities in Euler angles.
    compare = 50
    threshold = 0.1
    p_psi[np.abs(p_psi) > p_psi[compare] + threshold] = p_psi[compare]
    p_phi[np.abs(p_phi) > p_psi[compare] + threshold] = p_phi[compare]

    return p_psi, p_phi
