#!/usr/bin/env python3
import numpy as np
from typing import Callable


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

    return np.array((elev, azim, roll))


def principal_step_rk4(
    t: float,
    dt: float,
    w: np.ndarray,
    q: np.ndarray,
    I: np.ndarray,
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
        t: float, dt: float, w: np.ndarray, I: np.ndarray, gamma: Callable
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
