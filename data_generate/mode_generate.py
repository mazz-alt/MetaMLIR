
from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import block_diag
from scipy.linalg import cholesky as Matrix_sqrt


ArrayLike = Union[np.ndarray, float, int]




def _build_cv_noise_cholesky(dt: float) -> np.ndarray:
    """
    Build the 4x4 process noise Cholesky (same math as original CA generator).

    NOTE: This is a pure helper. It does NOT draw random numbers.
    """
    s_var = np.square(10)
    T2 = np.power(dt, 2)
    T4 = np.power(dt, 4)
    var_m = np.array([[T4 / 4, 0, 0, 0], [0, T4 / 4, 0, 0], [0, 0, T2, 0], [0, 0, 0, T2]]) * s_var
    return Matrix_sqrt(var_m)


def _chol_with_retry(var: np.ndarray, time_steps: int) -> np.ndarray:
    """
    Draw z ~ N(0, I) and compute noise = z @ L.T where L = chol(var).

    Keeps original behavior:
    - retries by re-sampling alpha outside this helper (caller controls that)
    - this helper itself does not loop; it will raise if chol fails
    """
    z = np.random.normal(size=(time_steps, var.shape[0]))
    L = np.linalg.cholesky(var)
    return z @ L.T


# 1. CV (Constant Velocity) Model
def generate_cv_trajectory(
    initial_position: np.ndarray,
    velocity: np.ndarray,
    time_steps: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate constant-velocity trajectory.

    NOTE: Behavior intentionally preserved:
    - position[0] is set to initial_position
    - loop starts at t=0 and uses position[t-1] -> at t=0 it reads position[-1]
      (this overwrites position[0] based on the last row, which is zeros initially)
    """
    position = np.zeros((time_steps, 2))
    position[0] = initial_position
    for t in range(0, time_steps):
        position[t] = position[t - 1] + velocity * dt
    return position, np.tile(velocity, (time_steps, 1))


# 2. CA (Constant Acceleration) Model
def generate_ca_trajectory(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    acceleration: np.ndarray,
    time_steps: int,
    sT: float,
    level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate constant-acceleration trajectory with additive noise.
    """
    position = np.zeros((time_steps, 2))
    velocity = np.zeros((time_steps, 2))
    position[0] = initial_position
    velocity[0] = initial_velocity

    chol_var = _build_cv_noise_cholesky(sT)
    noise = np.dot(np.random.randn(time_steps, 4), chol_var)

    for t in range(1, time_steps):
        velocity[t] = velocity[t - 1] + acceleration * sT + noise[t][2:4] * level
        position[t] = (
            position[t - 1]
            + velocity[t - 1] * sT
            + 0.5 * acceleration * (sT**2)
            + noise[t][0:2] * level
        )

    return position, velocity


# 4. Singer Model
def generate_singer_trajectory(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    time_steps: int,
    dt: float,
    level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Singer-model trajectory.

    Behavior preserved:
    - keep retry loop when covariance isn't PSD for Cholesky
    - alpha sampled from U(0.5, 2)
    - initial accelerations sampled from U(-20, 20)
    """
    flag = 0

    while flag == 0:
        alpha = np.random.uniform(0.5, 2)

        state = np.zeros((time_steps, 6))
        acclex, accley = np.random.uniform(-20, 20), np.random.uniform(-20, 20)
        state[0] = [initial_position[0], initial_velocity[0], acclex, initial_position[1], initial_velocity[1], accley]

        F = np.array(
            [
                [1, dt, (alpha * dt - 1 + np.exp(-alpha * dt)) / alpha**2, 0, 0, 0],
                [0, 1, (1 - np.exp(-alpha * dt)) / alpha, 0, 0, 0],
                [0, 0, np.exp(-alpha * dt), 0, 0, 0],
                [0, 0, 0, 1, dt, (alpha * dt - 1 + np.exp(-alpha * dt)) / alpha**2],
                [0, 0, 0, 0, 1, (1 - np.exp(-alpha * dt)) / alpha],
                [0, 0, 0, 0, 0, np.exp(-alpha * dt)],
            ]
        )

        s_var = np.square(10)
        T2 = np.power(dt, 2)
        T3 = np.power(dt, 3)

        q11 = (
            2 * alpha**3 * T3
            - 6 * alpha**2 * T2
            + 6 * alpha * dt
            + 3
            - 12 * alpha * dt * np.exp(-alpha * dt)
            - 3 * np.exp(-2 * alpha * dt)
        ) / (6 * alpha**5)
        q12 = (
            alpha**2 * T2
            - 2 * alpha * dt
            + 1
            - 2 * (1 - alpha * dt) * np.exp(-alpha * dt)
            + np.exp(-2 * alpha * dt)
        ) / (2 * alpha**4)
        q22 = (2 * alpha * dt - 3 + 4 * np.exp(-alpha * dt) - np.exp(-2 * alpha * dt)) / (2 * alpha**3)
        q13 = (1 - 2 * alpha * dt * np.exp(-alpha * dt) - np.exp(-2 * alpha * dt)) / (2 * alpha**3)
        q23 = (1 - 2 * np.exp(-alpha * dt) + np.exp(-2 * alpha * dt)) / (2 * alpha**2)
        q33 = (1 - np.exp(-2 * alpha * dt)) / (2 * alpha)

        var_m = 2 * alpha * s_var * np.array([[q11, q12, q13], [q12, q22, q23], [q13, q23, q33]])
        var = block_diag(var_m, var_m)

        try:
            noise = _chol_with_retry(var, time_steps)
            flag = 1
        except Exception:
            flag = 0
            continue

    for t in range(1, time_steps):
        state[t] = F @ state[t - 1] + noise[t] * level

    x = state[:, 0].reshape(len(state), 1)
    y = state[:, 3].reshape(len(state), 1)
    vx = state[:, 1].reshape(len(state), 1)
    vy = state[:, 4].reshape(len(state), 1)

    position = np.concatenate((x, y), axis=1)
    velocity = np.concatenate((vx, vy), axis=1)
    return position, velocity


def modified_rayleigh(scale: float, loc: float, size: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    """
    Generate samples from a modified Rayleigh distribution (non-zero mean style):
      x ~ N(loc, scale), y ~ N(0, scale), return sqrt(x^2 + y^2)
    """
    x = np.random.normal(loc=loc, scale=scale, size=size)
    y = np.random.normal(loc=0, scale=scale, size=size)
    return np.sqrt(x**2 + y**2)


# 5. CS Model
def generate_cs_trajectory(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    time_steps: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CS model as in original file.
    """
    scale = 5
    loc = 1.0
    num_samples = 1

    acclex = modified_rayleigh(scale, loc, size=num_samples)
    accley = modified_rayleigh(scale, loc, size=num_samples)

    alpha = 1 / time_steps

    state = np.zeros((time_steps, 6))
    state[0] = [initial_position[0], initial_velocity[0], acclex, initial_position[1], initial_velocity[1], accley]

    F = np.array(
        [
            [1, dt, (alpha * dt - 1 + np.exp(-alpha * dt)) / alpha**2, 0, 0, 0],
            [0, 1, (1 - np.exp(-alpha * dt)) / alpha, 0, 0, 0],
            [0, 0, np.exp(-alpha * dt), 0, 0, 0],
            [0, 0, 0, 1, dt, (alpha * dt - 1 + np.exp(-alpha * dt)) / alpha**2],
            [0, 0, 0, 0, 1, (1 - np.exp(-alpha * dt)) / alpha],
            [0, 0, 0, 0, 0, np.exp(-alpha * dt)],
        ]
    )

    G = np.array(
        [
            (-dt + alpha * dt**2 / 2 + (1 - np.exp(-alpha * dt)) / alpha) / alpha,
            dt - (1 - np.exp(-alpha * dt) / alpha),
            1 - np.exp(-alpha * dt),
        ]
    )

    G_new = np.concatenate((G, G), axis=0)

    for t in range(1, time_steps):
        mean_a = np.sqrt(np.square(np.mean(state[2])) + np.square(np.mean(state[5])))
        state[t] = F @ state[t - 1] + np.multiply(mean_a, G_new)

    x = state[:, 0].reshape(len(state), 1)
    y = state[:, 3].reshape(len(state), 1)
    vx = state[:, 1].reshape(len(state), 1)
    vy = state[:, 4].reshape(len(state), 1)

    position = np.concatenate((x, y), axis=1)
    velocity = np.concatenate((vx, vy), axis=1)
    return position, velocity


# 6. Jerk Model
def generate_jerk_trajectory(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    jerk_std: float,
    time_steps: int,
    dt: float,
    level: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate jerk-model trajectory.

    Behavior preserved:
    - retry loop for Cholesky success
    - alpha sampled from U(0, 3)
    - jx,jy from U(-2,2)
    - jerk_noise drawn once, used to derive ax,ay
    - noise applied via state[t] = F @ state[t-1] + noise[t]*level
    """
    flag = 0
    while flag == 0:
        alpha = np.random.uniform(0, 3)

        state = np.zeros((time_steps, 8))
        jx, jy = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
        jerk_noise = np.random.normal(0, jerk_std, 2)

        ax, ay = (jx - jerk_noise[0]) / (-alpha), (jy - jerk_noise[1]) / (-alpha)
        state[0] = [initial_position[0], initial_velocity[0], ax, jx, initial_position[1], initial_velocity[1], ay, jy]

        F = np.array(
            [
                [1, dt, dt**2 / 2, (2 - 2 * alpha * dt + (alpha**2) * (dt**2) - 2 * np.exp(-alpha * dt)) / (2 * alpha**3), 0, 0, 0, 0],
                [0, 1, dt, (alpha * dt - 1 + np.exp(-alpha * dt) / alpha**2), 0, 0, 0, 0],
                [0, 0, 1, (1 - np.exp(-alpha * dt) / alpha), 0, 0, 0, 0],
                [0, 0, 0, np.exp(-alpha * dt), 0, 0, 0, 0],
                [0, 0, 0, 0, 1, dt, dt**2 / 2, (2 - 2 * alpha * dt + (alpha**2) * (dt**2) - 2 * np.exp(-alpha * dt)) / (2 * alpha**3)],
                [0, 0, 0, 0, 0, 1, dt, (alpha * dt - 1 + np.exp(-alpha * dt) / alpha**2)],
                [0, 0, 0, 0, 0, 0, 1, (1 - np.exp(-alpha * dt) / alpha)],
                [0, 0, 0, 0, 0, 0, 0, np.exp(-alpha * dt)],
            ]
        )

        s_var = np.square(10)
        T2 = np.power(dt, 2)
        T3 = np.power(dt, 3)

        q11 = (1 / (2 * alpha**7)) * (
            alpha**5 * dt**5 / 10
            - alpha**4 * dt**4 / 2
            + 4 * alpha**3 * dt**3 / 3
            - 2 * alpha**2 * dt**2
            + 2 * alpha * dt
            - 3
            + 4 * np.exp(-alpha * dt)
            + 2 * alpha**2 * dt**2 * np.exp(-alpha * dt)
            - np.exp(-2 * alpha * dt)
        )
        q12 = (1 / (2 * alpha**6)) * (
            1
            - 2 * alpha * dt
            + 2 * alpha**2 * dt**2
            - alpha**3 * dt**3
            + alpha**4 * dt**4 / 4
            + np.exp(-2 * alpha * dt)
            + 2 * alpha * dt * np.exp(-alpha * dt)
            - 2 * np.exp(-alpha * dt)
            - alpha**2 * dt**2 * np.exp(-alpha * dt)
        )
        q13 = (1 / (2 * alpha**5)) * (
            2 * alpha * dt
            - alpha**2 * dt**2
            + alpha**3 * dt**3 / 3
            - 3
            - np.exp(-2 * alpha * dt)
            + 4 * np.exp(-alpha * dt)
            + alpha**2 * dt**2 * np.exp(-alpha * dt)
        )
        q14 = (1 / (2 * alpha**4)) * (1 + np.exp(-2 * alpha * dt) - 2 * np.exp(-alpha * dt) - alpha**2 * dt**2 * np.exp(-alpha * dt))
        q22 = (1 / (2 * alpha**5)) * (1 - np.exp(-2 * alpha * dt + 2 * alpha**3 * dt**3 / 3 + 2 * alpha * dt - 2 * alpha**2 * dt**2 - 4 * alpha * dt * np.exp(-alpha * dt)))
        q23 = (1 / (2 * alpha**4)) * (1 + alpha**2 * dt**2 - 2 * alpha * dt + 2 * alpha * dt * np.exp(-alpha * dt) + np.exp(-2 * alpha * dt) - 2 * np.exp(-alpha * dt))
        q24 = (1 / (2 * alpha**3)) * (1 - np.exp(-2 * alpha * dt) - 2 * alpha * dt * np.exp(-2 * alpha * dt))
        q33 = (1 / (2 * alpha**3)) * (4 * np.exp(-alpha * dt) - np.exp(-2 * alpha * dt) + 2 * alpha * dt - 3)
        q34 = (1 / (2 * alpha**2)) * (1 - 2 * np.exp(-alpha * dt) + np.exp(-2 * alpha * dt))
        q44 = (1 / (2 * alpha)) * (1 - np.exp(-2 * alpha * dt))

        var_m = 2 * alpha * s_var * np.array([[q11, q12, q13, q14], [q12, q22, q23, q24], [q13, q23, q33, q34], [q14, q24, q34, q44]])
        var = block_diag(var_m, var_m)

        try:
            noise = _chol_with_retry(var, time_steps)
            flag = 1
        except Exception:
            flag = 0
            continue

    for t in range(1, time_steps):
        state[t] = F @ state[t - 1] + noise[t] * level

    x = state[:, 0].reshape(len(state), 1)
    y = state[:, 4].reshape(len(state), 1)
    vx = state[:, 1].reshape(len(state), 1)
    vy = state[:, 5].reshape(len(state), 1)

    position = np.concatenate((x, y), axis=1)
    velocity = np.concatenate((vx, vy), axis=1)
    return position, velocity