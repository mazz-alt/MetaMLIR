# /mnt/data/DTS_maneuver_trajectory_all.py
import os
from typing import List, Tuple

import numpy as np
import torch

from data_generate.CV_azi_smooth import Trajectory_Generator_2D_av as TG2D
from data_generate.mode_generate import generate_ca_trajectory, generate_jerk_trajectory, generate_singer_trajectory


def normalize_to_column(x) -> np.ndarray:
    """
    Standardize (x-mean)/std then reshape to (len(x), 1).

    NOTE: Intentionally keeps the original behavior:
    - accepts list/np.ndarray
    - uses np.mean/np.std
    """
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std
    x = np.array(x)
    x = x.reshape(len(x), 1)
    return x


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson correlation of two vectors with same shape.
    Behavior kept identical to original implementation.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"
    covariance = np.mean((x - np.mean(x)) * (y - np.mean(y)))
    std_x = np.std(x)
    std_y = np.std(y)
    correlation = covariance / (std_x * std_y)
    return float(correlation)


def calculate_curvature(x, y) -> np.ndarray:
    """
    Curvature of a 2D curve.
    """
    x = np.array(x)
    y = np.array(y)

    dx = np.gradient(x, edge_order=1)
    dy = np.gradient(y)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** (3 / 2)
    return curvature


def azimuth_smooth(x: np.ndarray) -> np.ndarray:
    """
    Phase compensation smoothing (kept exactly as original).
    """
    xl = np.size(x)
    x[0] = 0
    for i in range(1, xl):
        if x[i] - x[i - 1] > np.pi:
            x[i] = x[i - 1] - 2 * np.pi
        elif x[i] - x[i - 1] < -np.pi:
            x[i] = x[i - 1] + 2 * np.pi
        elif abs(x[i] - x[i - 1]) < np.pi:
            x[i] = x[i - 1]
    return x


def maneuvering_trajectory(
    bp: np.ndarray,
    turn_rates: List[float],
    segment_lengths: List[int],
    level: float,
) -> Tuple[TG2D, np.ndarray, List[np.ndarray], List[float]]:
    """
    Generate piecewise maneuvering trajectory with TG2D.

    Kept identical to original:
    - TG2D instantiated with TR=turn_rates (even though later overwritten)
    - per segment: N = ts + 1; store my_traj[:ts, :]
    - bp updated by my_traj[-1, :]
    """
    traj_gen = TG2D(TR=turn_rates)
    traj_gen.bp = bp

    traj_segments: List[np.ndarray] = []
    f_all: List[np.ndarray] = []
    w_all: List[float] = []

    for tr, ts in zip(turn_rates, segment_lengths):
        traj_gen.N = ts + 1
        traj_gen.turn_rate = tr

        my_traj, my_f, w = traj_gen.trajectory(level)
        traj_segments.append(my_traj[:ts, :])
        f_all.append(my_f)
        w_all.append(w)

        traj_gen.bp = my_traj[-1, :]

    traj_all = np.vstack(traj_segments)
    return traj_gen, traj_all, f_all, w_all


def _sample_initial_state() -> Tuple[np.ndarray, float, float, float, float]:
    """
    Sample initial position/velocity in polar-like parameterization.
    Keeps original sampling distributions and order of RNG calls.
    """
    bp_distance = np.random.uniform(1000, 10000)
    bp_dis_direction = (np.random.random() - 0.5) * 2 * 180
    bp_velocity = np.random.uniform(100, 200)
    bp_vel_direction = (np.random.random() - 0.5) * 2 * 180

    d_x = bp_distance * np.cos(bp_dis_direction * np.pi / 180)
    d_y = bp_distance * np.sin(bp_dis_direction * np.pi / 180)
    v_x = bp_velocity * np.cos(bp_vel_direction * np.pi / 180)
    v_y = bp_velocity * np.sin(bp_vel_direction * np.pi / 180)

    bp = np.array([[d_x, d_y, v_x, v_y]], "float64")
    return bp, d_x, d_y, v_x, v_y


def _class_from_index(i: int, total: int, num_class: int) -> int:
    """
    Keep the original 'if' ladder behavior.
    """
    if i >= 0 and i < total // num_class:
        return 0
    if i >= total // num_class and i < (total // num_class) * 2:
        return 1
    if i >= (total // num_class) * 2 and i < (total // num_class) * 3:
        return 2
    if i >= (total // num_class) * 3 and i < (total // num_class) * 4:
        return 3
    if i >= (total // num_class) * 4 and i < (total // num_class) * 5:
        return 4
    return 0


def _generate_single_mode_traj_xy(
    mode: int,
    bp: np.ndarray,
    d_x: float,
    d_y: float,
    v_x: float,
    v_y: float,
    ts: int,
    dt: float,
    level: float,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Generate a single segment (ts points) trajectory in XY for a given mode.
    Returns:
      - traj_xy: (ts, 2)
      - updated d_x, d_y, v_x, v_y (end state)
    Behavior kept identical to original switch logic and RNG calls.
    """
    if mode == 0:  # CV
        tr = [0]
        seg_ts = [ts]
        _, my_traj_all, _, _ = maneuvering_trajectory(bp, tr, seg_ts, level)
        traj_xy = my_traj_all[:, :2]
        d_x = traj_xy[-1][0]
        d_y = traj_xy[-1][1]
        v_x = my_traj_all[-1, 2]
        v_y = my_traj_all[-1, 3]
        return traj_xy, d_x, d_y, v_x, v_y

    if mode == 1:  # CA
        a = np.random.randint(1, 10)
        ca_pos, ca_vel = generate_ca_trajectory(
            np.array([d_x, d_y]),
            np.array([v_x, v_y]),
            np.array([a, a]),
            ts,
            dt,
            level,
        )
        traj_xy = ca_pos
        d_x = traj_xy[-1][0]
        d_y = traj_xy[-1][1]
        v_x = ca_vel[-1][0]
        v_y = ca_vel[-1][1]
        return traj_xy, d_x, d_y, v_x, v_y

    if mode == 2:  # CT
        tr = [np.random.randint(-100, 100)]
        seg_ts = [ts]
        _, my_traj_all, _, _ = maneuvering_trajectory(bp, tr, seg_ts, level)
        traj_xy = my_traj_all[:, :2]
        d_x = traj_xy[-1][0]
        d_y = traj_xy[-1][1]
        v_x = my_traj_all[-1, 2]
        v_y = my_traj_all[-1, 3]
        return traj_xy, d_x, d_y, v_x, v_y

    if mode == 3:  # Singer
        singer_pos, singer_vel = generate_singer_trajectory(
            np.array([d_x, d_y]),
            np.array([v_x, v_y]),
            ts,
            dt,
            level,
        )
        traj_xy = singer_pos
        d_x = traj_xy[-1][0]
        d_y = traj_xy[-1][1]
        v_x = singer_vel[-1][0]
        v_y = singer_vel[-1][1]
        return traj_xy, d_x, d_y, v_x, v_y

    if mode == 4:  # Jerk
        a = np.random.randint(1, 10)  # kept even if unused downstream
        jerk_pos, jerk_vel = generate_jerk_trajectory(
            np.array([d_x, d_y]),
            np.array([v_x, v_y]),
            3,
            ts,
            dt,
            level,
        )
        traj_xy = jerk_pos
        d_x = traj_xy[-1][0]
        d_y = traj_xy[-1][1]
        v_x = jerk_vel[-1][0]
        v_y = jerk_vel[-1][1]
        return traj_xy, d_x, d_y, v_x, v_y

    # fallback (should not hit)
    tr = [0]
    seg_ts = [ts]
    _, my_traj_all, _, _ = maneuvering_trajectory(bp, tr, seg_ts, level)
    traj_xy = my_traj_all[:, :2]
    d_x = traj_xy[-1][0]
    d_y = traj_xy[-1][1]
    v_x = my_traj_all[-1, 2]
    v_y = my_traj_all[-1, 3]
    return traj_xy, d_x, d_y, v_x, v_y


def main() -> None:
    # seed : train 2026 val 2024 test 2025
    seeds = [2026, 2024, 2025]
    train_nums = [100000, 0, 25000]

    num_class = 5

    for flag in range(3):
        np.random.seed(seeds[flag])
        train_num = train_nums[flag]

        labels: List[int] = []
        traj_all: List[np.ndarray] = []
        traj_move: List[np.ndarray] = []
        traj_infor_all: List[np.ndarray] = []

        for i in range(train_num):
            bp, d_x, d_y, v_x, v_y = _sample_initial_state()

            dt = 0.1
            ts_list: List[int] = []
            traj_segments: List[np.ndarray] = []

            azi_n = np.random.uniform(0, 0.002)
            dis_n = np.random.uniform(0, 4)

            mode_class = _class_from_index(i, train_num, num_class)
            num = 1  # kept

            for j in range(num):
                ts_list.append(100)
                data_len = 100
                mode = mode_class
                level = 1  # kept

                traj_xy, d_x, d_y, v_x, v_y = _generate_single_mode_traj_xy(
                    mode=mode,
                    bp=bp,
                    d_x=d_x,
                    d_y=d_y,
                    v_x=v_x,
                    v_y=v_y,
                    ts=ts_list[j],
                    dt=dt,
                    level=level,
                )
                traj_segments.append(traj_xy)

            if num == 1:
                traj_xy_all = traj_segments[0]
            else:
                traj_xy_all = np.concatenate((traj_segments[0], traj_segments[1]), axis=0)
                traj_xy_all = np.delete(traj_xy_all, -ts_list[1], axis=0)

            change_point = ts_list[0]  # kept (unused)
            obser = np.array([[0 for _ in range(2)] for _ in range(len(traj_xy_all))], "float64")

            obser[:, 0] = (
                azimuth_smooth(np.arctan2(traj_xy_all[:, 1], traj_xy_all[:, 0]))
                + np.arctan2(traj_xy_all[:, 1], traj_xy_all[:, 0])
                + np.random.normal(0, azi_n, len(traj_xy_all)) * level
            )
            obser[:, 1] = (
                np.sqrt(np.square(traj_xy_all[:, 0]) + np.square(traj_xy_all[:, 1]))
                + np.random.normal(0, dis_n, len(traj_xy_all)) * level
            )

            labels.append(mode_class)

            trajx = obser[:, 1] * np.cos(obser[:, 0])
            trajy = obser[:, 1] * np.sin(obser[:, 0])

            # Acceleration correlation
            time_steps = np.linspace(0, 10, ts_list[0])
            velocity_x = np.gradient(trajx, time_steps)
            velocity_y = np.gradient(trajy, time_steps)

            acceleration_x = np.gradient(velocity_x, time_steps)
            acceleration_y = np.gradient(velocity_y, time_steps)

            win_size = 50
            acc_x_cov: List[float] = []
            acc_y_cov: List[float] = []
            for k in range(data_len):
                if k + 1 + win_size <= data_len:
                    acc_x_cov.append(
                        pearson_correlation(
                            acceleration_x[k : k + win_size],
                            acceleration_x[k + 1 : k + 1 + win_size],
                        )
                    )
                    acc_y_cov.append(
                        pearson_correlation(
                            acceleration_y[k : k + win_size],
                            acceleration_y[k + 1 : k + 1 + win_size],
                        )
                    )
                else:
                    acc_x_cov.append(acc_x_cov[k - win_size])
                    acc_y_cov.append(acc_y_cov[k - win_size])

            # Normalization
            velocity_x = normalize_to_column(velocity_x)
            velocity_y = normalize_to_column(velocity_y)

            # NOTE: This is intentionally kept identical to original (even if odd).
            acceleration_x = normalize_to_column(velocity_x)
            acceleration_y = normalize_to_column(velocity_y)

            acc_x_cov = np.array(acc_x_cov).reshape(len(acc_x_cov), 1)
            acc_y_cov = np.array(acc_y_cov).reshape(len(acc_y_cov), 1)



            # Curvature
            curvature = calculate_curvature(trajx, trajy)
            curvature = normalize_to_column(curvature)

            # Trajectory normalization
            trajx = trajx.reshape(data_len, 1)
            trajy = trajy.reshape(data_len, 1)

            mean_x = trajx.mean()
            std_x = trajx.std()
            mean_y = trajy.mean()
            std_y = trajy.std()
            trajx = (trajx - mean_x) / std_x
            trajy = (trajy - mean_y) / std_y

            # Observation distance diff
            traj_de = [0]
            for kk in range(1, data_len):
                traj_de.append(obser[kk, 1] - obser[kk - 1, 1])
            traj_de = normalize_to_column(traj_de)

            # FFT phase
            sampling_rate = 50
            T = 1.0 / sampling_rate
            t = np.arange(0.0, 10.0, T)

            signal = obser[:, 1]
            fft_signal = np.fft.fft(signal)

            n = len(t)
            frequencies = np.fft.fftfreq(n, T)  # kept (unused)

            magnitude = np.abs(fft_signal)  # kept (unused)
            phase = np.angle(fft_signal)
            phase = normalize_to_column(phase)

            traj_features = np.concatenate((trajx, trajy, traj_de, curvature), axis=1)
            traj_all.append(traj_features)
            
            traj_move_features = np.concatenate(
                (velocity_x, velocity_y, acceleration_x, acceleration_y, acc_x_cov, acc_y_cov, phase),
                axis=1,
            )
            
            traj_move.append(traj_move_features)

            traj_infor = np.concatenate(
                (
                    trajx,
                    trajy,
                    traj_de,
                    curvature,
                    velocity_x,
                    velocity_y,
                    acceleration_x,
                    acceleration_y,
                    acc_x_cov,
                    acc_y_cov,
                    phase,
                ),
                axis=1,
            )
            traj_infor_all.append(traj_infor)

        labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
        traj_all_tensor = torch.tensor(np.array(traj_all), dtype=torch.float32)
        traj_move_tensor = torch.tensor(np.array(traj_move), dtype=torch.float32)
        traj_infor_tensor = torch.tensor(np.array(traj_infor_all), dtype=torch.float32)

        mydata_1 = {"samples": traj_all_tensor, "labels": labels_tensor}
        mydata_2 = {"samples": traj_move_tensor, "labels": labels_tensor}
        mydata_3 = {"samples": traj_infor_tensor, "labels": labels_tensor}

        number = "9"

        out_dir = f"./dataset/motion{number}_3"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if flag == 0:
            torch.save(mydata_3, f"{out_dir}/train.pt")
        elif flag == 1:
            torch.save(mydata_3, f"{out_dir}/val.pt")
        elif flag == 2:
            torch.save(mydata_3, f"{out_dir}/test.pt")


if __name__ == "__main__":
    main()