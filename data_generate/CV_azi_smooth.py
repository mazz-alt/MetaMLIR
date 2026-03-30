
"""
Radar Tracking Environment
"""

import numpy as np
from scipy.linalg import cholesky as Matrix_sqrt



class Trajectory_Generator_2D_av(object):
    """
    2D trajectory generator with optional acceleration (av) augmentation.

    NOTE: This refactor preserves the original behavior exactly:
    - state_n is forcibly set to 10 inside __init__
    - azimuth_smooth logic unchanged (including x[0]=0 and overwrite-to-previous behavior)
    - trajectory() uses the same update order & RNG calls
    - dtype/shape of arrays preserved
    """

    def __init__(
        self,
        sT=0.1,
        data_len=50,
        state_n=10,
        bp_distance=1000,
        bp_dis_direction=30,
        bp_velocity=100,
        bp_vel_direction=30,
        dis_n=10,
        azi_n=8,
        TR=0,
        av=0,
        av_mode=0,
        tau=30,
    ):
        self.sT = sT
        self.N = data_len

        d_x = bp_distance * np.cos(bp_dis_direction * np.pi / 180)
        d_y = bp_distance * np.sin(bp_dis_direction * np.pi / 180)
        v_x = bp_velocity * np.cos(bp_vel_direction * np.pi / 180)
        v_y = bp_velocity * np.sin(bp_vel_direction * np.pi / 180)
        self.bp = np.array([[d_x, d_y, v_x, v_y]], "float64")

        # Keep original behavior: override incoming state_n.
        state_n = 10

        self.var_m, self.chol_var = self._build_process_noise(self.sT, state_n)

        self.azi_n = azi_n / 1000
        self.dis_n = dis_n
        self.R = np.array([[(azi_n / 1000) ** 2, 0], [0, dis_n**2]])

        self.turn_rate = TR

        self.av = av
        self.av_mode = av_mode

        self.tau = tau
        self.alpha = 1 / self.tau
        self.AT = self.alpha * self.sT
        self.E_AT = np.exp(-self.AT)

        self.AV_M0 = np.array([[(self.sT**2) / 2, self.sT]], "float64")
        self.AV_M1 = np.array(
            [[(self.AT - 1 + self.E_AT) / (self.alpha**2), (1 - self.E_AT) / self.alpha, self.E_AT]],
            "float64",
        )

    @staticmethod
    def _build_process_noise(sT: float, state_n: float):
        """
        Build process noise covariance and its Cholesky factor.
        Kept identical to original math.
        """
        s_var = np.square(state_n)
        T2 = np.power(sT, 2)
        T4 = np.power(sT, 4)
        var_m = np.array(
            [[T4 / 4, 0, 0, 0], [0, T4 / 4, 0, 0], [0, 0, T2, 0], [0, 0, 0, T2]]
        ) * s_var
        chol_var = Matrix_sqrt(var_m)
        return var_m, chol_var

    @staticmethod
    def _transition_matrix(sT: float, turn_rate_deg: float):
        """
        Build F_c and w (rad/s), then transpose to match original.
        """
        if turn_rate_deg == 0.0:
            w = 0.0
            F = np.array(
                [[1.0, 0.0, sT, 0.0], [0.0, 1.0, 0.0, sT], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
        else:
            w = turn_rate_deg * np.pi / 180
            F = np.array(
                [
                    [1.0, 0.0, np.sin(w * sT) / w, (np.cos(w * sT) - 1) / w],
                    [0.0, 1.0, -(np.cos(w * sT) - 1) / w, np.sin(w * sT) / w],
                    [0.0, 0.0, np.cos(w * sT), -np.sin(w * sT)],
                    [0.0, 0.0, np.sin(w * sT), np.cos(w * sT)],
                ]
            )

        F_c = np.transpose(F, [1, 0])
        return F_c, w

    # 相位补偿（保持原逻辑不变）
    def azimuth_smooth(self, x):
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

    def _apply_av_compensation(self, state_temp: np.ndarray) -> np.ndarray:
        """
        Apply acceleration compensation.
        Keeps the original branch logic and self.av update rule.
        """
        if self.av == 0:
            return state_temp

        if self.av_mode == 0:
            a_comp = self.av * self.AV_M0
            return state_temp + np.array([a_comp[0, 0], a_comp[0, 0], a_comp[0, 1], a_comp[0, 1]])

        a_comp = self.av * self.AV_M1
        state_temp = state_temp + np.array([a_comp[0, 0], a_comp[0, 0], a_comp[0, 1], a_comp[0, 1]])
        self.av = a_comp[0, 2]
        return state_temp

    def trajectory(self, level):
        """
        Generate trajectory (with noise scaled by `level`).
        Returns: (Tj_n, F_c, w) exactly like original.
        """
        F_c, w = self._transition_matrix(self.sT, self.turn_rate)

        Tj = np.array([[0 for _ in range(4)] for _ in range(self.N)], "float64")
        state_temp = self.bp

        for i in range(self.N):
            Tj[i, :] = state_temp
            state_temp = np.dot(state_temp, F_c)
            state_temp = self._apply_av_compensation(state_temp)

        # Keep RNG call & multiplication order identical
        Tj_n = Tj + level * np.dot(np.random.randn(self.N, 4), self.chol_var)
        return Tj_n, F_c, w