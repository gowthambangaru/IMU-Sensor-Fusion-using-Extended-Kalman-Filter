"""
IMU Simulator — Phase 1
Simulates accelerometer and gyroscope readings with
realistic noise and bias drift for EKF testing.
"""

import numpy as np


class IMUSimulator:
    def __init__(self, dt=0.01, duration=30.0, seed=42):
        """
        Args:
            dt       : timestep in seconds (100 Hz)
            duration : total simulation time in seconds
            seed     : random seed for reproducibility
        """
        np.random.seed(seed)
        self.dt = dt
        self.duration = duration
        self.timesteps = int(duration / dt)
        self.t = np.linspace(0, duration, self.timesteps)

        # --- Noise parameters (realistic IMU values) ---
        # Accelerometer white noise (m/s^2)
        self.accel_noise_std = 0.05

        # Gyroscope white noise (rad/s)
        self.gyro_noise_std = 0.01

        # Gyroscope bias drift (rad/s) — slowly changing offset
        self.gyro_bias_std = 0.003

    def generate_true_trajectory(self):
        """
        Generate smooth ground truth roll, pitch, yaw angles (radians).
        Uses sinusoidal motion to simulate realistic attitude changes.
        """
        roll  = 0.3 * np.sin(2 * np.pi * 0.1 * self.t)           # ±0.3 rad (~17 deg)
        pitch = 0.2 * np.sin(2 * np.pi * 0.05 * self.t + 0.5)    # ±0.2 rad (~11 deg)
        yaw   = 0.4 * np.sin(2 * np.pi * 0.07 * self.t + 1.0)    # ±0.4 rad (~23 deg)

        return np.stack([roll, pitch, yaw], axis=1)  # (N, 3)

    def compute_angular_velocity(self, angles):
        """
        Compute true angular velocity (rad/s) from angle derivatives.
        Uses finite differences on ground truth angles.
        """
        omega = np.gradient(angles, self.dt, axis=0)  # (N, 3)
        return omega

    def generate_gyroscope(self, omega_true):
        """
        Simulate gyroscope readings:
          measured = true + bias_drift + white_noise
        """
        N = len(omega_true)

        # Slowly varying bias (random walk)
        bias = np.cumsum(
            np.random.randn(N, 3) * self.gyro_bias_std * np.sqrt(self.dt),
            axis=0
        )
        # Clip bias to realistic range
        bias = np.clip(bias, -0.05, 0.05)

        # White noise
        noise = np.random.randn(N, 3) * self.gyro_noise_std

        gyro_measured = omega_true + bias + noise

        return gyro_measured, bias  # return bias for analysis

    def generate_accelerometer(self, angles):
        """
        Simulate accelerometer readings:
          In static/slow motion, accelerometer measures gravity projection.
          measured = gravity_projection(roll, pitch) + white_noise
        """
        N = len(angles)
        g = 9.81  # m/s^2

        accel_measured = np.zeros((N, 3))

        for i in range(N):
            roll  = angles[i, 0]
            pitch = angles[i, 1]

            # Gravity projection onto body frame axes
            ax = -g * np.sin(pitch)
            ay =  g * np.cos(pitch) * np.sin(roll)
            az =  g * np.cos(pitch) * np.cos(roll)

            accel_measured[i] = [ax, ay, az]

        # Add white noise
        accel_measured += np.random.randn(N, 3) * self.accel_noise_std

        return accel_measured

    def simulate(self):
        """
        Run full IMU simulation.

        Returns dict with:
            t            : time array (N,)
            angles_true  : ground truth roll/pitch/yaw (N, 3)
            omega_true   : true angular velocity (N, 3)
            gyro         : noisy gyroscope readings (N, 3)
            accel        : noisy accelerometer readings (N, 3)
            gyro_bias    : true gyro bias (N, 3)
        """
        angles_true = self.generate_true_trajectory()
        omega_true  = self.compute_angular_velocity(angles_true)
        gyro, bias  = self.generate_gyroscope(omega_true)
        accel       = self.generate_accelerometer(angles_true)

        print(f"IMU Simulation Complete")
        print(f"  Duration   : {self.duration}s")
        print(f"  Timesteps  : {self.timesteps} @ {1/self.dt:.0f} Hz")
        print(f"  Gyro noise : {self.gyro_noise_std} rad/s")
        print(f"  Accel noise: {self.accel_noise_std} m/s^2")
        print(f"  Bias drift : {self.gyro_bias_std} rad/s (std)")

        return {
            "t"           : self.t,
            "angles_true" : angles_true,
            "omega_true"  : omega_true,
            "gyro"        : gyro,
            "accel"       : accel,
            "gyro_bias"   : bias
        }


def raw_gyro_integration(gyro, dt):
    """
    Naive integration of gyroscope — no correction.
    This drifts over time. Used as baseline comparison vs EKF.
    """
    N = len(gyro)
    angles = np.zeros((N, 3))
    for i in range(1, N):
        angles[i] = angles[i-1] + gyro[i] * dt
    return angles


if __name__ == "__main__":
    sim = IMUSimulator(dt=0.01, duration=30.0)
    data = sim.simulate()

    print(f"\nSample ground truth (t=1s): {data['angles_true'][100]}")
    print(f"Sample gyro reading (t=1s): {data['gyro'][100]}")
    print(f"Sample accel reading (t=1s): {data['accel'][100]}")