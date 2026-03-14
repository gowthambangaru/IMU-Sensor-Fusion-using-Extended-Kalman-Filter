"""
Extended Kalman Filter (EKF) — Phase 2
Fuses accelerometer and gyroscope data for
real-time 6-DOF attitude estimation.

State vector: x = [roll, pitch, yaw, bias_x, bias_y, bias_z]
"""

import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, dt=0.01):
        """
        Args:
            dt : timestep in seconds (must match IMU simulator)
        """
        self.dt = dt
        self.n = 6  # state size: [roll, pitch, yaw, bx, by, bz]

        # --- Initial State ---
        self.x = np.zeros(self.n)  # start at zero angles, zero bias

        # --- Initial Covariance P ---
        # High uncertainty at start
        self.P = np.eye(self.n) * 0.1

        # --- Process Noise Q ---
        # How much we trust the gyroscope prediction
        # Small = trust gyro more, Large = trust gyro less
        self.Q = np.diag([
            1e-4, 1e-4, 1e-4,   # angle process noise
            1e-6, 1e-6, 1e-6    # bias process noise (bias changes slowly)
        ])

        # --- Measurement Noise R ---
        # How much we trust the accelerometer
        # Small = trust accel more, Large = trust accel less
        self.R = np.diag([
            0.1, 0.1, 0.1       # accelerometer noise (m/s^2)^2
        ])

        # Gravity constant
        self.g = 9.81

    # ----------------------------------------------------------------
    # PREDICT STEP — uses gyroscope
    # ----------------------------------------------------------------
    def predict(self, gyro):
        """
        Propagate state forward using gyroscope measurement.

        Args:
            gyro : (3,) array — measured angular velocity [wx, wy, wz] rad/s
        """
        roll, pitch, yaw = self.x[0], self.x[1], self.x[2]
        bx,   by,   bz   = self.x[3], self.x[4], self.x[5]

        # Correct gyro by subtracting estimated bias
        wx = gyro[0] - bx
        wy = gyro[1] - by
        wz = gyro[2] - bz

        # --- State Transition ---
        # Euler angle update from corrected angular velocity
        roll_new  = roll  + wx * self.dt
        pitch_new = pitch + wy * self.dt
        yaw_new   = yaw   + wz * self.dt

        # Bias assumed constant (random walk — unchanged in predict)
        bx_new, by_new, bz_new = bx, by, bz

        # Update state
        self.x = np.array([
            roll_new, pitch_new, yaw_new,
            bx_new,   by_new,   bz_new
        ])

        # --- Jacobian F (linearized state transition) ---
        # Partial derivatives of f(x) w.r.t. x
        F = np.eye(self.n)

        # d(roll)/d(bx) = -dt, d(pitch)/d(by) = -dt, d(yaw)/d(bz) = -dt
        F[0, 3] = -self.dt
        F[1, 4] = -self.dt
        F[2, 5] = -self.dt

        # --- Propagate Covariance ---
        self.P = F @ self.P @ F.T + self.Q

    # ----------------------------------------------------------------
    # UPDATE STEP — uses accelerometer
    # ----------------------------------------------------------------
    def update(self, accel):
        """
        Correct state using accelerometer measurement.

        Args:
            accel : (3,) array — measured acceleration [ax, ay, az] m/s^2
        """
        roll  = self.x[0]
        pitch = self.x[1]

        # --- Expected accelerometer reading from current state ---
        # Gravity projection onto body frame
        ax_pred = -self.g * np.sin(pitch)
        ay_pred =  self.g * np.cos(pitch) * np.sin(roll)
        az_pred =  self.g * np.cos(pitch) * np.cos(roll)

        h = np.array([ax_pred, ay_pred, az_pred])

        # --- Innovation (measurement residual) ---
        y = accel - h

        # --- Jacobian H (linearized measurement model) ---
        # Partial derivatives of h(x) w.r.t. x
        H = np.zeros((3, self.n))

        # d(ax)/d(pitch)
        H[0, 1] = -self.g * np.cos(pitch)

        # d(ay)/d(roll), d(ay)/d(pitch)
        H[1, 0] =  self.g * np.cos(pitch) * np.cos(roll)
        H[1, 1] = -self.g * np.sin(pitch) * np.sin(roll)

        # d(az)/d(roll), d(az)/d(pitch)
        H[2, 0] = -self.g * np.cos(pitch) * np.sin(roll)
        H[2, 1] = -self.g * np.sin(pitch) * np.cos(roll)

        # --- Kalman Gain ---
        S = H @ self.P @ H.T + self.R          # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)    # Kalman Gain (6x3)

        # --- State Update ---
        self.x = self.x + K @ y

        # --- Covariance Update (Joseph form for numerical stability) ---
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P

    # ----------------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------------
    def run(self, gyro_data, accel_data):
        """
        Run EKF over full dataset.

        Args:
            gyro_data  : (N, 3) gyroscope measurements
            accel_data : (N, 3) accelerometer measurements

        Returns:
            estimates  : (N, 6) state estimates [roll, pitch, yaw, bx, by, bz]
        """
        N = len(gyro_data)
        estimates = np.zeros((N, self.n))

        for i in range(N):
            self.predict(gyro_data[i])
            self.update(accel_data[i])
            estimates[i] = self.x.copy()

            if i % 500 == 0:
                print(f"  Step {i:4d}/{N} | "
                      f"roll={self.x[0]:.3f} "
                      f"pitch={self.x[1]:.3f} "
                      f"yaw={self.x[2]:.3f}")

        return estimates


def compute_rmse(true_angles, estimated_angles):
    """
    Compute RMSE for roll, pitch, yaw separately and combined.

    Args:
        true_angles      : (N, 3) ground truth
        estimated_angles : (N, 3) estimated

    Returns:
        dict with per-axis and overall RMSE
    """
    errors = true_angles - estimated_angles
    rmse_roll  = np.sqrt(np.mean(errors[:, 0] ** 2))
    rmse_pitch = np.sqrt(np.mean(errors[:, 1] ** 2))
    rmse_yaw   = np.sqrt(np.mean(errors[:, 2] ** 2))
    rmse_total = np.sqrt(np.mean(errors ** 2))

    return {
        "roll" : rmse_roll,
        "pitch": rmse_pitch,
        "yaw"  : rmse_yaw,
        "total": rmse_total
    }


if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from imu_simulator import IMUSimulator, raw_gyro_integration

    print("=" * 50)
    print("Running IMU Simulation...")
    print("=" * 50)
    sim  = IMUSimulator(dt=0.01, duration=30.0)
    data = sim.simulate()

    print("\n" + "=" * 50)
    print("Running Extended Kalman Filter...")
    print("=" * 50)
    ekf       = ExtendedKalmanFilter(dt=0.01)
    estimates = ekf.run(data["gyro"], data["accel"])

    # Raw gyro integration (baseline)
    raw = raw_gyro_integration(data["gyro"], dt=0.01)

    # RMSE comparison
    true_angles = data["angles_true"]
    ekf_rmse    = compute_rmse(true_angles, estimates[:, :3])
    raw_rmse    = compute_rmse(true_angles, raw)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"\nRaw Gyro Integration RMSE:")
    print(f"  Roll  : {raw_rmse['roll']:.4f} rad")
    print(f"  Pitch : {raw_rmse['pitch']:.4f} rad")
    print(f"  Yaw   : {raw_rmse['yaw']:.4f} rad")
    print(f"  Total : {raw_rmse['total']:.4f} rad")

    print(f"\nEKF Estimated RMSE:")
    print(f"  Roll  : {ekf_rmse['roll']:.4f} rad")
    print(f"  Pitch : {ekf_rmse['pitch']:.4f} rad")
    print(f"  Yaw   : {ekf_rmse['yaw']:.4f} rad")
    print(f"  Total : {ekf_rmse['total']:.4f} rad")

    reduction = (1 - ekf_rmse["total"] / raw_rmse["total"]) * 100
    print(f"\n>>> RMSE Reduction: {reduction:.1f}%")
    print(f">>> This is your resume metric!")