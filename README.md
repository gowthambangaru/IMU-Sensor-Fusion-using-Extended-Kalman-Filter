Problem Statement
Any moving vehicle — satellite, rocket, or aircraft — must know its orientation at every moment. Two sensors are available:
SensorStrengthWeaknessGyroscopeAccurate short-termDrifts over time due to biasAccelerometerNo driftNoisy, sensitive to vibrations
Neither sensor alone is reliable. The EKF mathematically fuses both to produce an estimate that is accurate, drift-corrected, and robust to noise.

System Architecture
┌─────────────────────────────────────────────────────┐
│                  IMU Simulator                       │
│  True Trajectory → Gyroscope + Accelerometer + Noise│
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   EKF Pipeline       │
          │                      │
          │  ┌─────────────────┐ │
          │  │  PREDICT STEP   │ │  ← Gyroscope (propagate state)
          │  │  x = f(x, gyro) │ │
          │  │  P = FPFᵀ + Q   │ │
          │  └────────┬────────┘ │
          │           │          │
          │  ┌────────▼────────┐ │
          │  │  UPDATE STEP    │ │  ← Accelerometer (correct state)
          │  │  K = PH ͵(HPHᵀ+R)│ │
          │  │  x = x + K·y   │ │
          │  │  P = (I-KH)P   │ │
          │  └────────┬────────┘ │
          └───────────┼──────────┘
                      │
          ┌───────────▼──────────┐
          │     Estimated         │
          │  Roll, Pitch, Yaw     │
          │  + Gyro Bias          │
          └───────────────────────┘

EKF State Vector
x = [roll, pitch, yaw, bias_x, bias_y, bias_z]   (6 states)

roll, pitch, yaw — attitude angles in radians
bias_x, bias_y, bias_z — gyroscope bias estimates (rad/s)


Mathematics
Predict Step (Gyroscope)
x^k∣k−1=x^k−1+(ωmeasured−b^)⋅Δt\hat{x}_{k|k-1} = \hat{x}_{k-1} + (\omega_{measured} - \hat{b}) \cdot \Delta tx^k∣k−1​=x^k−1​+(ωmeasured​−b^)⋅Δt
Pk∣k−1=FkPk−1FkT+QP_{k|k-1} = F_k P_{k-1} F_k^T + QPk∣k−1​=Fk​Pk−1​FkT​+Q
where FkF_k
Fk​ is the Jacobian of the state transition function.

Update Step (Accelerometer)
Kk=Pk∣k−1HkT(HkPk∣k−1HkT+R)−1K_k = P_{k|k-1} H_k^T (H_k P_{k|k-1} H_k^T + R)^{-1}Kk​=Pk∣k−1​HkT​(Hk​Pk∣k−1​HkT​+R)−1
x^k=x^k∣k−1+Kk(zk−h(x^k∣k−1))\hat{x}_k = \hat{x}_{k|k-1} + K_k (z_k - h(\hat{x}_{k|k-1}))x^k​=x^k∣k−1​+Kk​(zk​−h(x^k∣k−1​))
Pk=(I−KkHk)Pk∣k−1P_k = (I - K_k H_k) P_{k|k-1}Pk​=(I−Kk​Hk​)Pk∣k−1​
Measurement Model
Gravity projection onto body frame:
h(x)=[−gsin⁡(θ)gcos⁡(θ)sin⁡(ϕ)gcos⁡(θ)cos⁡(ϕ)]h(x) = \begin{bmatrix} -g \sin(\theta) \\ g \cos(\theta)\sin(\phi) \\ g \cos(\theta)\cos(\phi) \end{bmatrix}h(x)=​−gsin(θ)gcos(θ)sin(ϕ)gcos(θ)cos(ϕ)​​
where ϕ\phi
ϕ = roll, θ\theta
θ = pitch.


Results
RMSE Comparison
AxisRaw Gyro RMSEEKF RMSEReductionRoll0.1128 rad0.0020 rad98.2%Pitch0.2257 rad0.0020 rad99.1%Yaw0.2146 rad0.2139 rad0.3%Total0.1913 rad0.1235 rad35.4%

Note on Yaw: Yaw is unobservable from accelerometer alone since gravity has no horizontal component — a fundamental physical constraint of accelerometer-gyroscope fusion. In practice, this is resolved using a magnetometer or GPS aiding, exactly as implemented in ISRO's Inertial Navigation Systems.


Plot 1 — Angle Estimation: EKF vs Raw Gyro vs Ground Truth

EKF closely tracks ground truth on Roll and Pitch while raw gyro integration visibly drifts.

Plot 2 — Error Over Time
Show Image
Raw gyro error grows continuously due to bias drift. EKF error remains near zero throughout.

Plot 3 — RMSE Comparison
Show Image
98%+ RMSE reduction on Roll and Pitch axes confirms EKF effectiveness.

Plot 4 — Gyroscope Bias Estimation
Show Image
EKF simultaneously estimates and tracks the hidden gyroscope bias — a key feature for real inertial navigation systems.

Noise Model Parameters
ParameterValueDescriptionGyro noise std0.01 rad/sWhite noise on angular velocityGyro bias drift0.003 rad/sRandom walk biasAccel noise std0.05 m/s²White noise on accelerationSample rate100 HzTimestep: 0.01sDuration30s3000 timesteps
