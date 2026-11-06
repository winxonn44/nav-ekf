# Conventions

Most estimation bugs are convention bugs. Everything in this repo, Python and
C++, follows this page. If code and this page disagree, the code is wrong
(or the page gets a reviewed change). Each rule names the test that enforces it.

## 1. Units and time

- SI units everywhere: m, s, rad, m/s², rad/s. Never use degrees inside the
  filter. Convert at the I/O boundary only.
- Timestamps are `int64` **nanoseconds** (EuRoC uses these natively). Convert
  to seconds only when forming a step: `dt = (t_k − t_{k−1}) · 1e-9`, in `double`.
- All filter math is in `double` (`float64`).

## 2. Frames

| Name | Axes | Used by |
|---|---|---|
| `W` (world) | **ENU**: x east, y north, z up | our simulator, KITTI, this filter |
| `B` (body / IMU) | **FLU**: x forward, y left, z up | our simulator |
| EuRoC `W` | its own Vicon/Leica world frame, z up | EuRoC ground truth (aligned by evo, not assumed ENU) |
| TartanGround / VoroNav | **NED** world, FRD body | VoroNav's poses (convert at the boundary, §2.2) |

The filter's state uses one world frame and one body frame. The IMU frame
**is** the body frame: an extrinsic `T_B_I`, if a dataset needs one, is applied
in the loader before the filter sees the samples.

### 2.1 Naming transforms

- `R_A_B` is the rotation that takes a vector **expressed in B** and expresses it
  **in A**: `v_A = R_A_B · v_B`.
- `T_A_B` (a 4×4 or (R, t) pair) is the pose of B in A: `p_A = R_A_B · p_B + t_A_B`.
- Compose by matching inner frames: `T_A_C = T_A_B · T_B_C`. If the inner
  letters don't match, the code is wrong.
- In code, write the frames in the name: `R_W_B`, `q_W_B`, `p_W`, `v_W`, `f_B`.
  The state's `q` is `q_W_B` and `R = R(q_W_B) = R_W_B`.

### 2.2 ENU ↔ NED

`R_ENU_NED = [[0, 1, 0], [1, 0, 0], [0, 0, −1]]` (it is its own inverse). The
body change FRD ↔ FLU is `diag(1, −1, −1)`. A NED pose converts as
`R_ENU_B(FLU) = R_ENU_NED · R_NED_B(FRD) · diag(1, −1, −1)`.
*Test: `test_ned_enu_roundtrip`, plus a pose "facing north, level" in both frames.*

## 3. Quaternions and rotations

- **Hamilton** convention: `i² = j² = k² = ijk = −1`. Composition `q_A_C = q_A_B ⊗ q_B_C`.
- Storage order in this repo, Python and C++ alike: **`[w, x, y, z]`**.
  **Eigen trap:** `Eigen::Quaterniond(w, x, y, z)` takes this order in its
  constructor but **stores** `(x, y, z, w)`, so `.coeffs()` returns `[x, y, z, w]`.
  Never pass `.coeffs()` or `.data()` across the Python/C++ boundary; convert
  with an explicit helper. TUM output files use `qx qy qz qw`.
  *Test: `test_eigen_quaternion_order` (C++), `test_tum_quaternion_order`.*
- `R(q)` is the rotation matrix of `q`, so `v_A = R(q_A_B) v_B = q_A_B ⊗ [0, v_B] ⊗ q_A_B*`.
- Sign: `q` and `−q` are the same rotation. Keep `w ≥ 0` when **comparing** or
  **logging**, never inside the propagation (flipping mid-integration is harmless
  for rotation but confuses diff-based tests).
- Normalize `q` after every propagation step and every injection.

### 3.1 Skew, Exp and Log

- `[a]×` is the skew matrix with `[a]× b = a × b`:
  `[a]× = [[0, −a_z, a_y], [a_z, 0, −a_x], [−a_y, a_x, 0]]`.
- `Exp(φ)` maps a rotation vector to a unit quaternion:
  `Exp(φ) = [cos(‖φ‖/2), sin(‖φ‖/2) · φ/‖φ‖]`. For `‖φ‖ < 1e-8` use the Taylor
  form `[1 − ‖φ‖²/8, φ/2 · (1 − ‖φ‖²/24)]`, then normalize.
- `Log(q)` is the inverse, returning `‖φ‖ ≤ π` (flip `q → −q` first if `w < 0`).
  With `q = [w, v]`: `φ = 2·atan2(‖v‖, w)·v/‖v‖`. For `‖v‖ < 1e-8`, use the Taylor form
  `φ = (2/w)(1 − ‖v‖²/(3w²))·v`. The C++ port uses the same branches and thresholds.
  *Test: `Log(Exp(φ)) = φ` for random `φ` with `‖φ‖ < π`, including `‖φ‖ = 1e-12`.*
- Matrix form: `Exp(φ)` as a rotation is `I + sin θ [u]× + (1 − cos θ)[u]×²` (Rodrigues).

### 3.2 Which side the perturbations live on

**Errors are local (right-multiplied, body frame):**
`q_true = q̂ ⊗ Exp(δθ)`, i.e. `R_true = R̂ · Exp(δθ)`. The angular rate integrates
on the right too: `q ← q ⊗ Exp(ω_B · dt)`. Every Jacobian in
[`derivation.md`](derivation.md) assumes this. (Solà §5 calls this the
"local angular error"; mixing it with a global `Exp(δθ) ⊗ q̂` error flips signs
in `F` and `H`.)

## 4. Gravity and the IMU model

- `g_W = [0, 0, −9.81] m/s²` in ENU (configurable magnitude; one value per run).
- The **accelerometer measures specific force** in the body frame, not acceleration:
  `a_m = R_W_Bᵀ (a_W − g_W) + b_a + n_a`.
  A still, level IMU reads `a_m = [0, 0, +9.81]`.
  *Test: `test_static_imu_reads_plus_g`.*
- The **gyro measures body angular rate**: `ω_m = ω_B + b_g + n_g`, where
  `q̇_W_B = ½ q_W_B ⊗ [0, ω_B]`.
- Noise model (continuous-time densities, per axis, all isotropic at first):

| Symbol | Meaning | Units |
|---|---|---|
| `σ_a` | accelerometer white noise | m/s²/√Hz (= m/s^{3/2}) |
| `σ_g` | gyro white noise (angle random walk) | rad/s/√Hz (= rad/√s) |
| `σ_ba` | accelerometer bias random walk | m/s³/√Hz |
| `σ_bg` | gyro bias random walk | rad/s²/√Hz |

  Sampled at period `dt`, white noise has per-sample std `σ/√dt`, and a bias steps by
  `σ_b · √dt · N(0, 1)` per sample. Datasheets often quote `°/√h` or
  `µg/√Hz`; convert in the config loader and write the conversion next to the
  number.

## 5. State layout

**Nominal state** `x̂` (16 numbers): `p_W (3), v_W (3), q_W_B (4, [w,x,y,z]), b_a (3), b_g (3)`.

**Error state** `δx` (15 numbers), fixed index order everywhere (`P` is 15×15):

| Block | Indices | Definition |
|---|---|---|
| `δp` | 0–2 | `p = p̂ + δp` |
| `δv` | 3–5 | `v = v̂ + δv` |
| `δθ` | 6–8 | `q = q̂ ⊗ Exp(δθ)` (body frame) |
| `δb_a` | 9–11 | `b_a = b̂_a + δb_a` |
| `δb_g` | 12–14 | `b_g = b̂_g + δb_g` |

Process noise vector `n` (12): `[n_a, n_g, n_ba, n_bg]`, so `Q_c = diag(σ_a² I, σ_g² I, σ_ba² I, σ_bg² I)`.
Position fix `z` (3): antenna position in `W`; `R_meas = σ_gps² I` by default.

## 6. Discrete propagation (the one integrator)

For IMU sample `k` covering `[t_k, t_{k+1})`, with `â = a_m,k − b̂_a`, `ω̂ = ω_m,k − b̂_g`,
`R = R(q̂_k)` (values at the **start** of the step, zero-order hold):

```
a_W = R · â + g_W
p ← p + v·dt + ½·a_W·dt²
v ← v + a_W·dt
q ← normalize(q ⊗ Exp(ω̂·dt))
b_a, b_g unchanged
```

Python and C++ implement exactly this, in this order of operations, so they
can agree to 1e-9. Any higher-order integrator is a later, separate option
with its own name.

## 7. Simulator contract (what "truth" means)

The integrator in §6 has `O(dt)` discretization error against a continuous
trajectory, so "tracks truth to 1e-9" needs a precise definition. The simulator
provides two IMU streams from the same smooth trajectory:

1. **Consistent stream** (exactness tests). IMU samples are generated by
   *inverting* §6 between the true states at `t_k` and `t_{k+1}`:
   `ω_m,k = Log(q_k* ⊗ q_{k+1}) / dt`,
   `a_W,k = (v_{k+1} − v_k)/dt`, `a_m,k = R_kᵀ (a_W,k − g_W)`; the reference
   position is then defined by the §6 recursion
   (`p_{k+1} = p_k + (v_k + v_{k+1})/2 · dt`). Noise-free dead reckoning
   must reproduce `q, v, p` to ≈1e-9. This tests the code, not the math.
2. **Analytic stream** (convergence tests). `a_m, ω_m` are the exact
   instantaneous values at `t_k` from the analytic derivatives. Dead-reckoning
   error must shrink as `O(dt)` when the rate doubles (fit the slope). This
   tests that the integrator converges to the real trajectory.

Ground-truth output always reports the analytic trajectory, and every result
says which stream it used.

## 8. Files and logs

- Trajectories: TUM format, `t x y z qx qy qz qw`, `t` in seconds (`%.9f`).
- Per-step filter log: one row per IMU step or update with `t_ns`, state, the
  diagonal of `P`, and for updates `NIS`, `accepted`.
- Random numbers: every run takes an explicit `seed`; Monte Carlo run `i` uses
  `seed + i`. Never use the global NumPy RNG (`np.random.default_rng(seed)` only).
