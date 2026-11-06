# Derivation: error-state EKF for IMU + position fixes

Notation and frames follow [`conventions.md`](conventions.md): ENU world `W`,
body `B`, Hamilton quaternions `[w, x, y, z]`, **local** errors
`R_true = R̂ Exp(δθ)`, error order `[δp, δv, δθ, δb_a, δb_g]`. Every result here
is checked numerically by `scripts/check_derivation.py` (finite differences,
Van Loan, χ² bounds); the tests in E1/E2 re-check them against the code.

**Model assumptions.** Flat, non-rotating Earth (no Coriolis, transport rate or
Schuler dynamics); constant gravity; IMU frame = body frame. That's fine for
minutes-long runs with consumer or tactical IMUs, and it's what Solà §5 assumes.
Earth rate (7.3e-5 rad/s) is below EuRoC's gyro bias, so ignoring it is honest
there; revisit for KITTI if drift looks systematic.

Reference: J. Solà, *Quaternion kinematics for the error-state Kalman filter*
(2017), §5–7. Equation shapes match his local-error formulation.

---

## 1. Three states

| | Position | Velocity | Orientation | Biases |
|---|---|---|---|---|
| true | `p` | `v` | `R` | `b_a, b_g` |
| nominal (no noise, estimated biases) | `p̂` | `v̂` | `R̂` | `b̂_a, b̂_g` |
| error | `δp = p − p̂` | `δv = v − v̂` | `R = R̂ Exp(δθ)` | `δb = b − b̂` |

Continuous true kinematics, with IMU readings `a_m, ω_m` and white noises `n_a, n_g`:
```
ṗ = v
v̇ = R (a_m − b_a − n_a) + g
Ṙ = R [ω_m − b_g − n_g]×
ḃ_a = n_ba,   ḃ_g = n_bg
```
The nominal state obeys the same equations with all noises zero and biases `b̂`.
Write `â = a_m − b̂_a` and `ω̂ = ω_m − b̂_g` (bias-corrected readings).

First-order tools used below:
- `Exp(δθ) ≈ I + [δθ]×`
- `[a]× b = −[b]× a`
- `[a]×[b]× − [b]×[a]× = [a × b]×`

---

## 2. Error dynamics

### 2.1 Position
`δṗ = ṗ − p̂̇ = v − v̂ = δv`. Exactly linear.

### 2.2 Velocity (the one to know on a whiteboard)
```
δv̇ = v̇ − v̂̇
    = R̂ (I + [δθ]×)(â − δb_a − n_a) + g − (R̂ â + g)
    = R̂ [δθ]× â − R̂ δb_a − R̂ n_a          + (second-order terms: [δθ]×δb_a, [δθ]×n_a)
    = −R̂ [â]× δθ − R̂ δb_a − R̂ n_a          (using [δθ]× â = −[â]× δθ)
```
Intuition: a small tilt `δθ` rotates the measured specific force (mostly the
+9.81 "up" reaction), leaking gravity into the horizontal axes. The leak is
`g·δθ`, which is why gyro errors dominate long-term position drift (§7).

### 2.3 Orientation
Differentiate `R = R̂ Exp(δθ)` and match it with the true kinematics:
```
Ṙ  = R̂̇ Exp(δθ) + R̂ d/dt Exp(δθ)
   ≈ R̂ [ω̂]× (I + [δθ]×) + R̂ [δθ̇]×
Ṙ  = R [ω]× = R̂ (I + [δθ]×) [ω]×,      with ω = ω̂ − δb_g − n_g
```
Set them equal, cancel `R̂`, keep first-order terms:
```
[δθ̇]× = [ω]× − [ω̂]× + [δθ]×[ω̂]× − [ω̂]×[δθ]×
      = [ω − ω̂]× + [δθ × ω̂]×
⇒ δθ̇ = −[ω̂]× δθ − δb_g − n_g
```
The `−[ω̂]× δθ` term exists because `δθ` is expressed in the rotating body frame.
A global error would not have it, and its `F` would differ.

### 2.4 Biases
`δḃ_a = n_ba`, `δḃ_g = n_bg` (random walks).

---

## 3. Continuous-time linear model

`δẋ = F δx + G n`, with `n = [n_a, n_g, n_ba, n_bg]` (12) and
`E[n(t) n(τ)ᵀ] = Q_c δ(t − τ)`, `Q_c = diag(σ_a² I₃, σ_g² I₃, σ_ba² I₃, σ_bg² I₃)`.

```
        δp    δv      δθ        δb_a    δb_g
F = [   0     I       0          0       0   ]  δp
    [   0     0    −R̂[â]×      −R̂       0   ]  δv
    [   0     0    −[ω̂]×        0      −I   ]  δθ
    [   0     0       0          0       0   ]  δb_a
    [   0     0       0          0       0   ]  δb_g

        n_a   n_g   n_ba  n_bg
G = [   0     0     0     0  ]  δp
    [  −R̂    0     0     0  ]  δv
    [   0    −I     0     0  ]  δθ
    [   0     0     I     0  ]  δb_a
    [   0     0     0     I  ]  δb_g
```
Since `R̂` is orthogonal and the noise is isotropic, `R̂ σ_a² I R̂ᵀ = σ_a² I`.
The velocity block of `G Q_c Gᵀ` is just `σ_a² I`, which is a useful sanity check.

---

## 4. Discretization over one IMU step

`F` and `G` are evaluated at the **start** of the step (`R̂_k`, `â_k`, `ω̂_k`),
matching the zero-order-hold integrator in conventions §6.

Transition matrix `Φ = exp(F dt)`:
- first order: `Φ₁ = I + F dt`
- second order: `Φ₂ = I + F dt + ½ F² dt²`
- exact (reference): `scipy.linalg.expm(F dt)`

Process noise `Q_d = ∫₀^dt Φ(τ) G Q_c Gᵀ Φ(τ)ᵀ dτ`:
- first order: `Q_d ≈ G Q_c Gᵀ dt`
- **Van Loan (exact reference):** form the 30×30 matrix
  ```
  A = [ −F    G Q_c Gᵀ ] · dt,      B = expm(A) = [ B11  B12 ]
      [  0       Fᵀ    ]                          [  0   B22 ]
  ```
  then `Φ = B22ᵀ` and `Q_d = Φ · B12`.

Covariance propagation: `P ← Φ P Φᵀ + Q_d`, then symmetrize
`P ← ½(P + Pᵀ)`.

Expected size of the approximation error (checked in the script; E2 makes it a test):
`‖Φ₁ − expm‖ = O(dt²)` and `‖Φ₂ − expm‖ = O(dt³)`. At 200 Hz both are far below
the noise, so first order is the default, and the test proves it rather than assuming it.

---

## 5. Position update (with lever arm)

The GPS antenna sits at `l` (body frame) from the IMU:
`z = p + R l + n_z`, `n_z ~ N(0, R_meas)`. Predicted `ẑ = p̂ + R̂ l`.
```
z − ẑ ≈ δp + R̂ (I + [δθ]×) l − R̂ l = δp + R̂ [δθ]× l = δp − R̂ [l]× δθ
⇒ H = [ I₃   0   −R̂[l]×   0   0 ]      (3 × 15)
```
With `l = 0` (our simulator at first), `H = [I 0 0 0 0]`.

Update (innovation `y = z − ẑ`):
```
S  = H P Hᵀ + R_meas
K  = P Hᵀ S⁻¹                       (solve, don't invert: K = solve(S, H P)ᵀ)
δx̂ = K y
P  ← (I − K H) P (I − K H)ᵀ + K R_meas Kᵀ     (Joseph form)
```
**Gate:** `NIS = yᵀ S⁻¹ y ~ χ²(3)` if the filter is consistent. Reject the fix (log
it, skip the update) when `NIS > 11.345` (the 99% quantile).

---

## 6. Injection and reset

Injection folds the estimated error into the nominal state:
```
p̂ ← p̂ + δp̂,   v̂ ← v̂ + δv̂,   q̂ ← normalize(q̂ ⊗ Exp(δθ̂)),   b̂ ← b̂ + δb̂
```
Then `δx̂ ← 0`. Covariance must follow the **new** error definition. With
`q̂⁺ = q̂ ⊗ Exp(δθ̂)`, the new orientation error satisfies
`Exp(δθ⁺) = Exp(−δθ̂) Exp(δθ)`. Write `δθ = δθ̂ + ε` and use
`Exp(φ + ε) ≈ Exp(φ) Exp(J_r(φ) ε)`:
```
δθ⁺ ≈ J_r(δθ̂) ε,      J_r(φ) = I − (1 − cos‖φ‖)/‖φ‖² [φ]× + (‖φ‖ − sin‖φ‖)/‖φ‖³ [φ]×²
```
So the reset Jacobian is `G_r = blockdiag(I, I, J_r(δθ̂), I, I)`, `P ← G_r P G_rᵀ`.
To first order `J_r(δθ̂) ≈ I − ½[δθ̂]×` (Solà's form). Since `δθ̂` is typically
< 1e-2 rad, the correction is ~0.5% on cross-terms. E2 measures whether dropping it
(`G_r = I`) changes NEES, rather than assuming either way.

---

## 7. Dead-reckoning drift laws (what E1 must reproduce)

Pure IMU integration (no updates), level and nearly still, one horizontal axis.
`W(t)` is a unit Wiener process, and `g = 9.81`.

| Error source | Velocity error | Position error | log–log slope (position) |
|---|---|---|---|
| accel white noise `σ_a` | `σ_a √t` | `σ_a t^{3/2} / √3` | 1.5 |
| accel bias `b_a` | `b_a t` | `½ b_a t²` | 2 |
| gyro white noise `σ_g` (tilt) | `g σ_g t^{3/2} / √3` | `g σ_g t^{5/2} / √20` | 2.5 |
| gyro bias `b_g` (tilt) | `½ g b_g t²` | `⅙ g b_g t³` | 3 |
| accel bias random walk `σ_ba` | `σ_ba t^{3/2} / √3` | `σ_ba t^{5/2} / √20` | 2.5 |
| gyro bias random walk `σ_bg` (tilt) | `g σ_bg t^{5/2} / √20` | `g σ_bg t^{7/2} / √252` | 3.5 |

(General rule: the n-fold integral of `W` has variance `t^{2n+1} / ((2n+1)(n!)²)`.)

Derivations, all from §2:
- **Accel white noise:** `δv = σ_a W(t)`, so `δp = σ_a ∫₀ᵗ W`. `Var ∫₀ᵗ W = t³/3`.
- **Tilt from gyro noise:** `δθ = σ_g W(t)`, and from §2.2 the horizontal
  `δv̇ = g δθ`. So `δv = g σ_g ∫W`, with `Var = g² σ_g² t³/3`, and
  `δp = g σ_g ∫∫W`, with `Var ∫₀ᵗ∫₀ˢ W = t⁵/20`.
- **Biases:** constant inputs integrated once, twice or three times.

E1's "drift grows as theory predicts" means: over ≥ 50 Monte Carlo runs with
one source switched on at a time, the empirical RMS error matches the table's
curve within sampling error (E1 uses 100 runs; ±20% at 50 runs holds only at a single time point) and the fitted slope
matches the last column.

---

## 8. Consistency statistics (E2)

- `NEES_k = (x_k − x̂_k)ᵀ P_k⁻¹ (x_k − x̂_k)`, with the 15-dim error (orientation via
  `Log(q̂* ⊗ q)`). If consistent, `NEES ~ χ²(15)`. Over `N` runs the mean at each
  `k` times `N` is `~ χ²(15N)`.
- `NIS ~ χ²(3)`. Its N-run mean times `N` is `~ χ²(3N)`.

95% two-sided bands on the **mean**, from `scripts/check_derivation.py`:

| N runs | NEES (15-dim) band | NIS (3-dim) band |
|---|---|---|
| 1 | [6.26, 27.49] | [0.22, 9.35] |
| 50 | **[13.52, 16.56]** | **[2.36, 3.72]** |

The NIS gate is 11.345 (the 99% quantile of χ²(3)). At 100 Hz the first-order `Q_d` is
0.5% off Van Loan and `Φ₁` is 4e-4 off `expm`. Both errors scale as the
script predicts (×4 per halving for `Φ₁`, ×8 for `Φ₂`).

Above the band means the filter is overconfident (Q or R too small, or a wrong
Jacobian). Below the band means it is too pessimistic.
