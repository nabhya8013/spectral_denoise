# ResUNet1D Model Review — Sample 778 Analysis & Tuning Guide

## Revision History

| Round | Date | Peak Shift | RMSE FP | Noise Floor | PSNR | Overall Quality |
|-------|------|-----------|---------|-------------|------|-----------------|
| v1 (baseline) | Apr 17 R1 | 0.4062 | 0.1835 | 0.0343 | 49.51 | 94.25% |
| v2 | Apr 17 R2 | 0.1562 | 0.1823 | 0.0295 | 47.77 | 95.10% |
| v4 scientific polish | May 11 | 0.201 | 0.1180 | 0.0411 | 42.71 | 95.50% |
| **v5 metric polish** | **May 11** | **0.195** | **0.1114** | **0.0379** | **43.20** | **95.81%** |
| **v6 head-only polish** | **May 11** | **0.193** | **0.1117** | **0.0336** | **43.45** | **95.97%** |

---

## Round 6 — Head-Only Metric Polish

Round 6 changed the optimization strategy rather than the architecture: the encoder/backbone was frozen and only the output/refinement heads were fine-tuned (`TRAINABLE_SCOPE=head`, 0.106M / 8.122M trainable parameters). This preserved learned spectral features while polishing the official quality metric.

| Metric | v5 metric polish | **v6 head-only polish** | Delta | Verdict |
|--------|:---:|:---:|:---:|---|
| **Overall Quality** | 95.81% | **95.97%** | +0.16 pp | ✅ Promoted |
| **MSE** | 0.000282 | **0.000269** | −4.6% | ✅ |
| **PSNR** | 43.20 dB | **43.45 dB** | +0.25 dB | ✅ |
| **SSIM** | 0.99847 | **0.99856** | +0.00009 | ✅ |
| **Correlation** | 0.99709 | **0.99724** | +0.00015 | ✅ |
| **Boundary RMSE** | 0.684 | **0.668** | −2.3% | ✅ |
| **Midband RMSE** | 0.085 | **0.058** | −32% | ✅ |
| **High-wn RMSE** | 0.081 | **0.070** | −14% | ✅ |
| **Noise floor std** | 0.0379 | **0.0336** | −11% | ✅ |
| **Peak shift** | 0.195 | **0.193** | −1% | ✅ |
| **Fingerprint RMSE** | 0.1114 | **0.1117** | +0.0003 | ⚠️ effectively flat |

The head-only run peaked at **95.97%** and was promoted. A follow-up lower-LR head-only continuation plateaued and was rejected by the improvement gate, so the repo now keeps only the best promoted checkpoint.

Current clean repo state:

- Best checkpoint: `models/resunet1d_single_stage_final.pth`
- Best metrics: `results/resunet_single_stage_final.json`
- Candidate checkpoints/results from failed or superseded runs have been removed.

---

## Round 5 — Metric-Polish Fine-Tuning

### Strategy Shift

Round 4 used aggressive physics-aware losses and spectral attention to push fingerprint fidelity. Round 5 took the opposite approach — **gentle metric polish**:

| Dimension | Round 4 (scientific) | Round 5 (metric polish) | Rationale |
|---|---|---|---|
| **LR** | max 1e-5 / min 3e-6 | **max 6e-6 / min 2e-6** | 60% lower — avoid disrupting learned features |
| **Epochs** | 12 | **16** | Longer fine-tuning runway |
| **Augmentation** | off | **off** | Clean gradient signal |
| **Checkpoint selection** | review score | **overall_quality** | Optimize the primary metric directly |
| **Global reconstruction** | MSE 0.75 | **MSE 0.90** | Increased — let the model sharpen global fit |
| **Peak/physics aggressiveness** | high | **reduced** | Pulled back peak_center (1.35→1.1), peak_profile (0.58→0.42), peak_point_weight (2.7→2.2), fingerprint_point_weight (3.0→2.0) |
| **Smoothing** | smooth_consistency 0.01 | **0.018** | Increased — helps recover noise floor |
| **New losses** | extrema 0.08, sobolev 0.06, baseline_drift 0.02 | **extrema 0.04, sobolev 0.03, baseline_drift 0.015** | Halved — conservative presence without gradient competition |
| **Peak dominance** | 0.02 | **0.0** | Disabled — was conflicting with global MSE |

### Results — All Metrics Improved

| Metric | v4 (scientific) | **v5 (metric polish)** | Δ | Verdict |
|--------|:---:|:---:|:---:|---|
| **Overall Quality** | 95.50% | **95.81%** | +0.31 pp | ✅ Promoted |
| **MSE** | 0.000361 | **0.000282** | −22% | ✅ |
| **PSNR** | 42.71 dB | **43.20 dB** | +0.49 dB | ✅ |
| **SSIM** | 0.99803 | **0.99847** | +0.00044 | ✅ |
| **Correlation** | 0.99674 | **0.99709** | +0.00035 | ✅ |
| **Fingerprint RMSE** | 0.1180 | **0.1114** | −0.0066 | ✅ |
| **Boundary RMSE** | 0.767 | **0.684** | −11% | ✅ |
| **Midband RMSE** | 0.101 | **0.085** | −16% | ✅ Recovered |
| **High-wn RMSE** | 0.090 | **0.081** | −10% | ✅ |
| **Noise floor std** | 0.0411 | **0.0379** | −8% | ✅ Recovered |
| **Peak shift** | 0.201 | **0.195** | −3% | ✅ |
| **Selection score** | 92.64 | **92.98** | +0.34 | ✅ |
| **Review score** | 93.53 | **93.92** | +0.39 | ✅ |

> [!TIP]
> This is the first round where **every single metric improved simultaneously** — no tradeoffs. The key insight: after Round 4 added the right architectural capacity (spectral attention, detail head, positional bias), Round 5 let the model learn to *use* that capacity optimally by reducing gradient competition and giving global MSE more weight.

### Training Dynamics

- 16 epochs, all improving monotonically — zero oscillation
- Overall quality: 95.48% (ep1) → 95.81% (ep16), gaining ~0.02%/epoch
- Noise floor *decreased* every epoch (0.0413 → 0.0379) — the `smooth_consistency=0.018` bump worked
- Fingerprint RMSE tracked downward smoothly: 0.1183 → 0.1114
- **Still improving at epoch 16** — no plateau visible

### Loss Weight Budget (v5 vs v4)

```
v5 budget:
Global reconstruction:  MSE(0.90) + L1(0.16) + D1(0.42) + D2(0.32) = 1.80   (38%) ← UP from 30%
Smoothing forces:       TV(0.0015) + smooth(0.018) + baseline(0.035) = 0.055  (1%)
Edge terms:             edge_L1(0.06) + edge_D1(0.05) = 0.11                  (2%)
Peak terms:             amp(0.22) + profile(0.42) + center(1.1) + align(0.38) = 2.12 (45%)
Fingerprint terms:      fp_L1(0.22) + fp_MSE(0.18) + fp_D1(0.12) = 0.52      (11%) ← down from 17%
New physics:            extrema(0.04) + sobolev(0.03) + drift(0.015) = 0.085  (2%)  ← halved
Valley:                 valley(0.30) + valley_center(0.20) = 0.50             (11%)
Other:                  curvature(0.40) + fft(0.05) = 0.45                    (10%)
                                                          TOTAL ≈ 4.71
```

Key shift vs v4: global reconstruction share increased from ~30% to ~38%, fingerprint-specific dropped from 17% to 11%, physics losses halved. The model spent more gradient budget on overall fit quality, which lifted all boats.

### Why This Worked

1. **Architectural capacity was already in place** — SpectralAttention1D, detail head, positional/derivative bias from Round 4 gave the model the *ability* to reconstruct fine spectral features. Round 5 just needed to optimize the weights.

2. **Checkpoint selection by `overall_quality`** instead of `review_score` — this directly optimized the promotion metric, avoiding proxy misalignment.

3. **Higher MSE weight (0.75→0.90)** provided a stronger "gravitational pull" toward the clean target across all spectral regions, which secondarily improved fingerprint/boundary/midband without dedicated losses.

4. **Smooth consistency bump (0.01→0.018)** directly targeted the noise floor regression from Round 4, recovering it from 0.041 to 0.038.

5. **Very low LR (max 6e-6)** prevented catastrophic forgetting of Round 4's learned spectral attention patterns.

---

## Round 4 — Scientific-Grade Promotion Verdict

**Verdict: promotion justified.** The promoted checkpoint improved the official validation quality from **95.15% → 95.50%** while preserving the safety gate: training wrote to a candidate checkpoint first and promoted only after exceeding both the 95% floor and the previously promoted model.

Core validation deltas versus the previous promoted model:

| Metric | Previous | Round 4 | Delta | Verdict |
|--------|----------|---------|-------|---------|
| `overall_quality` | 95.149% | **95.497%** | +0.348 pp | ✅ promoted |
| `mse` | ~0.000488 | **0.000361** | ~−26% | ✅ better reconstruction |
| `psnr` | 42.19 dB | **42.71 dB** | +0.52 dB | ✅ cleaner global signal |
| `ssim` | 0.9973 | **0.9980** | +0.0007 | ✅ stronger structure |
| `corr` | 0.9963 | **0.9967** | +0.0004 | ✅ stronger correlation |
| `rmse_fingerprint` | 0.1221 | **0.1180** | −0.0041 | ✅ better analytical band |
| `noise_floor_std` | 0.0313 | **0.0411** | +31% | ⚠️ acceptable tradeoff |

Key implementation changes reviewed:

- `SpectralAttention1D` is zero-initialized and position-aware, so it can learn fingerprint emphasis without disrupting pretrained weights at load time.
- Promotion is production-safe: candidate checkpoint path, promoted checkpoint path, 95% minimum quality, and improvement-over-current requirement are separate gates.
- New physics losses are active at conservative weights: `extrema_mse`, `sobolev_l1`, and `baseline_drift`.
- New realistic corruptions exist in augmentation: impulse spikes, fringe noise, and polynomial baseline drift. The promoted Round 4 model used `augment: false`, so these augmentations are implemented but not yet validated in a promoted run.

---

## Round 2 — What Changed

Architecture & loss function received significant overhaul:
- ✅ **Skip gates re-enabled** (`use_skip_gates: true`)
- ✅ **SE blocks re-enabled** (`use_se: true`)
- ✅ **Multiscale context enabled** (`use_multiscale_context: true`)
- ✅ **PeakAlignmentLoss added** (`w_peak_align: 0.30`)
- ✅ **Peak center loss added** (`w_peak_center: 1.10`) — softmax-based differentiable centering
- ✅ **Fingerprint-specific losses added** (L1: 0.16, MSE: 0.16, D1: 0.12)
- ✅ **Derivative weights increased** (D1: 0.15→0.60, D2: 0.10→0.35)
- ✅ **Valley undershoot penalty** (`w_valley_under: 0.22`)
- ✅ **Hard sample mining** (14× multiplier on worst samples)

---

## Round 2 — Visual Assessment (Sample 778)

### What Improved ✅

| Metric | v1 | v2 | Δ | Verdict |
|--------|----|----|---|---------|
| `peak_shift_mean_abs` | 0.4062 | **0.1562** | −0.25 | ✅ **Major fix** — hit the ≤0.15 target |
| `noise_floor_std` | 0.0343 | **0.0295** | −0.005 | ✅ Modest improvement |
| `peak_amp_mae` | 0.1277 | **0.1254** | −0.002 | ✅ Marginal |
| `corr` | 0.9995 | **0.9996** | +0.0001 | ✅ Held |

### What Regressed or Stalled 🔴

| Metric | v1 | v2 | Δ | Verdict |
|--------|----|----|---|---------|
| `psnr` | 49.51 | **47.77** | −1.74 dB | 🔴 **Regressed** — new losses competing with reconstruction |
| `rmse_fingerprint` | 0.1835 | **0.1823** | −0.001 | 🔴 **Stalled** — still far from 0.05–0.10 target |
| Peak sharpness | Rounded | Still slightly rounded | — | 🟡 Marginal visual improvement |

### Visually on the Plots

1. **Peak alignment is genuinely better** — In the After plot, the teal line's absorption dips now land precisely on the same x-positions as the dashed blue clean target. The 0.40→0.16 shift reduction is visually confirmed — narrow troughs at ~500, ~700, ~1050 cm⁻¹ no longer wobble sideways.

2. **But peaks remain slightly rounded** — Zooming into the ~1050 cm⁻¹ deep trough and the cluster at 400–500 cm⁻¹, the denoised troughs don't quite reach the full depth of the clean target. The bottoms are subtly "filled in" by ~0.5–1.0 intensity units.

3. **PSNR dropped because the loss landscape is now crowded** — With 18 active loss terms competing (MSE, L1, D1, D2, TV, FFT, amplitude, baseline, smooth_consistency, peak_profile, peak_center, edge_L1, edge_D1, fingerprint_L1, fingerprint_MSE, fingerprint_D1, valley_under, peak_align), the model is trying to satisfy too many objectives simultaneously. The peak alignment improved at the cost of overall reconstruction fidelity.

4. **Fingerprint RMSE barely moved** — Despite adding 3 fingerprint-specific losses, the RMSE only dropped 0.001. The fingerprint weights (0.16, 0.16, 0.12) are being diluted by the sheer number of other loss terms.

---

## Root Cause Analysis — Why the Remaining Issues Persist

### 1. Loss Term Overcrowding
The loss function now has **18 weighted terms**. Many are overlapping (e.g., `w_l1` and `w_fingerprint_l1` both penalise L1 error; `w_d1` and `w_fingerprint_d1` both penalise derivative error). The optimiser sees a blurred, conflicting gradient landscape. Each term individually pushes in the right direction, but collectively they create gradient interference.

**Evidence:** PSNR dropped despite all other metrics holding or improving. This is the classic sign of multi-objective conflict — the model finds a Pareto compromise that satisfies no single objective optimally.

### 2. Too Few Epochs at Too Low LR
The training ran for only **8 epochs** at `max_lr=6e-5`. With 18 loss terms and a complex loss surface, the model barely had time to settle. The history shows selection_score was still climbing at epoch 8 (92.21 → 92.31) — it hadn't plateaued.

### 3. Fingerprint Weight Insufficient Relative to Total
The fingerprint band covers indices 104–726 (~622 points out of 1868). The fingerprint-specific weights (L1: 0.16, MSE: 0.16, D1: 0.12 = total 0.44) are small compared to the global weights (MSE: 1.0, L1: 0.25, D1: 0.6, etc. = total ~4.5). The fingerprint-specific gradient is ~10% of the total, yet this band is where the hardest reconstruction happens.

### 4. Peak Sharpness vs TV Tension
`w_tv: 0.006` penalises high-frequency variation — including the sharp V-shapes at absorption dips. Meanwhile `w_peak_profile: 0.4` tries to preserve those same shapes. These two losses directly oppose each other at peak boundaries. The current TV weight is too high relative to the peak sharpness demand.

---

## Historical Notes

### Round 3 Planning (Superseded)

Round 3 was planned as a "Simplify & Amplify" rebalance of loss weights from v2 defaults (documented in the original review). The actual execution diverged from the plan — Round 4 took a more aggressive approach with new architecture modules (SpectralAttention1D, detail head, positional/derivative bias) and physics-aware losses, which proved more effective than pure weight rebalancing. The Round 3 env var tables and shell blocks have been removed from this document since they no longer reflect the current training pipeline.

### v3 (skipped)

No v3 checkpoint was promoted. The jump from v2 to v4 in the revision history reflects that the first successful promotion after v2 was the scientific-grade Round 4 run.

---

## Success Criteria — Current Status

| Metric | v2 baseline | Original Target | Current (v6) | Status |
|--------|:-----------:|:---------------:|:------------:|--------|
| `peak_shift_mean_abs` | 0.1562 | ≤ 0.12 | **0.193** | 🟡 Aggregate peak shift remains above target |
| `rmse_fingerprint` | 0.1823 | **≤ 0.10** | **0.1117** | 🟡 Significant progress (0.18→0.11), not yet at 0.10 |
| `noise_floor_std` | 0.0295 | ≤ 0.015 | **0.0336** | 🟡 Slightly above v2, well below v4's 0.041 |
| `psnr` | 47.77 | ≥ 49.0 | **43.45** | 🔴 Below target — tradeoff for fingerprint/peak fidelity |
| `overall_quality` | 95.10% | ≥ 96.0% | **95.97%** | ✅ Nearly at target |

> [!NOTE]
> The next work should target the final 0.03 percentage points to reach 96% overall without reintroducing full-backbone drift. The most productive approach so far is scoped metric polish (`TRAINABLE_SCOPE=head`, `CHECKPOINT_SCORE_MODE=overall`), not stronger augmentation or heavier physics losses. Future runs should also set `MAX_PROMOTION_FP_RMSE=0.1118` so a checkpoint cannot promote by trading away fingerprint fidelity.

### Round 7 Attempt — Peak Integrity Constraint

Several candidate runs were tested after the 95.97% checkpoint:

- `attention_head` could cross 96% overall, but did so by increasing fingerprint RMSE and was rejected.
- `head` and `decoder` peak-focused runs improved or held individual peak terms in places, but did not beat the full validation gates.
- A 10% weight interpolation improved sample `696` peak shift to `0.0` and PSNR to ~48.01 dB, but full validation regressed slightly (`overall_quality`, fingerprint RMSE, aggregate peak shift, and peak amplitude), so it was not promoted.

Conclusion: the current best model remains the correct production checkpoint. The FRMSE/peak-integrity gates are necessary because the easiest path to 96% overall is currently to trade away fingerprint fidelity.
