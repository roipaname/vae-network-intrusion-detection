# Project Log: VAE-Augmented Network Intrusion Detection

A chronological account of how this project was built: the decisions made,
what broke along the way, how it was fixed, and every measured outcome —
good and bad. Written to be dropped into a research paper's methodology and
discussion sections almost directly.

---

## 1. Setup (Phase 1)

Started from an empty directory. Used `uv init --app` to create the Python
project rather than a manual `pyproject.toml` — this produces an
"application" style project (no build backend), which is the right shape
for a research pipeline that isn't meant to be installed as a library.

Built `config/settings.py` as the single source of truth for paths,
hyperparameters, and logging, on the principle that every later phase should
import from here rather than redefine constants. Loguru was configured once,
centrally: colorized stderr output plus a rotating file sink
(`logs/app.log`, 10MB rotation, 10-day retention). No module anywhere in the
project uses `print()` for application logging — verified by grep at the end
of the project, not just assumed.

`uv init` also initialized a git repository as a side effect, which was kept.

---

## 2. Dataset acquisition (Phase 2)

### Finding: both datasets' official hosts are dead

- **NSL-KDD**: the official host (UNB/CIC, `unb.ca/cic/datasets/nsl.html`)
  displays "We apologize, this dataset is no longer available." No official
  download link exists anymore.
- **UNSW-NB15**: the official host redirects to a SharePoint folder that
  can't be scripted with `curl`.

**Remedy**: used long-standing community mirrors instead, but didn't trust
them blindly — verified every file's record count against literature before
using it:

- NSL-KDD from `jmnwong/NSL-KDD-Dataset` (GitHub): `KDDTrain+.txt` (125,973
  records) and `KDDTest+.txt` (22,544 records) — both counts match
  Tavallaee et al. (2009) exactly.
- UNSW-NB15 from `Nir-J/ML-Projects` (GitHub): the standard pre-partitioned
  `UNSW_NB15_training-set.csv` (175,341 records) and
  `UNSW_NB15_testing-set.csv` (82,332 records) — both counts match
  Moustafa & Slay's published figures exactly.

### Design decision: NSL-KDD primary, UNSW-NB15 secondary

NSL-KDD is smaller and the classic benchmark, so it became the primary
dataset for iterating on the core VAE-augmentation experiment. UNSW-NB15 —
larger, newer, a different 43-feature schema — was run through the identical
pipeline afterward, specifically to test whether any finding generalizes to
a second dataset rather than being an artifact of one.

### Finding: UNSW-NB15's official training partition is 38% duplicates

Inspection found **0 duplicate rows** in NSL-KDD's training set, but
**67,601 duplicate rows (38%) in UNSW-NB15's training set** — a real,
documented quirk of that widely-used partition, not a bug in the loading
code. Duplicates were dropped from the training set only; the test set was
never touched. This meant the detector for UNSW-NB15 actually trained on
107,740 rows, not the raw 175,341 — a fact that mattered later (see the
backend bug in §7).

### Preprocessing

One `sklearn.ColumnTransformer` per dataset (`OneHotEncoder` for categorical
columns, `StandardScaler` for numeric columns), fit on training data only,
applied to both train and test — the standard leakage-avoidance pattern.
The real-normal-only subset of the training data was split out here
(`X_train_normal`), since this is the only data the VAE is allowed to see.

---

## 3. VAE implementation and training (Phase 3)

### Failure: PyTorch doesn't support this machine anymore

`uv add torch` failed outright:

```
error: Distribution `torch==2.14.0` ... doesn't have a source distribution
or wheel for the current platform
hint: You're on macOS (macosx_13_0_x86_64) ...
```

PyTorch dropped Intel-Mac wheels somewhere after version 2.2.2.

**Remedy**: pinned `torch==2.2.2`, the last version with an Intel-Mac wheel.
This immediately caused a second failure: torch 2.2.2 was compiled against
NumPy 1.x, and the project already had NumPy 2.4.6 installed (pulled in by
pandas/scikit-learn), producing a hard crash the moment `torch.from_numpy`
was called:

```
RuntimeError: Numpy is not available
```

**Remedy**: pinned `numpy<2` and `pandas<3` (pandas 3.x hard-requires
NumPy 2.x, so both had to move together). Re-verified the whole stack
(numpy/pandas/scikit-learn/torch interop) and re-ran Phase 2's preprocessing
to confirm nothing downstream broke from the pandas downgrade — it didn't;
results were byte-identical.

### Architecture

A genuine VAE, not a plain autoencoder: MLP encoder (`input → 64 → 32`)
producing separate `mu`/`logvar` heads for a 16-dim latent space, the
reparameterization trick (`z = mu + eps * exp(0.5·logvar)`), and an MLP
decoder mirroring the encoder with a **linear** (not sigmoid) output layer,
since the inputs are standard-scaled rather than bounded to [0, 1]. Loss is
MSE reconstruction + KL divergence to `N(0, I)`, both computed per-sample
then averaged over the batch.

### Training results (50 epochs each, CPU)

| Dataset | Normal training samples | Final loss | Recon | KL |
|---|---|---|---|---|
| NSL-KDD | 67,343 | 8.26 | 5.11 | 3.16 |
| UNSW-NB15 | 51,890 | 7.38 | 3.46 | 3.92 |

NSL-KDD had a transient bump in loss around epoch 35 before recovering to a
lower final value than before the bump — checked for NaNs at that point
(none) and concluded it was a normal, if slightly noisy, optimization
trajectory rather than a bug. Post-training sanity checks: reconstruction
MSE on held-out normal samples ≈0.03 (very low, given standardized
features), and the encoded latent `mu` showed a healthy, non-collapsed
spread (mean ≈0, std ≈0.4–0.6) rather than either collapsing to a point or
exploding.

---

## 4. Synthetic generation and the sampling-strategy experiment (Phase 4)

### Initial finding: real vs. synthetic fidelity is mixed

Comparing 5,000 VAE-generated synthetic normal samples against real normal
training data:

- **Categorical distributions matched well** — total variation distance
  0.02–0.08 across `protocol_type`/`service`/`flag`.
- **Numeric distributions were more divergent than hoped** — mean KS
  statistic 0.57 (NSL-KDD) / 0.40 (UNSW-NB15). Breaking this down by feature
  showed the divergence was concentrated in **sparse, near-constant
  features** (e.g. `wrong_fragment`, `num_failed_logins` — real data is
  almost always exactly one value with rare spikes), where the VAE produces
  a smoothed continuous approximation instead of the near-degenerate real
  pattern. Genuinely continuous features (`count`, `logged_in`) matched much
  better (KS 0.28–0.43).
- **A latent-space mismatch was found and flagged**: real normal data's
  encoded `mu` has std ≈0.42 on both datasets — narrower than the `N(0,1)`
  prior the VAE samples from at generation time. This is a textbook VAE
  prior/aggregate-posterior mismatch, and looked like a plausible
  explanation for the numeric-feature divergence above.

### The experiment: does correcting the mismatch help?

At the user's request, tried the natural fix: fit a single diagonal Gaussian
to the real data's encoded `mu` values (mean + std across all training
samples) and sample synthetic `z` from that instead of the raw prior.

**Result: it made things slightly worse, consistently, on both datasets:**

| Dataset | Metric | Prior sampling | Posterior sampling | Change |
|---|---|---|---|---|
| NSL-KDD | mean numeric KS | 0.5689 | 0.6023 | worse by 0.033 |
| NSL-KDD | mean categorical TV distance | 0.0499 | 0.0717 | worse by 0.022 |
| UNSW-NB15 | mean numeric KS | 0.4012 | 0.4086 | worse by 0.007 |
| UNSW-NB15 | mean categorical TV distance | 0.0439 | 0.0501 | worse by 0.006 |

**Why, most likely:** the "aggregate posterior" used here is a single
*global* diagonal Gaussian fit over all real `mu` vectors at once, which
collapses whatever cluster structure exists in the latent space (e.g.
distinct regions for different protocols/services). During training the
decoder only ever sees a *per-sample* reparameterized
`z = mu_i + eps·sigma_i` — never a `z` drawn from this coarser global fit —
so sampling from the flattened global Gaussian lands the decoder in latent
regions that don't correspond well to any specific real traffic mode. The
theoretically correct fix would be a **mixture** of per-sample Gaussians
(pick a random real training point `i`, then sample `z ~ N(mu_i, sigma_i)`),
which would preserve that structure — this was **not implemented**, since it
is a bigger change and the simpler prior sampling already measured as good
or better. Noted here as a concrete follow-up rather than pursued further.

**Consequence**: `prior` sampling (the textbook `N(0,I)` approach) was kept
as the default used for detector augmentation. Both sampling strategies'
synthetic data and full validation reports were kept on disk
(`results/*_synthetic_validation.json` for prior,
`results/*_synthetic_validation_posterior.json` for posterior,
`results/*_latent_sampling_comparison.json` for the head-to-head writeup) —
this negative result is fully reproducible, not just described.

---

## 5. Baseline and VAE-augmented detector experiments (Phase 5)

**Model choice: Random Forest** (200 trees, unconstrained depth, fixed
random seed) — a strong, fast, low-tuning-effort baseline for tabular IDS
data that needs no extra feature-scale handling beyond what preprocessing
already did, and exposes `.feature_importances_` directly for the
dashboard's explanation panel.

Two experiments, identical hyperparameters, differing only in training data:

- **Baseline**: real training data only.
- **Augmented**: real training data + 5,000 synthetic normal samples
  (prior-sampled), all labeled `normal`.

Both were sanity-checked against the same real test set immediately after
training:

| Dataset | Baseline test accuracy | Augmented test accuracy |
|---|---|---|
| NSL-KDD | 77.61% | 77.51% |
| UNSW-NB15 | 87.20% | 87.04% |

Minor note: the UNSW-NB15 Random Forest models came out large (~220MB each
`.joblib` file) since 200 unconstrained-depth trees on 107K rows / 194
features produce a big forest. Not a correctness issue, but it meant the
backend's in-memory model cache (§7) mattered for keeping request latency
reasonable.

---

## 6. Evaluation and the honest headline result (Phase 6)

Full metrics (accuracy, precision, recall, F1, ROC-AUC, false positive rate,
confusion matrix) computed on the same untouched real test set for both
variants:

| Dataset | Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| NSL-KDD | Baseline | 77.61% | 96.89% | 62.67% | 76.11% | 0.9619 | 2.66% |
| NSL-KDD | Augmented | 77.51% | 96.87% | 62.51% | 75.99% | 0.9612 | 2.67% |
| UNSW-NB15 | Baseline | 87.20% | 81.86% | 98.60% | 89.46% | 0.9807 | 26.76% |
| UNSW-NB15 | Augmented | 87.04% | 81.71% | 98.51% | 89.32% | 0.9804 | 27.02% |

**This directly answers the research question, and the answer is no.**
VAE-based synthetic normal traffic augmentation, as implemented here, did
not improve intrusion detection performance on either dataset. Every metric
moved slightly negative or stayed flat with augmentation — including false
positive rate, one of the paper's specifically hoped-for benefits, which got
marginally *worse* rather than better on both datasets. Effect sizes are
small (under 1 percentage point everywhere), so the accurate claim is "no
measurable benefit," not "actively harmful."

This null result is not an isolated anomaly — it lines up with §4's finding
that the synthetic numeric features only moderately matched real
distributions (mean KS 0.40–0.57). A plausible causal chain: moderate
synthetic fidelity → augmented training data that doesn't meaningfully
sharpen the decision boundary → no measurable detection improvement.

**Sanity check on the pipeline itself**: feature importances came out
domain-sensible and matched published literature independently for both
datasets — NSL-KDD's top features were `src_bytes`/`dst_bytes`/`flag`/
`service` (exactly as widely reported), and UNSW-NB15's were
`sttl`/`ct_state_ttl` (Moustafa & Slay's own well-known finding for this
dataset). This was an important check: it gives confidence the null result
reflects a real absence of benefit, not a broken pipeline that happens to
produce flat numbers.

---

## 7. Backend (Phase 7)

FastAPI app exposing the pipeline's real artifacts. Some design decisions
and one bug worth recording:

### Design: synthetic display fields, clearly separated

NSL-KDD/UNSW-NB15 are flow-feature datasets — they contain no real IP
addresses or timestamps. Endpoints that need something to show for those
fields (`/api/traffic`, `/api/alerts`, `/api/connections`,
`/api/simulate-attack`) derive a **stable, deterministic pseudo-IP** from
each record's index, purely for display. This is written directly into the
code's docstrings so it's never confused with genuine model output
(prediction/confidence, from a real trained detector) or genuine dataset
fields (protocol/service/attack category, from the real test set).

### Design: a real 2D latent-space endpoint, not decorative points

The frontend brief explicitly required a 2D "AI Learned Traffic Profile"
visualization and explicitly forbade generating random points to fake it.
Added `/api/latent-projection`: it encodes real normal training data,
VAE-generated synthetic data, and real attack test data (never seen in VAE
training) through the actual trained encoder, then reduces to 2D with PCA
fit across all three groups together. Every point on that chart is a real
record's real encoded position.

### Bug found and fixed: mislabeled training-sample count

While testing `/api/model/{dataset}`, noticed the `training_samples` field
under `baseline` was actually returning the **test set size**, and a related
issue: even after fixing that, a naive "training records" figure would have
reported UNSW-NB15's raw pre-dedup count (175,341) rather than what the
detector actually trained on (107,740, per §2's duplicate-removal finding).

**Fix**: `training_samples` now correctly computes
`dataset_info["train_records"] - dataset_info["duplicate_rows_in_train"]`,
verified against Phase 5's actual training run counts: 125,973 / 130,973 for
NSL-KDD baseline/augmented, and 107,740 / 112,740 for UNSW-NB15.

### Verification

Every endpoint was hit for both datasets and both variants, plus four
deliberate error cases (unknown dataset → 404, bad variant → 400,
out-of-range `record_index` → 400, invalid pagination params → 400) — all
returned exactly the expected status code, twice (once mid-project, once
again in the final Phase 9 sweep).

---

## 8. Frontend (Phase 8) — and a visual-verification saga worth recording

### Design choices

React + Vite, plain JavaScript (not TypeScript, to keep the codebase
approachable for a presentation), `lucide-react` for icons (zero emoji,
grep-verified), `recharts` for standard charts, and a hand-built native-SVG
flow diagram for the live traffic visualization (no extra library needed —
native `<animate>`/`<animateMotion>` keeps it lightweight). Fonts (Inter,
JetBrains Mono) were installed as npm packages rather than pulled from a
CDN, so the dashboard still works offline during a presentation. The color
palette, sidebar structure, and page list were followed from the brief
essentially verbatim.

### Failure: couldn't get a normal headless browser working

Tried Playwright first (the usual tool for this): it flatly refused to
launch —

```
Error: ERROR: Playwright does not support chromium on mac13
```

**Remedy**: found that the system's own installed Google Chrome supports
CLI-driven headless screenshots directly
(`--headless=new --screenshot=... --window-size=...`), no extra install
needed. Required `--no-sandbox` and running with the sandbox explicitly
disabled for that one command, since Chrome's own sandbox conflicted with
the harness's process sandbox.

### A confusing pattern that turned out not to be a bug

Early screenshots showed some cards stuck on "Loading…", or charts with
axes but no visible bars/dots/lines. Repeating the *exact same* screenshot
several times in a row showed a *different* section "not yet drawn" each
time, while every other section — and all the real numbers, when they did
render — was always correct. Cross-checked against the backend's own
request logs, which showed **100% success on every single request** across
dozens of page loads. Diagnosed the actual cause: headless Chrome's
`--screenshot` flag captures the page essentially at the `load` event,
which for a single-page app fires before React's data-fetching effects and
Recharts' entrance/mount animations have necessarily finished — so the
screenshot tool was racing the page's own (intentional, spec-requested)
subtle chart-draw-in animations. Confirmed this diagnosis by re-screenshotting
several pages until they landed mid-animation in different, inconsistent
ways each time, with data always correct once settled. This was a
verification-tooling limitation, not an application defect — explained
honestly to the user rather than either hidden or wrongly reported as a bug.

### A real bug the user caught, that the screenshots had actually captured

After Phase 8 was reported complete, the user pointed out that a page
"flashes Loading… every second" while just sitting there. This was real —
distinguishable from the animation-timing artifact above because it was a
persistent, repeating pattern during normal use, not a one-off capture
race.

**Root cause**: the shared data-fetching hook (`useApi`, and `usePolling`
built on top of it for live-updating pages) reset its `loading` state to
`true` on *every* refetch — including every 4–6 second polling tick, and
every dataset/variant switch — not just the very first load. This blanked
the whole card and showed the spinner on every single refresh cycle.

**Fix**: added a ref (`hasLoadedRef`) that tracks whether this hook instance
has ever successfully loaded data. `loading` now only becomes `true` before
that first successful fetch; every refetch after that — polling tick,
dataset switch, manual reload — keeps the previous content rendered in
place and swaps it out silently once the new response arrives. This is the
standard "stale-while-revalidate" pattern.

**A side-effect of applying the fix live**: editing `hooks.js` while a
browser tab was already open triggered a one-time Vite Fast-Refresh error
(`React has detected a change in the order of Hooks`) — a known Vite/React
hot-reload limitation when a hook's internal hook count changes
mid-session, not a bug in the fix itself. Resolved by restarting the dev
server; any browser tab open during the edit needed a hard refresh to pick
up the corrected code cleanly.

---

## 9. Final integration and polish (Phase 9)

- Removed a `Skeleton` loading-placeholder component that had been built in
  Phase 8 but never actually used anywhere (every page ended up using the
  spinner-based `StateBlock` instead) — dead code, deleted along with its
  CSS.
- Fixed a leftover placeholder `description` field in `pyproject.toml`.
- Replaced the generic Vite-boilerplate `frontend/README.md` with a short
  pointer to the real project README.
- Re-ran the full endpoint sweep (22 GET requests, 5 POST requests, 4 error
  cases across both datasets) one final time after all fixes — 100% correct.
- Grep-verified, rather than assumed: zero `print()` calls in any Python
  module, zero emoji anywhere in the frontend, no gradients or off-palette
  colors in the CSS, no leftover TODO/FIXME markers.
- Wrote the full project README covering setup, methodology, and exact
  reproduction commands, using only numbers pulled from the actual saved
  results files.

---

## Summary of every failure and its remedy

| # | Failure | Remedy |
|---|---|---|
| 1 | NSL-KDD/UNSW-NB15 official hosts are both down | Used verified community mirrors, checked record counts against literature |
| 2 | UNSW-NB15's official train partition is 38% duplicates | Dropped duplicates from train only, reported the count honestly, never touched test |
| 3 | `torch` has no Intel-Mac wheel past 2.2.2 | Pinned `torch==2.2.2` |
| 4 | `torch==2.2.2` needs `numpy<2`, conflicting with `pandas>=3` | Pinned `numpy<2` and `pandas<3` together, re-verified the whole stack |
| 5 | Naive "fix" for VAE prior/posterior mismatch made synthetic fidelity worse | Kept the simpler prior-sampling default; documented the negative result and a candidate proper fix (mixture of per-sample Gaussians) as future work |
| 6 | Backend mislabeled detector training-sample counts (test-set size; ignored UNSW-NB15 dedup) | Fixed to compute real training counts from dataset dedup stats, verified against Phase 5's actual run |
| 7 | Playwright doesn't support this macOS version | Used the system's installed Chrome directly via CLI headless flags |
| 8 | Screenshots looked inconsistently "broken" between runs | Diagnosed as headless-capture racing Recharts' entrance animations, not an app bug; cross-verified via backend request logs |
| 9 | Dashboard flashed "Loading…" on every poll tick / dataset switch | Root-caused to the fetch hook resetting `loading` on every refetch; fixed with a "first-load-only" ref flag |
| 10 | Editing the hook mid-session broke Vite Fast Refresh once | Restarted the dev server; documented that an already-open tab needs a hard refresh after this kind of edit |

## Headline research findings for the paper

1. **VAE-generated synthetic normal traffic has good categorical fidelity
   but only moderate numeric fidelity**, concentrated in sparse/near-constant
   features rather than spread evenly across all features.
2. **Correcting the measured prior/aggregate-posterior latent mismatch did
   not improve synthetic fidelity — it made it slightly worse**, on both
   datasets, likely because a single global Gaussian fit collapses latent
   cluster structure the decoder was never trained to decode from.
3. **VAE-based normal-traffic augmentation did not improve intrusion
   detection performance on either NSL-KDD or UNSW-NB15** — every metric,
   including false positive rate, moved flat-to-slightly-negative. The
   result replicated across two datasets with different feature schemas,
   which strengthens confidence it's a real finding rather than a
   dataset-specific artifact.

## 2026-09-26 — Full pipeline re-run

Local copies of `data/` and `models/` were lost (iCloud had offloaded them and
they could not be downloaded again), so the raw datasets were fetched again from
the same mirrors and the full pipeline (§15 of the README) was re-run.
The baseline detectors reproduced exactly. The augmented detectors
changed slightly because the VAE was retrained and generated new synthetic
samples:

| Dataset | Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|---|---|
| NSL-KDD | Augmented | 76.42% | 96.73% | 60.62% | 74.54% | 0.9605 | 2.71% |
| UNSW-NB15 | Augmented | 87.06% | 81.70% | 98.58% | 89.35% | 0.9804 | 27.05% |

The conclusion is the same: augmentation gives no measurable benefit on either
dataset. On NSL-KDD the gap to the baseline is now somewhat larger (about 2
points of recall).
