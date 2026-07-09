# Classroom associative memory pipeline

The classroom dataset (`lorinachey/spot-telluride-workshop-dataset`, a Spot
walk-through with synchronized D455 RGB + `odometry_lio_sam`) runs through
five scripts in `scripts/classroom/`, each consuming the previous stage's
output:

```
detect_and_embed_classroom.py
        |  detections.csv, embeddings.pt
        v
plot_embedding_correlation.py   plot_embedding_uncertainty.py   plot_pose_time_heading_correlation.py
        (diagnostics only -- read embeddings.pt / odometry, write PNGs, nothing downstream depends on them)
        v
classroom_associative_memory.py  build   -->  associative_memory_<subset>.pt
classroom_associative_memory.py  query   -->  query_result.png
classroom_associative_memory.py  evaluate -->  evaluate_result.png
classroom_associative_memory.py  demo    -->  demo.mp4
```

This doc covers the math for each stage and the commands to run them in
order. It assumes the `Phasor` algebra (bind $\otimes$, unbind $\oslash$,
bundle, FPE $B^v$, and the two similarity definitions) from
[`docs/associative_memory_math.md`](associative_memory_math.md) §1 — a quick
recap:

$$A \otimes B = A_m B_m \text{ (elementwise)}, \quad A \oslash B = A_m / B_m, \quad \mathrm{bundle}(A^{(1..k)}) = \tfrac{1}{k}\sum_j A^{(j)}, \quad B^v_m = e^{iv\phi_m}$$

$$\mathrm{corr}(Z)_{ij} = \mathrm{Re}(\hat z_i \cdot \bar{\hat z}_j), \qquad \hat z_i = z_i / \lVert z_i \rVert_2 \quad \text{(cosine similarity over } \mathbb{C}^d \cong \mathbb{R}^{2d}\text{)}$$

All code below lives in `scripts/classroom/` unless noted otherwise.

## 1. Detect & embed (`detect_and_embed_classroom.py`)

Pure computer vision, no VSA yet. YOLOv8n runs twice per frame (once for
detection boxes, once with `embed=` to short-circuit to the penultimate-layer
feature vector) over every D455 RGB frame:

- `detections.csv` — long format, one row per detected box (`frame_idx,
  timestamp_ns, class_id, class_name, confidence, x1, y1, x2, y2`).
- `embeddings.pt` — `{"frame_idx": LongTensor[N], "timestamp_ns":
  LongTensor[N], "embedding": FloatTensor[N, 256]}`, one 256-dim feature
  vector per frame. This is the raw content every later stage projects into
  phasor space.

**Run:**

```bash
python scripts/classroom/detect_and_embed_classroom.py \
  --out-dir outputs/classroom_detections
# add --visualize to also render observations.mp4 (RGB + boxes + odometry + embedding self-correlation, side by side)
```

## 2. Diagnostics (optional, read-only against `embeddings.pt`/odometry)

None of these three write anything the `build`/`query`/`evaluate`/`demo`
commands read back in — they're sanity checks on the raw material before
committing to a memory build.

### 2a. Projection fidelity (`plot_embedding_correlation.py`)

Raw-embedding cosine self-correlation next to the same embeddings' cosine
self-correlation *after* `random_project_to_phasor` (§4 below) projects them
to HD space — how much of the raw neighbor structure survives the
projection, via `docs/associative_memory_math.md`'s §4 metrics:

$$\mathrm{fidelity} = \mathrm{corr}\big(\{\mathrm{corr}^{\text{raw}}_{ij}\}_{i<j},\ \{\mathrm{corr}^{\text{content}}_{ij}\}_{i<j}\big), \qquad \mathrm{orthogonality} = 1 - \tfrac{1}{\binom{N}{2}}\sum_{i<j}\lvert \mathrm{corr}^{\text{content}}_{ij}\rvert$$

```bash
python scripts/classroom/plot_embedding_correlation.py
```

### 2b. Embedding uncertainty (`plot_embedding_uncertainty.py`)

Two "how surprising is this frame" signals, computed on the raw-embedding
cosine self-correlation matrix $C$ and again on the phasor-projected version
(`phasor_correlation_matrix`), to see how much survives projection:

$$\text{per-frame (local)}: \quad u^{\text{local}}_i = 1 - C_{i,\,i-1}, \qquad u^{\text{local}}_0 = 0$$

$$\text{global}: \quad u^{\text{global}}_i = 1 - \frac{1}{N-1}\sum_{j \ne i} C_{ij}$$

Local uncertainty is just the matrix's immediate sub-diagonal (how big was
the last step); global is each frame's row mean excluding the diagonal (how
atypical this frame is relative to the *whole* video — catches a segment
that drifts smoothly frame-to-frame but ends up somewhere unlike the rest of
the walk). `top_k_peaks` greedily flags each signal's `--top-k` largest
values, skipping any candidate within `--min-peak-distance` frames of an
already-picked peak so one drawn-out transition isn't counted several times.

This script's `select_subset`-equivalent (`adjacent_frame_uncertainty` /
`global_frame_uncertainty`) is exactly what `classroom_associative_memory.py
build --subset uncertain` uses to pick which frames to bundle — see §5.

```bash
python scripts/classroom/plot_embedding_uncertainty.py --hd-dim 256
```

### 2c. Context-channel correlation (`plot_pose_time_heading_correlation.py`)

Encodes position/heading/time independently (same encoders as §3 below,
**no cross-channel binding here** — the point is to see what each axis
contributes on its own) and plots each one's `phasor_correlation_matrix`
side by side, plus each channel's `orthogonality_score`.

```bash
python scripts/classroom/plot_pose_time_heading_correlation.py --hd-dim 256
```

## 3. Context encoders (shared by build / query / evaluate / demo)

Three independent FPE channels, each against its own random base phasor(s)
of dimension `hd_dim`, with a length-scale controlling kernel width (smaller
`length_scale` → faster phase rotation per physical unit → narrower
similarity kernel, more distinguishable but less generalizing):

**Position** — two bases $B_x, B_y$, bound together so nearby $(x,y)$ correlate:

$$\mathrm{ctx}^{\text{pos}}(x, y) = B_x^{\,x/\ell_{\text{pos}}} \otimes B_y^{\,y/\ell_{\text{pos}}}$$

**Heading** — yaw wraps at $\pm\pi$, so an ordinary continuous-random-phase
base isn't circular-safe ($B^{v+2\pi} \ne B^v$). `Phasor(circular=True)`
instead draws each dimension's phase as a random **integer** frequency $k_m
\in [-3, 3] \setminus \{0\}$ (capped at 3, since $|k|>3$ can't be recovered
from where it wraps on the unit circle — see `Phasor._MAX_CIRCULAR_FREQ`),
which makes every dimension exactly $2\pi$-periodic:

$$\mathrm{ctx}^{\text{head}}(\psi) = B_{\text{head}}^{\,\psi}, \qquad (B_{\text{head}})_m = e^{ik_m \cdot 0}, \quad (B_{\text{head}}^{\,\psi})_m = e^{ik_m\psi}$$

**Time** — a single base against row position (capture order, not the
dataset's own `frame_idx`, which isn't monotonic once sorted by
`timestamp_ns`):

$$\mathrm{ctx}^{\text{time}}(t) = B_t^{\,t/\ell_{\text{time}}}$$

**Content** — every frame's 256-dim YOLO embedding $e_n$ projected to
`hd_dim` phasor content via `random_project_to_phasor` (the "random-proj"
encoder from `docs/associative_memory_math.md` §2: Gaussian projection
$W\!\sim\!\mathcal N(0, 1/d_{in})$ to $(I,Q)$ pairs, normalized onto the unit
circle, phase kept / magnitude discarded). `--pca-whiten` is an optional
*preprocessing* step before that same projection — not the separate
"pca-fpe" encoder from the general doc — decorrelating and standardizing
$e_n$ first via `pca_components`'s SVD:

$$\tilde e = \frac{(e - \bar e)V_{:,1:K}}{\sigma_{1:K}} \quad \text{(unit-variance per component)}, \qquad \mathrm{content}_n = \mathrm{RandomProjectToPhasor}(\tilde e_n)$$

This targets a specific problem: raw embedding cosine similarity on this
dataset sits in a tight 0.70–1.00 band, so whitening tries to spread out the
discriminative structure a single dominant shared direction is otherwise
swamping, before the random projection ever sees it.

## 4. Build (`classroom_associative_memory.py build`)

Unlike the general pipeline's single quad-binding trace ($\mathrm{content}
\otimes B_t^t \otimes B_x^x \otimes B_y^y$, all bound into *one* memory —
`docs/associative_memory_math.md` §3), the classroom build keeps the three
context channels in **three separate memories**, so each only has to
discriminate along its own axis:

$$M_{\text{axis}} = \frac{1}{|S|}\sum_{n \in S} \mathrm{content}_n \otimes \mathrm{ctx}^{\text{axis}}_n, \qquad \text{axis} \in \{\text{position}, \text{heading}, \text{time}\}$$

where $S$ is the subset of frame indices actually bundled in, chosen by
`--subset`:

- **`all`** — $S = \{1,\dots,N\}$, every frame. The real stress test: does
  bundling ~2,478 distinct traces overwhelm a `hd_dim`-dim memory? (Bundling
  capacity scales roughly with dimensionality, so noise is expected — this
  is why `--hd-dim 8192` recalls cleaner than the default 2048.)
- **`uncertain`** — reuses §2b's uncertainty signals on the *raw* embeddings,
  keeps frames at or above `--uncertainty-quantile` (default 0.8 → top 20%
  by `--uncertainty-type` local/global uncertainty). Fewer, more distinctive
  items bundled together → less crosstalk.
- **`stride`** — every `--stride`-th frame (a train split), holding out the
  rest for `demo` (§7) to query with content the memory never saw.

The saved `.pt` also stores the codebook (content + context ground truth) for
exactly the frames in $S$ — `query`/`evaluate` only ever cleanup-match
against that same subset, never the full dataset.

**Run** (the 8192-dim, PCA-whitened, uncertain-subset build from earlier in
this conversation):

```bash
python scripts/classroom/classroom_associative_memory.py build \
  --subset uncertain --uncertainty-quantile 0.8 --hd-dim 8192 --pca-whiten \
  --out outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt
```

## 5. Query (`classroom_associative_memory.py query`)

"What did I see at this position/heading/time?" — unbind the axis's memory
by a candidate context, then cleanup (`best_matches`-equivalent) against that
axis's codebook of stored content vectors:

$$\mathrm{residual} = M_{\text{axis}} \oslash \mathrm{ctx}^{\text{axis}}(\text{query}), \qquad \mathrm{sims}_j = \mathrm{corr}(\mathrm{residual}, \mathrm{content}_j)_{j \in S}$$

Ranked by $\mathrm{sims}_j$ descending, top-`--top-k` printed per queried
axis. If more than one of `--query-x/--query-y`, `--query-yaw`,
`--query-time` is given, a **combined** ranking sums similarities
elementwise across axes before taking the argmax — a simple late-fusion, not
a joint bind:

$$\mathrm{sims}^{\text{combined}}_j = \sum_{\text{axis queried}} \mathrm{sims}^{\text{axis}}_j$$

```bash
python scripts/classroom/classroom_associative_memory.py query \
  --memory outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt \
  --query-x -2.0 --query-y 3.0
```

## 6. Evaluate (`classroom_associative_memory.py evaluate`)

Self-recall: for every stored item $n \in S$, unbind $M_{\text{axis}}$ by
that item's own true context, cleanup against the *whole* codebook, and
measure how far the top-1 recalled item's true context is from $n$'s own —
in physical units, since exact top-1 match is too harsh a metric on its own
(near-duplicate frames make an "almost right" recall look identical to a
wildly wrong one):

$$\hat n = \arg\max_{j \in S} \mathrm{corr}\big(M_{\text{axis}} \oslash \mathrm{ctx}^{\text{axis}}_n,\ \mathrm{content}_j\big), \qquad \mathrm{err}_n = d_{\text{axis}}(\mathrm{truth}_n,\ \mathrm{truth}_{\hat n})$$

with $d_{\text{axis}}$ the physically appropriate distance per channel:

$$d_{\text{position}} = \sqrt{(x_n - x_{\hat n})^2 + (y_n - y_{\hat n})^2} \ \text{[m]}, \qquad d_{\text{heading}} = |\mathrm{atan2}(\sin\Delta\psi, \cos\Delta\psi)| \ \text{[rad]}, \qquad d_{\text{time}} = |t_n - t_{\hat n}| \ \text{[frames]}$$

Reported as both L1 (mean absolute error) and L2 (root-mean-square) per axis,
directly comparable to each other since both stay in the same physical unit
(unlike raw MSE, which would be squared units):

$$L1 = \frac{1}{|S|}\sum_n |\mathrm{err}_n|, \qquad L2 = \sqrt{\frac{1}{|S|}\sum_n \mathrm{err}_n^2}$$

The output plot also shows, per axis, the embedding self-correlation, the
context channel's own self-correlation, the bound (content⊗context) vectors'
self-correlation, and how similar the final bundled memory is to each
individual pre-bundle trace (low = that trace got drowned out by bundling).

```bash
python scripts/classroom/classroom_associative_memory.py evaluate \
  --memory outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt
```

## 7. Demo (`classroom_associative_memory.py demo`)

Reverse query direction over **held-out** frames (those not in $S$): instead
of context → content, it's content → context — "where/when have I seen
something like this before?" Unbind each memory by the *current* held-out
frame's own content:

$$\mathrm{residual}_{\text{axis}} = M_{\text{axis}} \oslash \mathrm{content}_{\text{val}}$$

Rather than collapsing straight to an argmax, position and heading are swept
**continuously** — over a dense $(x,y)$ grid and a dense angle sweep,
respectively — so genuine ambiguity (two similar-looking spots, a
symmetric room) is visible as a heatmap/polar curve instead of hidden behind
one arbitrary best guess:

$$\mathrm{heat}(x, y) = \mathrm{corr}\big(\mathrm{residual}_{\text{position}},\ \mathrm{ctx}^{\text{pos}}(x,y)\big), \qquad \mathrm{polar}(\psi) = \mathrm{corr}\big(\mathrm{residual}_{\text{heading}},\ \mathrm{ctx}^{\text{head}}(\psi)\big)$$

Time recall stays a discrete lookup against the train codebook (recalling a
specific past frame, not a continuous quantity) — same cleanup as §6 but
against $S$ from the *train* build only.

Renders one video frame per held-out frame in chronological order, three
panels: current RGB view (with detections), the position heatmap, the
heading polar plot.

```bash
python scripts/classroom/classroom_associative_memory.py demo \
  --memory outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt \
  --out-path outputs/classroom_detections/demo_uncertain_8192.mp4
```

Note: `demo`'s held-out set is *whatever wasn't bundled into the codebook*,
regardless of `--subset`. For `--subset stride`, that's a clean, intentional
2/3 val split. For `--subset uncertain`, held-out is just "the ~80% of
frames that weren't among the most distinctive" — still runs, but isn't a
designed train/val split the way `stride` is.

## 8. End-to-end example

```bash
# 1. detect + embed (once per dataset)
python scripts/classroom/detect_and_embed_classroom.py --out-dir outputs/classroom_detections

# 2. diagnostics (optional, informs --subset/--hd-dim choices below)
python scripts/classroom/plot_embedding_uncertainty.py
python scripts/classroom/plot_pose_time_heading_correlation.py

# 3. build a memory
python scripts/classroom/classroom_associative_memory.py build \
  --subset uncertain --uncertainty-quantile 0.8 --hd-dim 8192 --pca-whiten \
  --out outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt

# 4a. point query
python scripts/classroom/classroom_associative_memory.py query \
  --memory outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt \
  --query-x -2.0 --query-y 3.0

# 4b. quantitative self-recall error
python scripts/classroom/classroom_associative_memory.py evaluate \
  --memory outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt

# 4c. qualitative held-out video
python scripts/classroom/classroom_associative_memory.py demo \
  --memory outputs/classroom_detections/associative_memory_uncertain_hd8192_pca.pt \
  --out-path outputs/classroom_detections/demo_uncertain_8192.mp4
```
