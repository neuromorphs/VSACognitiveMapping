# Results so far, and what turned out to be wrong

Read this before quoting any number from this repo. Several published-looking
figures in older documents on this branch are superseded, and a few evaluation
setups were measuring something other than what they claimed.

Full chronological record with method detail: `wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md`,
progress log entries (x) through (ag).

---

## The headline findings

### 1. Isotropy predicts capacity, not key quality

Four independent designs, all pointing the same way:

- Whitening harder degrades retrieval monotonically: 0.720 → 0.693 → 0.654 →
  0.623 → 0.592 as 32, 64, 128 then all 256 directions are equalised.
- Frame embeddings are *more* isotropic than object crops (IsoScore 0.0315 vs
  0.0222) and retrieve far worse (0.435 vs 0.720).
- `school_run1` is more anisotropic than the classroom and retrieves better.
- ResNet-50 has nearly the worst IsoScore of the trained encoders and the best
  retrieval.

A perfectly isotropic encoding of the wrong thing retrieves nothing. Isotropy is
necessary for capacity and insufficient for a useful key.

### 2. Crosstalk comes from the shared mean, not the spectrum

Centring leaves IsoScore and effective rank **bit-identical** — they are
computed on the centred covariance, so centring cannot change them — yet halves
crosstalk (χ 10.55 → 5.24). Same spectrum, 2× different crosstalk.

Consequence: **centre or z-score, do not whiten.** Per-dimension z-scoring
improves retrieval *and* cuts crosstalk 2.3×, with no trade-off. Whitening to
128 components scores **below chance** on held-out data.

### 3. This is not the anisotropy the NLP literature describes

Timkey & van Schijndel (EMNLP 2021) found 1–5 "rogue dimensions" cause
transformer anisotropy; delete five and it collapses. Not here:

| | top dim's share | dims for half |
|---|---|---|
| our crops | **5.3%** | 49 of 256 |
| GPT-2 L12 | 76% | ~1 |
| BERT L11 | 88% | ~1 |
| XLNet | 99% | ~1 |

Two orders of magnitude away. The cheap known fix does not apply. The untrained
network makes the distinction cleanest: effective rank **1.0** (maximally
low-rank) with a top-*coordinate* share of only **1.3%** — a dense shared
direction, not one bad axis.

### 4. Whitening's benefit is encoder-dependent; its floor is not

Matched N = 2,429, identical boxes:

| encoder | raw χ | whitened χ | gain |
|---|---|---|---|
| yolov8n | 180.85 | 9.11 | **19.9×** |
| resnet50 | 87.96 | 9.83 | 8.9× |
| dinov2 | 79.83 | 9.87 | **8.1×** |
| untrained | 201.80 | 11.03 | 18.3× |

Raw crosstalk spans 2.5×; whitened spans **1.2×**. Whitening lands every
backbone — including a randomly initialised one — on the same floor. Only the
distance travelled differs.

Why the floor exists: after whitening, content vectors still sit ~7× above the
random floor, and splitting by time shows where it comes from — frames <4 s
apart sit at |cos| ≈ 0.29 while frames >40 s apart approach the floor. It is
**scene redundancy**, a property of the room, and no reweighting of feature axes
can remove it.

### 5. Two literature predictions tested

- **Liang et al. (NeurIPS 2022) replicated to three decimals.** Predicted
  untrained networks near 0.99 mean cosine; measured **0.999**, effective rank
  1.0 of 2048. Training *reduces* anisotropy (0.999 → 0.332).
- **Godey et al. (EACL 2024) inverted.** Anisotropy was predicted to track
  self-attention, so ViTs bad and CNNs fine. Here the ViT is the best-behaved
  trained encoder and the worst is a CNN (YOLOv8n). What predicts it is the
  *training objective*, not the architecture.

### 6. The memory bounds odometric drift

A Kalman-style filter where both steps are native — predict is one bind (the FPE
homomorphism), update is one bundle:

| odometry noise | dead reckoning | best filtered |
|---|---|---|
| σ=0.02 | 0.743 m | **0.409 m** |
| σ=0.10 | 3.803 m | **0.627 m** |
| σ=0.40 | 15.212 m | **0.627 m** |

Odometry degrades 20×; the fused estimate does not move. The honest claim is
*not* "the VSA localises better than odometry" (it does not) but "**a fixed-size
associative memory caps unbounded drift at O(D) per frame**". Physically
impossible jumps fall from 242 to 4.

---

## Corrections — do not repeat these

**Never quote a bare whitening multiplier.** The public 365× and 22× figures
were measured on YOLOv8n's penultimate detection feature, the most anisotropic
representation available. It is 8.1× on DINOv2. Quote the encoder with it.

**A random train/test split is invalid at 15 fps.** Neighbours of a held-out
frame stay in the memory, so the oracle drops to 0.01 m, everything scores ~95%,
and an **untrained** network ties DINOv2. Use `--modes blocked` (contiguous
held-out segments plus an eviction radius) or `contiguous`. Note that blocking
by scattering *individual* frames does not work either — at 30% held out the
mean spacing between queries is ~3 frames, so any useful eviction radius deletes
the entire memory.

**Always compute a chance baseline, and a constant-predictor baseline.** In a
6.9 m room, chance was 5.24 m and the ceiling 2.02 m. Worse: scored against "a
random stored frame", raw features gave a suspiciously uniform 73.1% across all
four encoders — because they decode **every query to one grid cell**. A constant
predictor beats a random frame in a small room. Against a proper centroid
baseline those rows are negative.

**Closed-set recall barely tests descriptor quality.** An untrained ResNet-50
matched DINOv2's median (0.160 vs 0.156 m). Storing "arbitrary vector ↔ pose"
and querying with that same vector works whether or not the vector means
anything. Encoder differences only appear once leakage is removed: blocked
split gives dinov2 70.2 > resnet50 66.0 > yolov8n 59.6 > untrained 56.9.

**`school_run1`'s poses are unusable.** 23 km of path in 455 s, 874 frame-to-frame
jumps over 5 m, peak implied speed 1,595 m/s against a Spot's ~1.6. Image-only
analyses on it are fine and have been used; anything binding position is void.
`sequences validate` now rejects it automatically.

**The deployed footprint is not 512 KB.** The traces are, but the decoder grids
for the configuration on the demo page are 414 MB, and total map state 420 MB.
Decode is ~91% of query time.

**Superseded by later work:** the suggestion to cut the language pipeline from
128 PCA components to 32 (entry y). The answer is neither — centre or z-score.

---

## Open questions

1. **Does any of this survive a bigger backbone?** Everything used
   `dinov2:small`. Does the encoder-independent whitening floor hold for
   `base`/`large`/`giant`? Needs a GPU; publishable either way.
2. **Does it survive a second environment?** Every number is one room, one walk.
   `school_run1` was meant to be the scale-up and its poses are dead. Pose-free
   analyses can use almost any image sequence.
3. **Scored through the memory, does the learned (SIGReg) projection beat
   z-scoring?** The chain is built and unmeasured — see
   [ADVANCED_SIGREG_VSA.md](ADVANCED_SIGREG_VSA.md).
4. **Is the crosstalk multiplier mostly a mean-removal effect?** Given finding
   2, the 19.9×/8.1× figures may largely be centring in disguise. Not yet
   separated.
5. **Instance identity is absent.** `chair` has 23 instances; a single argmax
   over a multi-instance class has no well-posed answer, and this causes the
   headline language "miss" on the demo page.
6. **Single-seed sensitivity.** The Kalman numbers changed 2× on a different
   noise draw. Average over seeds before publishing.
