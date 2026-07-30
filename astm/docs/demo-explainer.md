---
title: Live VSA Demo Explainer — What the Classroom Demo Shows and Why
type: analysis
status: active
created: 2026-07-29
updated: 2026-07-30
source_paths:
  - wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md
  - wiki/experiments/2026-07-29-vsa-cognitive-map-classroom-results.md
  - vsa_cognitive_mapping/README.md
  - vsa_cognitive_mapping/workshop_demo.py
  - external/VSACognitiveMapping-init-am/docs/classroom_associative_memory_math.md
tags: [demo, explainer, vsa, fhrr, cognitive-map, classroom, astm, presentation]
---

## Summary

The live demo (`vsa_cognitive_mapping/workshop_demo.py`, port 8020) replays a
Boston Dynamics Spot walk through a classroom (2,478 RGB frames, 496 LIO-SAM
poses, ~7×6.5 m room) and answers spatial-memory questions — *where have I
seen this view? what did I see at that spot? where were the chairs?* — not
from a database, but from a handful of **single fixed-size vectors** (8,192
complex numbers each). Every detection the robot ever made is superimposed
into those vectors with a Vector Symbolic Architecture (FHRR phasor algebra),
and every query is a couple of element-wise vector operations followed by a
similarity sweep — milliseconds on CPU, no search structure, no growth in
memory as experience accumulates. The demo drives the *actual* Telluride
workshop code (`external/VSACognitiveMapping-init-am`) against a memory built
from our own Spot dataset, so it doubles as a cross-implementation validation
of the numbers going into the ASTM paper
([paper plan](2026-07-29-vsa-query-layer-paper-plan.md)).

## The 60-second version

Think of each 8,192-dimensional vector as a strip of tape holding 8,192 tiny
clock hands, each frozen at some angle.

- **Bind (⊗)** two vectors = add their clock angles position-by-position.
  Like stacking two tinted transparencies: the stack resembles *neither*
  sheet — it looks like noise — but nothing is lost.
- **Bundle (Σ, then average)** = record many such stacks onto **one** strip,
  like a multiple-exposure photograph or many voices on one tape. The strip
  never gets longer; the recordings just get slightly noisier as more go in.
- **Unbind (⊘)** = subtract the angles of a factor you know. Holding the
  exact complementary filter up to the stack makes whatever was bound with
  that factor snap back into focus, while every other recording stays
  incoherent noise.

So the robot stores `what-it-saw ⊗ where-it-was` for every moment, summed
into one vector. Later, unbinding by "where" recovers "what", and unbinding
by "what" recovers "where" — the same memory answers both directions. The
price is that recall is *approximate*: signal-to-noise falls roughly as
√(D/K) for K items in D dimensions, which is exactly the capacity trade the
demo lets you see.

## The pipeline, stage by stage

1. **YOLO detections + embeddings.** YOLOv8n runs on every D455 RGB frame:
   detection boxes (8,878 detections over 2,478 frames on the workshop run)
   plus one 256-d penultimate-layer feature vector per frame. Pure computer
   vision — no VSA yet.

2. **PCA whitening — the step that makes everything work.** Raw YOLO
   embeddings on this data are pathologically *anisotropic*: pairwise cosine
   similarity sits in a 0.76–1.00 band (mean 0.88), effective rank 8.1 of
   256, with a mean vector carrying 15.7× the residual energy. Every frame
   "looks like" every other frame, so when hundreds of them are bundled the
   shared component adds **coherently** — crosstalk ≈ 365× the signal, and
   recall is destroyed. Whitening (subtract the mean, decorrelate, scale to
   unit variance per component) drops off-diagonal |mean| similarity to
   0.067 and crosstalk to ≈ 27×; ~90% of the win is mean-removal alone.
   Concretely: the heading demo went from 28% to **98%** correct within 20°.
   This isotropy ⇔ capacity link is the scientific core of the ASTM paper —
   both prior VSA lineages used random (isotropic-by-construction) atoms;
   real learned features break exactly that assumption.

3. **Random projection to 8,192-d phasors.** The (whitened) 256-d embedding
   is pushed through a fixed Gaussian projection to (I,Q) pairs, normalised
   onto the unit circle, magnitude discarded — leaving a unit-modulus complex
   "content" vector. Neighbourhood structure survives the projection;
   unit-modulus is what makes unbinding exact (dividing by a phasor is its
   exact inverse).

4. **FPE context encoding.** Continuous quantities become vectors via
   fractional power encoding, `B^v` with phases `e^(i v φ)`:
   - **Position**: `ctx_pos(x,y) = Bx^(x/ℓ) ⊗ By^(y/ℓ)` — nearby places get
     similar vectors, with the length-scale ℓ setting the similarity kernel
     width.
   - **Heading**: yaw wraps at ±π, so its base uses random *integer*
     frequencies |k| ≤ 3, making every component exactly 2π-periodic
     (`ctx(ψ+2π) = ctx(ψ)`).
   - **Time**: one base over capture order.

5. **Memories.** The classroom design keeps **three per-axis memories**, so
   each only discriminates along its own axis:
   `M_axis = mean_n content_n ⊗ ctx_axis_n` for position / heading / time.
   Alongside them, the ASTM **multi-trace set** (`astm_traces.py`) bundles
   class ⊗ place, class ⊗ time, place ⊗ time, and the full
   class ⊗ place ⊗ time event trace — one trace per query shape, because
   *under-unbinding* a triple-bound trace (supplying only one factor) leaves
   the other factor bound in and attenuates the answer **22×** (0.247 vs
   0.011 measured). Marginal traces are the high-SNR pathways; the event
   trace answers the fully conditional questions.

6. **Queries = unbind + sweep.** Every question is: unbind the relevant
   memory by the factor(s) you know, then sweep the unknown axis — a dense
   (x,y) grid for place, a 181-angle sweep for heading, the stored codebook
   for time, the class vocabulary for "what" — and read out cosine
   similarity. Ambiguity is visible (a multi-modal heatmap) rather than
   hidden behind an argmax.

## What each demo panel shows

- **Replay / reverse-query panel.** Scrub or auto-play the 1,652 *held-out*
  frames (the memory stored only every 3rd frame — 826 frames — so the query
  content was never bundled in). Each tick unbinds all three memories by the
  current frame's content: the **position heatmap** answers "where have I
  seen something like this?" (peak marker vs the true pose marker), the
  **polar plot** answers "facing which way?", and the **time readout**
  recalls the nearest stored moment (typically 0.2–0.8 s from truth).
  ~10 ms per tick server-side (was 800 ms before the decoder matrices were
  precomputed — an 80× hot-path fix, same math). Watch for genuinely
  ambiguous views: two similar-looking spots produce two warm blobs, which
  is the honest answer.
- **Class → where dropdown.** Pick one of 21 classes ("chair", "tv", …).
  The stored frames containing that class are bundled, phase-normalised into
  a unit-modulus probe, and used to unbind the position memory (~7 ms). A
  mode select picks the semantic memory: **vantage** lights up the
  observation footprint (where the class was *seen from*), **object** binds
  to depth-derived object positions (where the class *is* — chair peak
  0.086 m from the largest furniture cluster vs 2.11 m for vantage), and
  **now** queries the λ-decayed working memory (fresh sightings 0.12–0.51 m,
  honest "forgotten" badge when stale).
- **Forward click-query.** Click anywhere on the map: the point is FPE-
  encoded, unbound from the position memory, and cleaned up against the
  stored codebook → top-k recalled frames with thumbnails and their
  detections. Verified case: clicking a trajectory point recalled a stored
  frame 2 cm away.
- **ASTM spatio-temporal panel** (multi-trace queries, live on the same
  page). Give any two of {what, where, when} and the router unbinds the
  matching trace to decode the third — "where was the chair at t=150?",
  "when was something here?", "what was at this spot in this window?".
  **Temporal range kernels** make a time *range* cost the same as a time
  *point* (the kernel `T_(a,b) = ∫ T^t dt` is closed-form per frequency),
  and small discrete unknowns fall back to a vocabulary scan. Bench numbers
  below.

## Measured numbers (all from this repo's runs — sources cited)

| Quantity | Value | Source |
|---|---|---|
| Dataset | 2,478 RGB frames, 496 LIO-SAM poses, ~7×6.5 m room | classroom results page |
| Raw YOLO embedding similarity | cos 0.76–1.00, mean 0.88; eff. rank 8.1/256 | classroom results page |
| Coherent crosstalk, raw → whitened | ≈365× → ≈27× signal (off-diag \|mean\| 0.88 → 0.067) | classroom results page |
| Heading demo, raw → whitened | 28% → 98% within 20° | classroom results page |
| Our build recall (413 frames, hd 8192) | position L1 0.14 m, heading 0.13 rad, time 3.7 frames | classroom results page |
| Workshop build recall (826 stored, stride 3) | position L1 **0.183 m**, heading **0.122 rad** | paper plan log (c) — their code, our data |
| Time-conditioned "where was chair at t" | 8–11 cm vs robot truth | classroom results page |
| Under-unbinding attenuation | 22× (0.247 vs 0.011) | classroom results page |
| ASTM P0 bench (4,498 events, 38 classes, 4 traces) | marginals 0.05–0.22 m / 11–13 frames; conjunctive point 0.09–0.15 m | paper plan log (b) |
| Point-query latency | 3–16 ms CPU; range-kernel grid decode 168 → **14.5 ms** after the fast-decode rewrite | paper plan log (b, g) |
| Calibrated abstention | z≥3 ≙ sim≈0.0037; 93% confident-correct, 80% abstention precision; AUROC z 0.872 ≈ sim 0.878 | paper plan log (g, l) |
| Memory footprint honesty | traces 512 KB; 420 MB total incl. decoder grids at hd 8192; D×N sweep Pareto knee D=4096 = 118 MB | paper plan log (b, j) |
| Rare-class bias fix | conf-weighted LOW-tercile err 5.15 m → **0.12 m** with log class-weighting; frequent classes unharmed | paper plan log (n) |
| Object-position mode | chair where-query peak 0.086 m from largest furniture cluster (vantage build: 2.11 m) | paper plan log (n) |
| Online whitening (causal Welford) | near-oracle by final third: 0.146 m vs oracle 0.090 m grid-decode L1 | paper plan log (k) |
| Synthetic moved object | which_moved ranks tv #1; 1.71 m recovered vs 1.68 m injected; control 0.38 m | paper plan log (l) |
| Language → VSA (never-stored paraphrases) | direct latent unbind median 0.17 m (12/15 < 1 m); soft grounding median 0.11 m | paper plan log (o) |
| Demo latency | reverse query 800 ms → 10 ms (80×); class query 7 ms | paper plan log (d) |
| le-marmotte VGPI | 0.13 m SE(2) relocalization, one 2048-d vector (~16 KB map), our Spot data | paper plan log (d) |

Cross-validation point: the workshop's own `evaluate` on their build
independently reproduces our July numbers — two implementations, same data,
same numbers.

## Language queries (new, 2026-07-30)

The demo story now extends to free text (`language_query.py`): MiniLM
sentence latents are whitened over a ~180-noun vocabulary — the same
isotropy medicine applied to the language modality — then projected to
phasors, and the memory is rebuilt with **text-derived class atoms** over
object-position events (log class-weighting). Two mechanisms, benchmarked on
15 paraphrases never stored in the memory:

- **Soft grounding** (text-similarity routing to class atoms): 10/15 top-1
  grounding, median position error 0.11 m.
- **Direct latent unbinding** (the free-text latent itself as the unbinding
  key — no codebook routing): median 0.17 m, 12/15 within 1 m —
  "luggage"→suitcase 0.03 m, "cold storage for food"→fridge 0.24 m,
  "cutting tool for paper"→scissors 0.11 m.

Failures are LM-semantic (ambiguous phrases like "something to sit on"
spread over couch/table/chair), not VSA-mechanical, and the two mechanisms
fail on *different* queries — they are complementary. Figure:
`outputs/classroom/language_query.png`.

## Honest limitations — updated 2026-07-30

The 2026-07-29 version of this section listed six limitations. Most have
since been measured or fixed; the ledger:

**Since measured/fixed (2026-07-30):**

- *Whitening was offline* → causal Welford whitening converges to
  near-oracle recall by the final third of the walk (0.146 vs 0.090 m);
  a frozen calibration lap is insufficient at any K (1.15–1.73 m), EMA is
  worse everywhere, and the design law is **store codes as encoded, never
  re-encode** (causal codes stay ~orthogonal to final-stats re-encodings,
  sim 0.29). Paper plan log (k).
- *Frequency bias* → manifests as position error (crosstalk hijacks
  rare-class peaks), not abstention; **log class-weighting fixes it** —
  rare-tercile mean error 5.15 m → 0.12 m, frequent classes unharmed
  (0.05–0.22 m). Paper plan log (n).
- *Similarities were uncalibrated* → per-decode-type null z-scores (z≥3 ≙
  sim≈0.0037; 93% confident-correct, 80% abstention precision); AUROC z
  0.872 ≈ raw sim 0.878 (within a decode type, z is a monotone affine of
  sim — its value is the interpretable threshold, not extra
  discrimination); risk–coverage 42% coverage at 29% selective risk;
  range-kernel queries need **shape-matched nulls** (point-probe nulls do
  not transfer). Paper plan log (g, l).
- *"Which object moved?" unproven* → demonstrated synthetically: relocating
  all second-half tv events by 1.68 m, gated which_moved ranks tv #1 and
  measures 1.71 m; the control run is clean (0.38 m); the timeless marginal
  is bimodal and misleading — the event trace is necessary. Paper plan
  log (l).
- *Robot pose, not object position* → `--place-mode object` binds classes
  to depth-derived object positions (2.1% fallback); chair where-query
  peak lands 0.086 m from the largest furniture cluster vs 2.11 m for the
  vantage build. Paper plan log (n).
- *Detector garbage faithfully mapped* → hygiene filters
  (`--min-class-events`, `--class-blocklist`) drop junk classes at the
  canonical-stream boundary. Paper plan log (n).

**Still open:**

- Staged own-data collection with position ground truth (required —
  school_run1's dataset odometry is diverged).
- Baselines B–D (temporal graph, LLM-compiler, VLM) + ReMEmbR.
- Online whitening *deployment* integration — the eval exists; the
  pipeline flag does not.
- Instance identity (tracks as atoms; needs a tracker front-end).
- Multimodal-time mode-picking (the dominant residual decode error).
- Real multi-loop robustness — the moved-object result is synthetic; the
  physical data is still one loop.

## Three lineages, one substrate

The demo sits at the junction of three in-house Telluride lineages, all
speaking the same FHRR phasor algebra:

1. **init-am** (this demo's engine) — *episodic* associative memory: bind
   frame content to position/heading/time, recall either direction.
2. **GC-VSA** (Krausse, Neftci, Sommer, Renner — IJCNN 2025) —
   what ⊗ where ⊗ when factorised event memory with algebraic queries,
   demonstrated on synthetic data; the ASTM multi-trace panel is its ideas
   run on real robot data.
3. **le-marmotte VGPI** — the complementary direction: *localization* from
   lidar in ONE 2048-d phasor vector (~16 KB map), 0.13 m SE(2)
   relocalization on this same Spot dataset, with FPGA precision-reduction
   branches.

One lineage remembers, one queries, one localizes — same bind/bundle/unbind
operations, same fixed-dimensional vectors. The thesis in one sentence:
**VSAs provide the algebraic composition layer for embodied spatial
cognition — perception, mapping, episodic memory, and localization as
operations in a single fixed-dimensional vector space** — and the first
joint real-world deployment of these threads is the ASTM paper's
unification story.

## Related Pages

- [ASTM paper plan v2 (live tracker)](2026-07-29-vsa-query-layer-paper-plan.md)
- [Classroom results (measured numbers)](../experiments/2026-07-29-vsa-cognitive-map-classroom-results.md)
- [VSA cognitive-map presentation outline](2026-05-17-vsa-cognitive-map-presentation-outline.md)
- [SIGReg VSA reframe (isotropy ⇔ atom quality)](2026-05-11-sigreg-vsa-reframe.md)
- [le-marmotte VGPI source page](../sources/2026-07-30-le-marmotte-vgpi.md) — the third lineage's repo survey (one-vector lidar localization, static-demo architecture)
- Figures: `outputs/classroom/evaluate.png`, `outputs/classroom/event_binding_demo.png`, `outputs/classroom/semantic_query_chair.png`
- Screen-share version: `outputs/classroom/demo_explainer.html`

## Open Questions

- Does the explainer need a recorded fallback (gif) in case the live server
  misbehaves during the meeting?
- A two-loop experiment (`vsa_cognitive_mapping/two_loop_experiment.py`) is
  in progress — fold its result into the "real multi-loop robustness" line
  when it lands.
