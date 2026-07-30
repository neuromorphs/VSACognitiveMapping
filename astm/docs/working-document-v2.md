---
title: VSA-Based Cognitive Maps — Working Document v2
type: analysis
status: active
created: 2026-07-30
updated: 2026-07-30
source_paths:
  - wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md
  - wiki/analysis/2026-07-29-demo-explainer.md
  - wiki/sources/2026-07-30-le-marmotte-vgpi.md
tags: [working-doc, vsa, cognitive-map, project-brief, roadmap]
---

# VSA-Based Cognitive Maps — Working Document v2

*Successor to the original project brief (Telluride era). Same structure,
updated with what has actually been built and measured as of 2026-07-30.
The companion tracker (deadlines, work packages, paper contributions) is
[the ASTM paper plan](2026-07-29-vsa-query-layer-paper-plan.md); this
document is the project-level view.*

## Main Goal

Demonstrate that VSA-based cognitive maps can be paired with full-scale
vision encoder embeddings to identify, store, and retrieve important
locations in an environment — a spatial associative memory connecting
*what the robot sees* with *where it saw it*.

**Status: demonstrated on real robot data.** All five objectives of the
original demonstration plan are met on the Spot classroom walk-through
(2,478 frames, D455 RGB + LIO-SAM poses), with live interactive queries
and measured error bars.

## Demonstration objectives — original list, scored

| # | Objective (original wording) | Status | Evidence |
|---|---|---|---|
| 1 | Explore with camera + SLAM poses | ✅ | Spot classroom + school_run1 datasets through the config-driven pipeline |
| 2 | Extract latent embeddings from a pretrained encoder | ✅ | YOLOv8n penultimate features (256-d); DINOv2 in the workshop jepa lineage |
| 3 | Store in a VSA spatial associative memory | ✅ | FHRR phasor traces at hd=8192; multi-trace design (see below) |
| 4 | Query with visual embeddings or object-level prompts | ✅ | Both: content→place (reverse replay queries) and class-prompt→place (semantic memories) |
| 5 | Retrieve locations of previously observed objects | ✅ | Heatmaps + ranked lists, live at ~2–15 ms/query; chair localized to 0.13–0.34 m of ground-truth furniture clusters |

The "key result" the brief asked for — *a heatmap or ranked list of
locations associated with a query object* — is exactly what the live demo
(`workshop_demo.py`, port 8020) and the shareable static export
(`share/classroom_demo/`) render.

## Architecture — as actually built

The brief's three components survived contact with reality, with one
crucial addition (isotropization) and one structural change (multiple
traces instead of a single M):

```
Camera image ──► Vision encoder (YOLO/DINO) ──► z_t (256-d)
                                                  │
                              PCA WHITENING ◄─────┘   ← the load-bearing step
                                                  │      (see finding #1)
                                                  ▼
SLAM (LIO-SAM) ──► pose p_t ──► FPE position code S(x,y)
                                                  │
                              random phasor projection → content V_t
                                                  ▼
              MULTI-TRACE COGNITIVE MAP (all hd = 8192)
              ├─ M_what_where  = Σ conf·C ⊗ S      "where are chairs, ever"
              ├─ M_what_when   = Σ conf·C ⊗ T      "when were chairs seen"
              ├─ M_where_when  = Σ conf·S ⊗ T      occupancy history
              ├─ M_event       = Σ conf·C ⊗ S ⊗ T  "where was X at time t"
              ├─ episodic (workshop): content ⊗ S/H/T   "have I been here?"
              └─ M_now (λ-decay, optional)          "where is the person NOW"
                                                  │
Query (class prompt / embedding / pose / time) ──► unbind + similarity sweep
                                                  ▼
              heatmap / time-curve / ranked classes + CALIBRATED CONFIDENCE
```

Query side, new: free-text queries via `language_query.py` — whitened MiniLM
latents as unbinding keys (direct latent unbind, median 0.17 m on
never-stored paraphrases).

Design rule (measured, 22× attenuation basis): never store more factors in
a trace than its target query can supply — the marginal traces are the
high-SNR paths; the event trace serves conditional queries only.

## Original open questions — now answered

1. **"How should dense vision embeddings be converted into VSA
   representations?"** → *Answered, two parts.* (a) The projection itself is
   easy (random Gaussian → unit phasors); (b) the hard part is that raw
   encoder embeddings are catastrophically anisotropic (pairwise cosine
   0.76–1.00, effective rank 8/256) → 365× coherent crosstalk → **PCA
   whitening first** drops it to 27× and takes heading recall from 28% to
   98%. This isotropy ⇔ capacity bridge is the paper's core scientific
   claim — narrowed post-verification: anisotropy + whitening is established
   for language embeddings (Mu & Viswanath 2018; Ethayarajh 2019; Su et al.
   2021) and learned HRR vectors (Ganesan et al. NeurIPS 2021); ours is the
   quantified anisotropy → associative-recall-collapse bridge measured on a
   deployed robot, plus the causal online estimator. *Now also measured online:* causal Welford whitening converges to
   near-oracle recall by the final third of the walk (0.146 vs 0.090 m); a
   frozen calibration lap is insufficient at any K and EMA is worse — and a
   new design law fell out: **store codes as encoded, never re-encode stored
   content** (causal codes stay ~orthogonal to final-stats re-encodings
   because degenerate eigen-subspaces keep rotating). *Still open:* learned
   isotropic encoders — the brief's
   [clifford-vae / HRR-VAE idea](https://github.com/momalekabid/clifford-vae)
   maps exactly onto the isotropization ladder's "learned projection /
   isotropy regularizer" rungs (and onto the SIGReg lineage) — the natural
   next experiment beyond closed-form whitening.
2. **"Full image embeddings, object-level embeddings, or both?"** → *Both,
   for different queries.* Whole-frame content → episodic "have I been
   here"; per-detection class atoms (+ depth-derived object positions,
   `--place-mode object`) → "where IS the object". Both live in the demo.
3. **"How should repeated observations be merged?"** → *Confidence-weighted
   bundling* for permanent memories; for dynamic objects, **per-class
   λ-decay working memory** (person 0.9, furniture 0.995). The
   decaying-trace mechanism follows Frady, Kleyko & Sommer (2018); our
   contributions are two measured engineering findings: rare-class mass
   imbalance requires per-class normalization, and normalization without an
   evidence floor silently cancels decay. Convergent with Krausse's distance-driven washout knob in
   le-marmotte (independent, both mid-July) — align terminology with him.
   *Extended (hybrid clocks):* the forgetting clock is now selectable —
   `--decay-per {event, frame, metre, both}` with
   λ_eff = λ_time^Δt_frames × λ_dist^Δd_metres, clocks counting from each
   class's last write. Demonstrated at a real stationary window in the walk:
   metre keeps weight 14.8 where frame forgets to 0.001 — a paused robot
   keeps its memory under the metre clock. Caveats: 'both' as a product is
   AND-forgetting (either clock advancing fades the memory), not an
   interpolation; the metre clock slightly smears fresh fields.
4. **"How sensitive is retrieval to SLAM drift?"** → *Brutally, and we have
   the case study:* school_run1's diverged LIO-SAM odometry (km-scale)
   makes position mapping impossible while heading/time survive. The map
   is only as good as its pose input — which motivates the VGPI
   (le-marmotte) integration: recover poses with a VSA localizer, then
   bind semantics to them. Loop-closure re-anchoring remains untested.
5. **"Best way to decode locations from the memory?"** → *Unbind + dense
   similarity sweep over an FPE grid*, precomputed decoder matrices
   (2–15 ms), **plus calibrated abstention**: null-distribution z-scores
   (z≥3), 93% confident-correct, 80% abstention precision — the memory
   knows when it doesn't know. *Proper metrics now measured* (544-query
   labeled battery): AUROC z 0.872 ≈ raw sim 0.878 — within a decode type
   z is a monotone affine of sim, so the z-score's value is the
   interpretable threshold, not extra discrimination; risk–coverage at z≥3:
   42% coverage at 29% selective risk (vs 62% error at full coverage). One
   trap found: range-kernel queries need **shape-matched nulls** — the
   stored point-probe null does not transfer to range kernels.
6. **"How does this compare to a conventional database?"** → *Measured,
   honestly:* the exact event table wins on raw accuracy at 4.5k events (as
   predicted); the VSA claim is the accuracy–latency–memory frontier at
   scale — D×N sweep gives Pareto knee at D=4096: 118 MB honest map state,
   0.80 battery accuracy, 2.7 ms median query, no capacity cliff at full
   stream. Baselines B–D (temporal graph, LLM-compiler, VLM) still to run.
7. **"What downstream tasks benefit most?"** → *Sharpened by the JEPA
   probe:* algebraic transport is exact, so prediction inherits content
   persistence — the memory's unique value is **query** (conditional
   what/where/when, range windows, which-moved) and **honest recall**, not
   next-frame prediction; a world model must learn appearance dynamics only.

## New open questions (v2, revised 2026-07-30)

- Staged moved-object collection: protocol, annotation, surveyed ground truth
  (REQUIRED for the scaling claim — next action #1 in the tracker; the
  synthetic moved-object case is now measured, the real one is not).
- Online whitening *deployment integration* — the eval is measured
  (near-oracle by the final third); the pipeline flag does not exist yet.
- Per-class clock assignment design: which classes get which forgetting
  clock (event / frame / metre / both) and at what rates — currently
  hand-set; needs a principled assignment.
- Range-null calibration integration into the query router: shape-matched
  nulls for range-kernel queries are measured as necessary but not yet
  wired into the router's stored null stats.
- LM-query ambiguity handling: language-query failures are LM-semantic
  ("something to sit on" spreads over couch/table/chair) — mixture-aware
  decoding or clarification, not a VSA fix.
- Instance identity (tracks as atoms: C ⊗ I ⊗ S ⊗ T) — SuperMap-style
  persistent IDs are the graph world's strength; our vocabulary-scan answer
  needs a tracker front-end.
- Multimodal-time decoding (mode-picking is the dominant residual error —
  not kernel width; multi-scale time measured as no-dominant-winner).
- HRR-VAE / learned isotropic encoder (see Q1).
- Loop-closure correction of an already-written map (re-anchoring bound
  events after pose-graph updates).

## Milestones — original list, updated

- ✅ Decide on vision architecture → YOLOv8n features now; DINOv2 in jepa
  lineage; HRR-VAE as the learned upgrade path
- ✅ Decide on dataset → Spot Telluride classroom (primary), school_run1
  (scale; position blocked by dataset odometry), staged own collection next
- ✅ Build VSA encoding + associative memory system → `vsa_cognitive_mapping/`
  (core, pipelines, ASTM traces, calibration, sweep harness)
- ✅ Build preliminary demo experiment → live 4-panel demo + static share
  export + three explainer documents
- ⬜ **Deploy on a robot** → the remaining original milestone; path: staged
  collection with surveyed GT → (optionally VGPI-recovered poses) → onboard
  pipeline (all CPU-capable already)

## Long-Term Vision (unchanged, now with footholds)

The example queries from the brief are no longer hypothetical: "where did I
see the missing tool?" is `query_class`/object-mode; "which hallway
contained the warning sign?" is a conditional event-trace query; "have I
visited a place like this before?" is the episodic reverse query running
live at 15 ms. The bridge claim — VSA memory connecting geometric maps,
semantic perception, and robot memory — now has three in-house lineages to
unify (init-am *remembers*, GC-VSA *queries*, le-marmotte *localizes*) and
one substrate under all of them.

## Related pages

- [ASTM paper plan / tracker](2026-07-29-vsa-query-layer-paper-plan.md) —
  deadlines, work packages, contributions, progress log a–o
- [Demo explainer](2026-07-29-demo-explainer.md) · code explainer:
  `outputs/classroom/code_explainer.html`
- [le-marmotte VGPI source page](../sources/2026-07-30-le-marmotte-vgpi.md)
