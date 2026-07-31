---
title: Code Provenance and Grounding Audit — vsa_cognitive_mapping/
type: analysis
status: active
created: 2026-07-30
updated: 2026-07-30
source_paths:
  - vsa_cognitive_mapping/
  - external/VSACognitiveMapping-init-am/
  - external/VSACognitiveMapping-jepa/
  - wiki/analysis/2026-07-29-vsa-query-layer-paper-plan.md
  - wiki/experiments/2026-07-29-vsa-cognitive-map-classroom-results.md
  - raw/2026-07-29-astm-fact-check.md
tags: [audit, provenance, grounding, astm, vsa, cognitive-map, code-review, icra-2027]
---

## Summary

**9,502 lines under audit in `vsa_cognitive_mapping/` (20 `.py` files); 7,059 lines of
workshop code in `external/` (two branches of Shay Snyder's and Sven Krausse's
Telluride repo).** Both directories are **untracked in git** — the entire evidence base
for the ICRA submission exists only on one disk, and no "unchanged from before" claim
about any file in either tree is currently falsifiable.

**No file lies about where it came from.** Three independent analysts tested every
provenance docstring and every one held. `vsa.py` diffs against
`external/VSACognitiveMapping-jepa/src/vsa_cognitive_mapping/vsa.py` to **two hunks and
zero changed executable lines** (a replaced docstring paragraph and
`from __future__ import annotations`, which is genuinely required — `vsa.py:44/63/83/180`
use PEP-604 hints that raise `TypeError` at def-time on Python 3.9). `test_vsa.py`'s only
in-body edits are five `pytest.approx` → `approx` renames with every expression, target
and tolerance unchanged; 18/18 pass. `grep` for `null_calib|which_moved|QueryRouter|
abstain` across all of `external/` returns **0 hits**, so the multi-trace architecture,
the range kernels, the calibration and the router are correctly labelled novel-mine. A
fabrication audit finds nothing here.

**The finding is grounding, not fabrication.** A scan of all 20 files for every author in
the verified-literature list plus `arxiv|doi` returns **exactly one real external
citation in 9,502 lines** — `language_query.py:29`, naming VLMaps / CLIP-Fields / NLMap.
(The only two other regex hits are false positives: "tem*plate*s" in `astm_sweep.py:137`
and "glass *plate* bowl" in `language_query.py:59`.) Zero files cite Plate, Frady,
Kleyko, Komer, Krausse, Snyder, Mu & Viswanath, or Ethayarajh. Every "why" pointer in
`astm_traces.py` is internal — "fact-check Round 4", "tracker entry (r)", "FIX 1/2/4".
The attributions exist; they live in the tracker and in `results_robotics.html`, and were
never carried back into the source. Two techniques have **no external citation available
anywhere in the project**: calibrated abstention (contribution 3's entire basis) and the
`conf/√W_c` / `conf·log(1+K)/K` class-weighting schemes.

**Where the lines sit.** 5,499 lines (58%) are load-bearing across 9 files; 3,028 (32%)
supporting; 477 (5%) scope-creep; 498 (5%) dead. Sub-file analysis moves roughly 1,000
more lines into redundant/scope-creep — `compare_clocks` (180), `object_map`'s second web
server (~324), `two_loop`'s tests 3–4 (~490), `workshop_demo`'s re-implemented router
(59). Presentation is a real cost centre: `workshop_demo.py`'s embedded `PAGE` string is
lines 784–1145, **362 of 1,173 lines = 30.9%**, and ~46% of the file exists to shape JSON
for it.

**The module's strongest quality signal is self-refutation.** `crosstalk_scaling.py`
defines the public 365×/27× figure in code and reports it reproduces under *no* definition
tried (whitened slope +0.86, not the predicted 0.5). `crossover_analysis.py` prints in
plain English that an indexed exact table is never beaten on latency. `online_whitening_eval.py`
shows a frozen calibration lap fails at every K. `calibration_eval.py` states that the
z-score barely beats raw similarity. Four files publish the number that hurts. That is
worth more to a reviewer than any figure in the set.

**Three input claims failed verification and are corrected here.** (1) The static export
is **not** broken — `share/classroom_demo/assets/frames/` exists with 551 JPEGs, 15 MB,
dated Jul 29 23:49. (2) `external/` is **also untracked**, not just `vsa_cognitive_mapping/`.
(3) Theirs is **7,059 lines**, not 2,710; the brief's figure understates the workshop
contribution by 2.6×, and 3,837 of those lines (17 files, all on the `init-am` branch)
carry the one-line Python-3.9 change.

**Verdict: keep the module, cut 48%.** A paper-critical set of **≈4,945 lines** reproduces
every number in the tracker. Two defects block reproduction today and are one-line fixes:
`bench --bias-bench` (`astm_traces.py:1322`, `KeyError('where')`) and `bench --compare`
(`astm_traces.py:1274-1276`, `KeyError('mean')`) both crash under the v2 calibration that
has been default since tracker entry (s) — the `_meta` guard was added at `:949` and
`:1156` and nowhere else. Tracker entries (n) and (g) are therefore **not currently
reproducible** while being presented as measured.

---

## 2. Provenance ledger

Provenance categories: **theirs-verbatim** (Snyder/Krausse code, byte-identical modulo the
py3.9 import) · **mixed** (their design + our code, both present) · **reimplemented-their-design**
(our code, their architecture) · **novel-mine** (ours, no upstream analogue).

### Theirs — `external/`, 7,059 lines, DO NOT claim as ours

| Branch | Lines | Local change | Status |
|---|---:|---|---|
| `VSACognitiveMapping-init-am/` | 4,508 | `from __future__ import annotations` in 17 files (3,837 lines touched) | Verbatim otherwise |
| `VSACognitiveMapping-jepa/` | 2,551 | **none** (0 files carry the future import) | Fully verbatim |

Key upstream files the paper must attribute: `src/vsa_cognitive_mapping/vsa.py` (the
Phasor algebra and FPE core), `scripts/classroom/classroom_associative_memory.py` (844
lines — the four-command classroom pipeline, `--pca-whiten`, physical units, per-axis
memories), `scripts/classroom/detect_and_embed_classroom.py` (328 — the detector/embedder
the shipped `run_demo_pipeline.py` actually invokes), `src/.../diagnostics.py`
(`effective_rank`).

### Ours — `vsa_cognitive_mapping/`, 9,502 lines

| File | Lines | Provenance | Came from |
|---|---:|---|---|
| `vsa.py` | 224 | **theirs-verbatim** | `jepa:src/vsa_cognitive_mapping/vsa.py`. Diff = 2 hunks, **0 changed executable lines**. Snyder/Krausse's work. |
| `test_vsa.py` | 246 | **mixed** | Their `jepa:tests/test_vsa.py` (18 `def test_`, both files). Ours: a 30-line `_Approx` shim + 5 `pytest.approx`→`approx` renames. Verified 18/18 pass. |
| `__init__.py` | 31 | novel-mine | All 31 lines local (upstream's `__init__.py` is **0 bytes**). Mirrors upstream module path only. |
| `classroom_pipeline.py` | 1,311 | **mixed** | Explicit reimplementation of `init-am`'s four classroom commands (embed/build/evaluate/demo) on the ported Phasor core. Header is honest. Contains one unheralded *improvement*: interpolated + yaw-unwrapped pose association (`:97-102`) vs their nearest-pose (`external:118-125`). |
| `object_map.py` | 504 | novel-mine | Design is textbook FPE spatial binding; no upstream analogue. Duplicates `classroom_pipeline`'s intrinsics **byte-for-byte** (`:50-52,69-79` vs `:68-70,222-233`) in a file that already imports from it (`:44-46`). |
| `astm_traces.py` | 1,480 | novel-mine | **This file is contribution 1.** Nothing in `external/` has a multi-trace, range kernel, calibration or router (verified: 0 grep hits). |
| `astm_sweep.py` | 667 | novel-mine | Contribution 4's D×N grid. |
| `calibration_eval.py` | 459 | novel-mine | Contribution 3's only labelled battery (544 queries). |
| `crosstalk_scaling.py` | 260 | novel-mine | Contribution 2's N-scaling law. |
| `online_whitening_eval.py` | 407 | novel-mine | Contribution 2's causal estimator. Whitening *step itself* is workshop prior art — see §3. |
| `crossover_analysis.py` | 327 | novel-mine | Bounded-state-vs-exact-table cost model. |
| `moved_object_synthetic.py` | 364 | novel-mine | Vocab-scan control; shipped its range-matched null back into the engine (credited `astm_traces.py:497-501`). |
| `two_loop_experiment.py` | 612 | novel-mine | Frame-interleaved pseudo-timeline. |
| `language_query.py` | 337 | novel-mine | Text-latent-as-unbinding-key. **The only file in the module with a real citation.** |
| `probe_jepa_predictability.py` | 274 | novel-mine | JEPA→FHRR workstream, not ASTM. |
| `workshop_demo.py` | 1,173 | **mixed** | First two panels genuinely drive *their* functions via a sys.path forcing trick (`:48-50`). Lines 283–689 (407 lines, 35%) drive **ours**; the docstring (`:2-8`) says "every encode/unbind in this server calls *their* functions" and does not disclose this. |
| `run_demo_pipeline.py` | 125 | novel-mine | Orchestrates *their* `detect_and_embed_classroom.py` + *their* `classroom_associative_memory.py` build — not this module's `embed`/`build`. |
| `export_static_demo.py` | 203 | reimplemented-their-design | Exports the *ported workshop* memory; excludes the ASTM router by design. |
| `demo_associative_memory.py` | 219 | reimplemented-their-design | Docstring cites "the upstream `evaluate` command" — **no such command exists on the `jepa` branch it claims to port from** (`jepa:scripts/associative_memory.py:27-30` stops at build-and-save). The referent is `init-am`, extracted 15 days *after* this file's mtime. |
| `visualize.py` | 279 | novel-mine | Same misattribution: "the upstream `demo` command … demo.mp4" (`:3-7`). `jepa` has no demo video and no query path. |
| **Total** | **9,502** | | |

**Provenance defects (2, both minor, both in the 14 Jul day-1 files):** the two dead-island
files attribute to the wrong branch (`jepa` vs `init-am`). Neither misrepresents ours as
theirs or theirs as ours — they misname *which of theirs*. `README.md:191` states the
accurate version ("is new — it is our own minimal driver"); the docstrings should match it.

---

## 3. Research grounding

**Citation reality check:** 1 real external citation in 9,502 lines (`language_query.py:29`).
18 of 20 files have zero. The attributions below mostly *exist* — in the tracker and in
`outputs/classroom/results_robotics.html` — and were never written back into the source.

| File | Technique | Category | Citation |
|---|---|---|---|
| `vsa.py` | FHRR unit-modulus phasors, elementwise binding | established-prior-art | Plate 2003 — **NOT IN FILE** (upstream's omission, inherited) |
| `vsa.py:3-12` | FPE group homomorphism; "bell/sinc-shaped similarity kernel" | established-prior-art | Komer et al. CogSci 2019; Frady et al. 2021 arXiv:2109.03429 — **NOT IN FILE** |
| `test_vsa.py` | bind/unbind inverse; FPE homomorphism; length-scale↔kernel-width | established-prior-art | Plate 2003; Frady et al. 2021 — **NOT IN FILE** |
| `astm_traces.py` (traces) | Marginal traces as query-shaped "secondary indexes" | extends-prior-art | Bundling SNR: Plate 2003 + Frady/Kleyko/Sommer 2018. Contrast vs one-trace+resonator: Krausse et al. NICE 2025 — **NOT IN FILE** |
| `astm_traces.py:450-463,715-717` | Closed-form temporal range kernels | established-prior-art | Frady et al. 2021 (VFA) — the interval integral of an FPE code *is* a VFA representation of the interval indicator. **NOT NOVEL.** **NOT IN FILE** |
| `astm_traces.py:465-648` | Null-conditioned evidence, z-threshold, abstention | **ungrounded** | **NONE EXISTS — needs one.** No selective-prediction / OOD citation anywhere in the project. Nearest available grounding (Frady 2018's *analytic* non-match similarity distribution) is neither cited nor used; the code Monte-Carlos the same quantity. |
| `astm_traces.py:871-889` | `conf/√W_c` and `conf·log(1+K)/K` class weighting | **ungrounded** | **NONE EXISTS — needs one.** No HDC class-imbalance citation in the verified list; `log` is an unattributed re-derivation of IR sublinear TF damping. |
| `astm_traces.py:226-321` | Exact temporal event table (baseline A) | methodological-construction | Invented baseline; limits partly documented |
| `astm_sweep.py` | Capacity-vs-dimension curves | methodological-construction | Frady/Kleyko/Sommer 2018; Thomas/Dasgupta/Rosing JAIR 2021 — **NOT IN FILE**. What it measures is a bespoke 15-query battery, not a literature capacity metric. |
| `calibration_eval.py` | AUROC / ECE / Brier / risk-coverage / Platt | methodological-construction | **NONE EXISTS — needs one** (same gap as abstention). Two limits *are* documented in-code (in-sample Platt; FIX-3 guard band). |
| `classroom_pipeline.py:378` | what⊗where⊗when conjunctive trace | established-prior-art | Krausse et al. NICE 2025 (GC-VSA) — **NOT IN FILE**. The `:360-362` comment correctly explains why no resonator is needed. |
| `classroom_pipeline.py:738-845` | Exponentially decaying per-class traces | established-prior-art | **Frady, Kleyko & Sommer 2018** (decaying traces / working memory) — **NOT IN FILE**, though tracker (r) records exactly this attribution. Metre-clock washout credited to Krausse's le-marmotte in tracker (m), not in code. |
| `classroom_pipeline.py:304-311` | PCA whitening of content embeddings | **extends-prior-art, mis-attributed** | Mu & Viswanath ICLR 2018; Ethayarajh EMNLP 2019; Su et al. 2021; Ganesan et al. NeurIPS 2021 — **NOT IN FILE**. *And the step is workshop prior art on this exact dataset*: `external/…/classroom_associative_memory.py:778-783` implements `--pca-whiten` with the swamping-direction rationale in its help string, and `diagnostics.py:18-33` already computes `effective_rank`. |
| `classroom_pipeline.py:1018-1194` | Hybrid frame/metre decay clocks | **ungrounded** | Claim is arithmetically forced by unmatched rate constants (0.9/frame vs 0.7/metre), not measured; and computed in robot-vantage mode only (`:1120` hardcodes `xy_by_ts=None`). |
| `crosstalk_scaling.py` | O(N)-coherent vs O(√N)-incoherent crosstalk | extends-prior-art | Frady/Kleyko/Sommer 2018; anisotropy premise Mu & Viswanath 2018 / Ethayarajh 2019 / Ganesan 2021 — **NOT IN FILE** (derived from first principles). |
| `online_whitening_eval.py` | Causal/streaming Welford whitening | extends-prior-art | Same four isotropy citations — **NOT IN FILE**. Docstring `:3-5` calls whitening "the pipeline's load-bearing preprocessing step"; it is **theirs**. |
| `crossover_analysis.py` | Bytes/latency crossover cost model | methodological-construction | None needed. Follows the tracker's "report ALL memory" rule exactly (`:214-221`). |
| `moved_object_synthetic.py` | Synthetic relocation + matched control | methodological-construction | **Model example** — artificiality in the title line, control alongside, non-default `hd=16384` disclosed with its measured justification. |
| `two_loop_experiment.py` | Frame-interleaved pseudo-timeline | methodological-construction | Caveat block repeated 4× and entirely true; but test 4's naming contradicts it (see §5). |
| `object_map.py` | class⊗place map, greedy clustering | established-prior-art | Plate 2003; Komer 2019; Frady 2021; Krausse NICE 2025; **Snyder et al. npj Unconv. Comp. 3:13 2026 (VSA-OGM)** — **NOT IN FILE**. Deeper issue: its clusters are the *reference* for the headline 0.086 m, and the self-consistency caveat lives only in an HTML page written 16 days later. |
| `language_query.py:29-32` | LM latent as unbinding key | extends-prior-art, **incomplete** | VLMaps ICRA 2023 / CLIP-Fields RSS 2023 / NLMap ICRA 2023 — **PRESENT**, but omits the asymmetry that matters: those are open-vocabulary *at the perception level*; this stores 38 closed-set COCO class atoms and is open-vocab at query time only. |
| `probe_jepa_predictability.py` | Derangement content-only null | methodological-construction | Retracts its own prior vacuous null in-code (`:26-32`) — genuine credit. FPE basis uncited. |
| `demo_associative_memory.py` | Synthetic Lissajous bundling demo | **ungrounded** | Reaches for methodological-construction but documents neither of its two material limits (see §5). |
| `visualize.py:135-156` | "bundling capacity: recall vs hypervector width" | **ungrounded** | Renders a capacity claim the underlying computation cannot produce. |
| `workshop_demo.py:140-162` | Class-probe bundle → vantage query | extends-prior-art | Plate 2003; Schlegel/Neubert/Protzel AIRev 2022 — **NOT IN FILE**. Magic cutoff `len(rows) < 5` (`:155-156`) undocumented; no null/abstention at all. |

### Techniques with NO external citation available — must acquire one before submission

1. **Calibrated abstention / null-conditioned evidence** (contribution 3, entire). Needs a
   selective-prediction or OOD-detection citation the project does not have. Also:
   reconcile the Monte-Carlo null against Frady 2018's analytic non-match distribution,
   which the project *does* have and does not use.
2. **HDC class-imbalance weighting** (`conf/√W_c`, `conf·log(1+K)/K`). Needs an HDC
   imbalance citation, or honest labelling as invented + the IR sublinear-TF antecedent.
3. **"VSA secondary index"** — our coinage. Grounded as a *design* (measured 22× under-unbinding
   attenuation, 0.247 vs 0.011) but the term is not in any cited work. Keep the measurement,
   drop or flag the marketing term.

---

## 4. Verdicts

Whole-file primary verdict. Arithmetic sums to 9,502.

| Verdict | Files | Lines | % | Files |
|---|---:|---:|---:|---|
| **load-bearing** | 9 | 5,499 | 57.9% | `astm_traces` 1480, `classroom_pipeline` 1311, `astm_sweep` 667, `calibration_eval` 459, `online_whitening_eval` 407, `moved_object_synthetic` 364, `crossover_analysis` 327, `crosstalk_scaling` 260, `vsa` 224 |
| **supporting** | 7 | 3,028 | 31.9% | `workshop_demo` 1173, `two_loop_experiment` 612, `object_map` 504, `language_query` 337, `test_vsa` 246, `run_demo_pipeline` 125, `__init__` 31 |
| **scope-creep** | 2 | 477 | 5.0% | `probe_jepa_predictability` 274, `export_static_demo` 203 |
| **dead** | 2 | 498 | 5.2% | `visualize` 279, `demo_associative_memory` 219 |
| **redundant** | 0 | 0 | 0% | *(all redundancy is sub-file — see below)* |

### Sub-file reallocation (verified regions)

Whole-file verdicts flatter the module. The honest picture:

| Region | Lines | Reallocate to | Why |
|---|---:|---|---|
| `classroom_pipeline` core (`:63-338`, `_class_seed`, `_localize_detections`) | ~300 | load-bearing | Imported by **7** modules; produced the canonical 1,239-frame / **4,498-detection** stream (verified: `detections.csv` = 4,499 lines incl. header) |
| `classroom_pipeline` `semantic_position`/`semantic_event`/`query_class`/`query_event` | ~170 | **redundant** | Strict subset of `astm_traces`' `M_what_where`/`M_event` + router. Survive only because `workshop_demo:518-546` reads `memory_object.pt` |
| `classroom_pipeline.compare_clocks` (`:1018-1194`) | ~180 | **scope-creep** | One figure, cited by no contribution, claim arithmetically forced |
| `classroom_pipeline` embed/build/evaluate/demo | ~660 | supporting | Parallel second implementation of stages `run_demo_pipeline` actually runs from `external/` |
| `object_map.localize` + `_cluster` | ~180 | supporting | Produces the reference cluster set behind the headline 0.086 m |
| `object_map.serve`/`snapshot`/`ObjectMap`/`_PAGE` | ~324 | **redundant** | Second web server (:8010) offering exactly the two queries `workshop_demo` serves on :8020, in a mutually incompatible encoding (index-ordered atoms, hd=4096, ls=0.6, no calibration), allocating **750 MB** at start-up |
| `workshop_demo.PAGE` (`:784-1145`) | **362** | presentation | **30.9% of the file** (verified) |
| `workshop_demo._astm_extras` (`:344-402`) | 59 | **redundant + drifted** | Re-implements `QueryRouter.query` (`astm_traces:754-817`); uses `enc.ctx_time` where the router uses `ctx_time_vec`, and divides time memory-side where the router's own docstring (`:713-717`) calls that invalid |
| `two_loop_experiment` tests 1–2 | ~120 | supporting | Machinery/unit test; keep as appendix |
| `two_loop_experiment` tests 3–4 | ~490 | **dead / redundant** | Test 3 duplicates `moved_object_synthetic` on a weaker construction; test 4 has no valid baseline |

Net: ≈**1,000 additional lines** are redundant or scope-creep once measured at sub-file
granularity, and the module contains **three mutually incompatible class→place VSA maps**,
**two web servers**, **two renderers**, **two bench batteries**, **two copies of the camera
intrinsics**, and **three copies of the position-field decode**.

---

## 5. The case for and against

### For (defence)

- **Provenance is clean.** Every claim held under `diff`. This is the baseline a
  fabrication audit looks for, and it passed.
- **42–58% of the lines directly produce contribution evidence.** `astm_traces.py` *is*
  contribution 1; `astm_sweep.py` is contribution 4's grid; `calibration_eval.py` is
  contribution 3's only battery; `crosstalk_scaling.py` + `online_whitening_eval.py` are
  contribution 2; `crossover_analysis.py` answers the "bounded state loses to a table"
  attack head-on.
- **Self-refutation across four files.** Publishing the number that hurts is the single
  best signal available to a reviewer, and the module does it unprompted.
- **`moved_object_synthetic.py` shipped a fix back into the engine** — its range-matched
  null is now v2 calibration, credited at `astm_traces.py:497-501`. A control that
  improves the system it controls is not scope-creep.
- **Real engineering merit**, specifically: `_class_seed`'s PYTHONHASHSEED diagnosis
  (`:193-197`); interpolated + unwrapped pose association, which is *better* than the
  workshop's nearest-pose; `workshop_demo`'s complex64 fast decode (measured error 1.2e-8
  vs a top1–top2 gap of ~1.1e-6, an 80× win); `crossover_analysis`'s full memory
  accounting including decoder grids (420 MB reported, not the flattering 512 KB).

### Against (prosecution)

- **Presentation is ~2,000 lines** (`workshop_demo` 1173, `visualize` 279,
  `demo_associative_memory` 219, `export_static_demo` 203, `run_demo_pipeline` 125) and
  produces **zero** numbers in the tracker.
- **Two bench modes are dead**, so tracker entries (n) and (g) are unreproducible while
  presented as measured.
- **Ground-truth circularity is untreated** in `calibration_eval.py`: 544 queries scored
  against a KDE whose bandwidths *are* the encoder's own length scales
  (`GT_POS_BW 0.75 = pos_l`, `GT_TIME_BW 20.0 = time_l`). The fix already exists in the
  same repo (`astm_traces.py:284-321`) and was applied to `cmd_bench` only. This is
  predicted reviewer attack #1 landing unopposed on contribution 3's headline.
- **Contribution 3 is invisible in the one surface anyone will look at.** `QueryRouter._pack`
  computes `z`/`confident`/`null` (`astm_traces:819-840`); `workshop_demo:431-434` throws
  all three away, and `astm_moved` (`:448`) passes `min_sim=0.0`, disabling the crosstalk
  filter. Paper plan line 144 cites this demo as evidence contribution 1 is "running".
- **`demo_associative_memory.py`'s headline is an artifact.** The Lissajous path
  (`:59`) self-intersects; frames 6 and 18 coincide to 8.8e-16 m at the default
  `n_frames=24`, capping recall at 23/24 = 0.958 for **every** `hd_dim`. The reproduced
  sweep is 0.917 / 0.958 / 0.958 / 0.958 — saturated at 512, i.e. **one real data point**
  backing `README.md:176-179`'s "the bundling-capacity tradeoff".
- **`probe_jepa_predictability.py` computes two algebraic identities numerically** —
  measurement (c) is *identically* (a) because the unit-modulus factor cancels exactly,
  and (b) returns 1.0 for any phase table. Its "place-code transport sanity 1.0000" never
  calls `enc.ctx_pos`, so it validates nothing about the encoder it claims to check.
- **`two_loop_experiment` test 4** calls interleaved ~0.1 s neighbours "revisit" frames and
  produces the 0.08 m "loop-closure recall" headline in `results_robotics.html:228-230`
  with no nearest-stored-frame baseline. Tracker (t) records that the whitening-leak fix
  moved it 0.08 → 0.08 m — a metric that does not respond to changing its own estimator.

### Reconciled recommendation — **KEEP, cut 48%, fix 4 things**

The defence is right that provenance is clean and that the science core is real; the
prosecution is right about volume. Three prosecution verdicts should be softened:

- **`probe_jepa_predictability.py` → supporting-out-of-scope, not scope-creep.** It served
  a *decision*: tracker (f) records that the bind adds nothing beyond persistence, which
  killed a line of work. It also retracted its own vacuous null in-code. Archive to the
  JEPA→FHRR workstream, don't delete.
- **`object_map.py` → split verdict, not "redundant".** `localize` + `_cluster` (~180
  lines) produce the reference set behind the headline numbers. Keep those; delete the
  ~324-line server.
- **`export_static_demo.py` → scope-creep stands, but it is not broken.** The prosecution's
  "frames missing" claim is false (551 JPEGs, 15 MB, present). Its self-verification and
  quantisation bookkeeping are careful. The charge lands on *what* it exports (the ported
  workshop memory, not ASTM), not on its existence.

### Proposed minimal paper-critical set — **≈4,945 lines (52% of current; 48% removed)**

| Component | Lines |
|---|---:|
| `vsa.py` (theirs — attribute, never claim) | 224 |
| `test_vsa.py` | 246 |
| `__init__.py` | 31 |
| `classroom_pipeline.py` core (encoders / pose / embed / localize only) | ~300 |
| `astm_traces.py` (fix the two dead bench modes) | 1,480 |
| `astm_sweep.py` | 667 |
| `calibration_eval.py` (+ bandwidth sweep) | 459 |
| `crosstalk_scaling.py` | 260 |
| `online_whitening_eval.py` | 407 |
| `crossover_analysis.py` | 327 |
| `moved_object_synthetic.py` | 364 |
| `object_map.localize` + `_cluster` | ~180 |
| **Total** | **≈4,945** |

Every number in the tracker survives this cut.

---

## 6. Actions

**Blocking — reproduction is broken today**

1. **Fix `astm_traces.py:1322`** — `tr.null_calib["where"]` → `KeyError('where')` under v2
   keys (`"decode|trace|kind"`). Restores `bench --bias-bench` and tracker entry (n)'s
   class-weighting result.
2. **Fix `astm_traces.py:1274-1276`** — add the `if d == "_meta": continue` guard that
   exists at `:949` and `:1156`. Restores `bench --compare` and tracker entry (g)'s
   multi-scale-time comparison.
3. **Commit `vsa_cognitive_mapping/` AND `external/` to git.** Both are untracked. Every
   claim of the form "verified bit-identical to the previous build" (tracker (m),
   `classroom_pipeline.py:758-759`) is currently unfalsifiable — there is no prior
   revision to diff against. This also fixes the audit trail on the 17 modified external
   files. Add `share/` to `.gitignore` (`.gitignore:24` covers only `share/python-wheels/`).

**Correctness — a reviewer will find these**

4. **`calibration_eval.py`: re-score at 3 bandwidths + `ExactEventTable.supported()`.** The
   fix already exists at `astm_traces.py:284-321`; apply it to the 544-query battery.
   Cheap, and it disarms predicted reviewer attack #1 on contribution 3.
5. **Apply FIX 1 to `astm_sweep.py` too** — no `--gt-bandwidths`, no `supported` criterion
   (0 grep hits). Contribution 4's D×N headline is still scored against estimator-matched
   ground truth.
6. **`two_loop_experiment.py` test 4: add a nearest-stored-frame baseline, or remove the
   0.08 m number from the paper.** Also strike/caveat `results_robotics.html:228-230`.
7. **Fix `classroom_pipeline.py:365`** — semantic/event traces are built inside `for n in S`,
   so under the default `--subset stride --subset-stride 3` they see ~1,499 of 4,498
   detections. The content memories' train/val split is silently inherited by memories with
   no held-out evaluation, discarding two-thirds of the evidence and inflating exactly the
   rare-class crosstalk reported as the √N frequency-bias finding.
8. **Fix the `cmd_demo` whitening leak** (`:492-498`): PCA statistics are fit over all
   frames including the ones labelled "held-out", and `W` is never persisted. The workshop
   explicitly stores and reuses `W` for this reason (`external:618-625`).
9. **Surface abstention in `workshop_demo.py`** — pass `z`/`confident`/`null` into the JSON
   (`:431-434`) and restore `which_moved`'s `min_sim` (`:448`). Then **strike paper-plan
   line 144's citation of the demo as contribution-1 evidence** — a live panel is a
   demonstration, not a measurement.

**Archive (move to `archive/`, do not delete)**

10. `demo_associative_memory.py` (219) + `visualize.py` (279) — closed island; nothing
    imports `visualize`, and `plots/vsa_demo/` (14 Jul) is cited by no wiki page. **Also
    fix `README.md:176-179`**, which reports the geometry artifact as a capacity result.
11. `probe_jepa_predictability.py` (274) → JEPA→FHRR workstream artifacts.
12. `two_loop_experiment.py` tests 3–4 (~490); keep tests 1–2 as a ~120-line machinery
    appendix.
13. `language_query.py` (337) — until it builds a real `TraceSet` with a time factor. The
    published differentiator (`results_robotics.html:474-480`: "composes algebraically with
    time and place in a single unbind") is **not exercised** — the memory at `:137` has no
    time factor and all 15 queries are pure where-queries. This is an afternoon's work and
    would make the file load-bearing.
14. `object_map.py`'s `serve` / `snapshot` / `ObjectMap` / `_PAGE` (~324); keep `localize`
    + `_cluster`.
15. `classroom_pipeline.compare_clocks` (`:1018-1194`, ~180) — or fix it: match λ_dist to
    λ_time at the walk's median motion (1.035 m / 50 frames, already printed at `:1063-1066`)
    and re-run in `--place-mode object`.
16. `workshop_demo.py` (1,173) + `export_static_demo.py` (203) + `run_demo_pipeline.py`
    (125) → reclassify as demo/outreach, explicitly **not** evidence.

**Merge — kill the duplication**

17. **One shared calibration/deprojection module.** `K_FX/K_FY/K_CX/K_CY`, `CAM_OFFSET` and
    `_pixel_to_world` are byte-identical live copies at `classroom_pipeline.py:68-70,222-233`
    and `object_map.py:50-52,69-79` — one feeding the ASTM event stream, one feeding the
    reference clusters those events are scored against. Change the extrinsic in one and the
    two halves of the same measurement silently disagree. The "self-contained script"
    justification (`classroom_pipeline.py:67`) does not hold: `object_map.py:44-46` already
    imports from it.
18. **Delete `workshop_demo._astm_extras` (`:344-402`); call `QueryRouter.query`.** Its
    header claims it "Mirrors QueryRouter's routing rules exactly" and it does not — under
    the module's own advertised multi-scale build it decodes the field with the wrong time
    key while the peak marker comes from the correct path, silently.
19. **Retire two of the three class→place maps.** `classroom_pipeline.semantic_*`,
    `ObjectMap`, `astm_traces.M_what_where` use different atom seeding (index-ordered vs
    md5), different bases, `ls` 0.6 vs 0.75, `hd` 4096 vs 8192. Their fields are not
    numerically comparable.
20. **Generate `share/classroom_demo/index.html` from `PAGE`** instead of hand-syncing 181
    duplicate lines.
21. **Sweep `--cluster-radius` and `--min-count`** (`object_map.py:344-345`). The greedy rule
    (`:95-101`) consumes neighbours of failed seeds, so the object count is order-dependent
    on detector confidence — and that count and those positions are the reference for the
    headline 0.086 m. Bound the reference's own uncertainty, and move the self-consistency
    caveat from `results_robotics.html` into `object_map.py` where the reference is produced.
    Also fix the hardcoded "83 objects" in the figure title (`:328`).

**Citations to add, at the point of description**

22. `vsa.py:3-12` and `astm_traces.py` (FPE core + range kernels) → **Plate 2003; Komer et
    al. CogSci 2019; Frady et al. 2021 arXiv:2109.03429.** State explicitly that the range
    kernel is *standard VFA*, not novel — the practitioner finding is the engineering
    constraint (the kernel is not unit-modulus, so probe-side only), which is correct and
    well-documented.
23. `classroom_pipeline.py:738-845`, `crosstalk_scaling.py`, `astm_traces.py:871-889`,
    `README.md:176-179` → **Frady, Kleyko & Sommer, Neural Computation 2018.**
24. `online_whitening_eval.py:3-5`, `crosstalk_scaling.py:39-42`,
    `classroom_pipeline._content_phasors` → **Mu & Viswanath ICLR 2018; Ethayarajh EMNLP
    2019; Su et al. 2021; Ganesan et al. NeurIPS 2021** — *and credit the workshop*
    (`external/…/classroom_associative_memory.py:778-783`; `diagnostics.py:18-33`) as the
    origin of `--pca-whiten` and the anisotropy observation on this dataset. Contribution 2's
    honest residue is the recall-collapse consequence, the N-scaling law and the causal
    estimator — which is exactly what our two files deliver.
25. `astm_traces.py:8-11` → **Krausse et al. NICE 2025**, stating the multi-trace design as
    an engineering *alternative* to one-trace-plus-resonator, not as a replacement.
26. `object_map.py` → **Snyder et al. npj Unconv. Comp. 3:13 2026 (VSA-OGM).**
27. `language_query.py:29-32` → add the open-vocabulary asymmetry sentence: VLMaps /
    CLIP-Fields / NLMap are open-vocab at the *perception* level; this stores 38 closed-set
    COCO atoms and is open-vocab at query time only.
28. **Find a selective-prediction / OOD-detection citation** for contribution 3, and
    reconcile the Monte-Carlo null against Frady 2018's analytic non-match distribution.
    Track the Furlong quasi-probability/KDE ask as *open*, not as grounding.
29. **Find or concede an HDC class-imbalance citation** for `_class_weights`. Meanwhile fix
    the docstring at `astm_traces.py:882-885`: `conf/√W_c` makes class mass ∝ √W_c — it
    **damps** the bias, it does not "equalize" it, which matches the project's own
    measurement that balanced only "fixes MID + 2/3 of LOW".

**Docstring overclaims to correct (code-level honesty pass)**

30. `workshop_demo.py:345` "Mirrors QueryRouter's routing rules exactly" — false under
    multi-scale time. `workshop_demo.py:2-8` "every encode/unbind … calls *their* functions"
    — 407 lines call ours. `astm_traces.py:45-49` names "energy" as a frontier axis; nothing
    measures energy, and the line is stale post-entry (t). `astm_traces.py:101` — justify
    `POS_OK_M = 1.0` (1.33× the FPE length scale, ~6% of a 6.9×6.7 m floor, against an
    8–11 cm headline claim) as `TIME_OK_FR` is justified. `run_demo_pipeline.py:3`
    "a NEW recording is just a config file" — false for 3 of 5 panels
    (`workshop_demo.py:295,467-469` hardcode classroom paths). `two_loop_experiment.py:39-46`
    "revisit frames". `astm_sweep.py:414-440` "Pareto" — bytes, latency and accuracy are all
    monotone in D, so `fig_pareto` is `fig_accuracy_vs_D` rescaled; restate as "accuracy
    saturates at D=4096". `demo_associative_memory.py:25` and `visualize.py:3-7` — attribute
    to `init-am`, not `jepa`.
31. **State D-dependence of the `which_moved` claim.** It is demonstrated at D=16384 — 2×
    the headline D=8192 and 4× the "Pareto knee" D=4096. Say so.
32. **Remove shipped agent-session artifacts**: `crossover_analysis.py:58-67` sleeps 60 s and
    retries its import because "another agent owns that file";
    `language_query.py:184-189` probes `load_events` for a parameter it has had since
    `astm_traces.py:207`.

**Paper-level attribution (not a code fix)**

33. **Contribution 1 cannot claim the Phasor core.** `vsa.py` is Snyder's and Krausse's. The
    novelty must sit entirely in the multi-trace architecture and query router above it
    (`astm_traces.py`).
34. **Strike "cross-implementation validation (two codebases, same numbers)"** (paper plan
    :200-201; tracker (c)). Verified: ours = 1,239 frames / 4,498 detections; theirs =
    2,478 / 8,878. Reported position L1 0.14 m vs 0.183 m is a 31% gap on different subsets,
    with different pose association (ours interpolates and unwraps, theirs takes nearest),
    and **both call the same verbatim-ported Phasor core**. This is a reproducibility check
    across two drivers of one algorithm.
35. **Add coverage.** `test_vsa.py` guards 224 of 9,502 lines (**2.4%**), and every file that
    produces paper evidence is untested. Tracker entries (r)/(s) found four evaluation-design
    defects in exactly those untested files by adversarial review rather than by test. For an
    ICRA submission that is the exposure worth naming.

---

## Related pages

- [ASTM Paper Plan (live tracker)](2026-07-29-vsa-query-layer-paper-plan.md)
- [VSA Cognitive Maps working doc v2](2026-07-30-vsa-cognitive-maps-working-doc.md)
- [Demo explainer](2026-07-29-demo-explainer.md)
- [Classroom results](../experiments/2026-07-29-vsa-cognitive-map-classroom-results.md)
