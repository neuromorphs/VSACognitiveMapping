# ASTM — Algebraic Spatio-Temporal Memory on the Spot classroom data

**Branch provenance.** This branch builds directly on `init-am` (Shay's
classroom associative-memory pipeline). Everything under `astm/` was
developed July 2026 in Paul Kirkland's SSP-SLAM repo against the
[`lorinachey/spot-telluride-workshop-dataset`](https://huggingface.co/datasets/lorinachey/spot-telluride-workshop-dataset),
running the *actual* `init-am` code as the episodic-memory engine (via the
`external/` snapshot noted below) and extending it with a multi-trace
spatio-temporal memory, calibration, working memory, and language queries.

## ▸ Read the results first (rendered, no clone needed)

**[Results briefing — every measured result, for a robotics audience](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/results_robotics_share.html)**
· [concept explainer](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/demo_explainer.html)
· [code explainer](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/code_explainer.html)
· [all docs + mirrors](docs/README.md)

(GitHub renders `.html` as source; those links proxy the raw files so they
display as pages. Markdown docs — [working document v2](docs/working-document-v2.md)
— render natively here.)

## What's here

| File | What it does |
|---|---|
| `vsa.py`, `test_vsa.py` | Port of the jepa-branch phasor core (18/18 upstream tests; py3.9 compat) |
| `classroom_pipeline.py` | embed → build → evaluate → demo on the classroom walk; semantic memories (`query_class`, `query_event`); object-position binding (`--place-mode object`); decaying working memory (`build_now`, hybrid decay clocks `--decay-per event|frame|metre|both`) |
| `astm_traces.py` | The ASTM core: canonical event stream, 4 traces (what⊗where, what⊗when, where⊗when, event), closed-form time-range kernels, query router, calibrated abstention (z-scores), exact-event-table baseline, class-weighting (`log` fixes the sqrt-N frequency bias), hygiene filters |
| `astm_sweep.py` | D×N evaluation matrix (60 cells / 7.7 min; Pareto knee D=4096) |
| `workshop_demo.py` | Live 4-panel browser demo driving the init-am code: replay reverse queries (~15 ms), class→where with [vantage/object/now] modes, ASTM conditional-query panel |
| `object_map.py` | Depth-localized object map + interactive server |
| `language_query.py` | Free-text queries: MiniLM latents (whitened) as direct unbinding keys — median 0.17 m on never-stored paraphrases |
| `online_whitening_eval.py` | Causal (Welford) whitening vs oracle; the "never re-encode stored content" finding |
| `calibration_eval.py` | AUROC / ECE / risk–coverage for the abstention machinery |
| `moved_object_synthetic.py`, `two_loop_experiment.py` | Moved-object detection + fake-second-loop (frame interleaving) revisit battery — 4/4 pass incl. loop-closure recall median 0.08 m |
| `probe_jepa_predictability.py` | VSA-JEPA probe: transport exact; prediction ≡ persistence |
| `run_demo_pipeline.py`, `configs/` | One-command config-driven pipeline (classroom, school_run1, or your dataset) |
| `export_static_demo.py` | 15 MB static shareable demo export (GitHub-Pages ready) |
| `docs/` | Working document v2, demo explainer (md + html), full code explainer (html) |
| `figures/` | Key result figures |

## Headline measured results (classroom walk, 2,478 frames)

- Episodic recall (init-am engine, whitened): position L1 0.14–0.18 m, heading 0.12 rad — cross-validated across the two implementations
- Isotropy ⇔ capacity: raw YOLO features 0.88 mean cosine → 365× coherent crosstalk; PCA whitening → 27×; heading demo 28%→98%
- Conditional queries (event trace): "where was X at t" 8–11 cm; range-window queries via closed-form kernels, 14.5 ms
- Frequency bias fixed: log class-weighting takes rare-class error 5.15 m → 0.12 m without harming frequent classes
- Object-position binding: chair decoded 0.086 m from independently clustered furniture
- Calibrated abstention: z≥3 ≈ sim 0.0037; 93% confident-correct; AUROC 0.87
- Online whitening: causal Welford converges to near-oracle by the final third; frozen calibration-lap insufficient; never re-encode stored content
- Working memory: per-class λ decay (+ frame/metre/both clocks — metre keeps memory through a stationary pause where frame forgets)
- Language: LM latents unbind the map directly (median 0.17 m, never-stored phrases)
- Two-loop (interleaved) battery: bimodal occupancy 5/5, clean cross-loop negative control, injected 1.68 m move detected at 1.51 m, loop-closure recall median 0.08 m (machinery ceiling — near-duplicate content caveat)

## Running

The module currently assumes the SSP-SLAM repo layout (it inserts its parent
directory on `sys.path` and, for `workshop_demo.py`, expects the init-am tree
under `external/VSACognitiveMapping-init-am/`). On this branch the quickest
path is: clone, place this `astm/` folder and the repo root side by side as
documented in `docs/code_explainer.html`, or run the pure-ASTM parts
(`astm_traces.py`, `astm_sweep.py`, `calibration_eval.py`, ...) directly —
they need only `events.csv`/`embeddings.pt` artifacts produced by
`classroom_pipeline.py embed`. Naming note: this folder deliberately does NOT
shadow `src/vsa_cognitive_mapping` (the jepa/init-am package); the demo's
loader handles the two-package coexistence explicitly.

## Related work this builds on / relates to

- Krausse, Neftci, Sommer, Renner — *GC-VSA* (NICE 2025, arXiv:2503.08608):
  the spatio-temporal what⊗where⊗when VSA lineage (synthetic); ASTM is the
  real-robot deployment of the same query algebra.
- Snyder, Capodieci, Gorsich, Parsa — *VSA-OGM* (npj Unconventional
  Computing 2026): SSP occupancy-grid mapping (geometric); ASTM is the
  semantic/temporal complement on the same substrate.
- Snyder et al. — *HyperSpace* (arXiv:2604.15113): modular VSA operator
  framework; independently confirms our "decode/cleanup dominates runtime"
  finding. Porting ASTM's operators onto HyperSpace is an obvious
  integration path.
- le-marmotte (VGPI): one-vector lidar localization; candidate pose source
  where dataset odometry fails (school_run1).

Tracker/log of every measured result: `docs/working-document-v2.md` and the
SSP-SLAM repo's wiki (`2026-07-29-vsa-query-layer-paper-plan.md`, entries a–p).
