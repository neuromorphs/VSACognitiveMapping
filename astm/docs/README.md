# ASTM documentation — read these in the browser

GitHub shows `.html` files as **source code**, not as rendered pages. Use the
"open rendered" links below (they work with no repo configuration; both
services simply proxy this branch's raw files).

| Document | Read it | What it is |
|---|---|---|
| **Results briefing** (start here) | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/results_robotics_share.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/results_robotics_share.html) | Every measured result, written for a SLAM / semantic-mapping audience. TL;DR box, plain-language explainers, 9 figures, and a positioned related-work review across five themes with 29 references. **Self-contained** (figures embedded) — this is the one to send people. |
| **Concept explainer** | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/demo_explainer.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/demo_explainer.html) | What the live demo shows and how to read each panel — the screen-share companion. |
| **Code explainer** | [open rendered ▸](https://raw.githack.com/neuromorphs/VSACognitiveMapping/astm/astm/docs/code_explainer.html) · [mirror](https://htmlpreview.github.io/?https://github.com/neuromorphs/VSACognitiveMapping/blob/astm/astm/docs/code_explainer.html) | Every module, when it was added, the memory taxonomy (which vector answers which question), and the λ-decay deep-dive. |
| **Working document v2** | [read on GitHub ▸](working-document-v2.md) | Project-level view: objectives scored, architecture as built, the original open questions with measured answers. Markdown — renders natively. |
| **Demo explainer (markdown)** | [read on GitHub ▸](demo-explainer.md) | The same concept explainer in markdown, with the limitations ledger. |

`results_robotics.html` is the same briefing but with figures as **relative
links** to `../figures/` — better for local reading after a clone, worse
through a proxy. Prefer `results_robotics_share.html` online.

## Figures

All result figures are in [`../figures/`](../figures/) and render natively on
GitHub. The ones worth looking at first:

- `two_loop_experiment.png` — revisit battery: bimodal occupancy, cross-loop control, loop-closure recall
- `crosstalk_scaling.png` — the interference-vs-N measurement (raw slope 0.97 vs whitened 0.86)
- `semantic_query_chair_memory_object.png` — "where are the chairs?" over depth-derived object positions
- `working_memory_person.png` — decaying working memory tracking a person, then honestly forgetting
- `language_query.png` — free-text query localised through the memory's own algebra
- `crossover_analysis.png` — the honest bytes/latency economics vs an exact event table
