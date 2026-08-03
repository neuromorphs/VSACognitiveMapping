# Advanced: making an encoder that suits the algebra

Everything in [CONCEPTS.md](CONCEPTS.md) treats the encoder as fixed and repairs
its output afterwards — centre it, z-score it, whiten it. This document is the
other direction: **train a representation whose output is shaped for the memory
in the first place**, and then push the algebra further than a single lookup.

Three rungs, in increasing order of ambition. Rung 1 exists and has been
measured. Rung 2 exists and is unmeasured. Rung 3 is a direction.

---

## Rung 1 — train the projection with an isotropy loss (SIGReg)

### What it is

The `jepa` branch of this repo trains a **128-d head on a frozen DINOv2
backbone** with a JEPA objective: predict the next frame's latent from the
current latent plus the action taken. The loss carries a **SIGReg** term
(`lam: 0.1`) that regularises the latent toward an isotropic Gaussian *during
training*, rather than reshaping it afterwards.

```
image ──► frozen DINOv2 ViT-S/14 (384-d) ──► Head: Linear + BatchNorm (128-d) ──► z
                                                        │
                            predictor(z_t, action) ≈ z_t+1     +  λ · SIGReg(z)
```

Only the head and predictor train; the backbone never moves.

### Running it

```bash
git worktree add ../vsacm_jepa origin/jepa      # or just check the branch out
cd ../vsacm_jepa
export PYTHONPATH=$PWD/src

# build the transition dataset from HuggingFace (~235 frames at 3 Hz)
python scripts/run1_to_transitions.py --out data/spot_run1 --rate 3.0

# train (minutes on CPU; set device: cuda in the yaml for a GPU)
python scripts/train.py --config configs/jepa_run1_scratch.yaml
```

`configs/jepa_run1_ft.yaml` fine-tunes from a SCAND-pretrained checkpoint
instead; `jepa_run1_scratch.yaml` is the from-scratch control that exists
specifically to separate "the isotropy loss did it" from "any training did it".

### What was measured

Held-out place recall on the classroom, bracketed by chance (5.24 m) and the
best achievable (2.02 m), against crosstalk:

| representation | IsoScore | eff-rank | signal (held-out) | χ |
|---|---|---|---|---|
| raw DINOv2 384-d | 0.115 | 8.4 | 46.9% | 10.55 |
| **centred only** | 0.115 | 8.4 | 50.7% | 5.24 |
| **z-scored per-dim** | 0.127 | 10.5 | **51.8%** | 4.67 |
| SIGReg head 128-d | 0.047 | 7.0 | 38.8% | 2.88 |
| whitened k=128 | 1.000 | 128 | **−3.9%** | 0.58 |

Three things came out of this, and only one of them is flattering.

**SIGReg reduces crosstalk without increasing isotropy.** Its IsoScore is
*lower* than raw DINOv2 (more anisotropic), yet χ drops 3.7×. The training log
shows why: signed mean cosine driven to −0.012 while |cos| stays 0.244. It kills
the *common direction*, not the anisotropy — the same mechanism centring uses.
That is a genuine and slightly surprising result.

**It generalises perfectly across the split** — held-out and all-frames signal
are identical at 38.8%, on only 70 training frames.

**But plain z-scoring beats it** (51.8% vs 38.8%) at a fraction of the cost. On
this task, a learned isotropy objective loses to subtracting a mean and dividing
by a standard deviation. Deflating, and worth stating plainly.

### Open questions on this rung

- **235 frames is tiny.** The whole comparison rests on a 3 Hz subsample of one
  82-second walk with 70 training transitions. Repeat it on a longer sequence
  before believing the ordering.
- **Does λ matter?** Only `lam: 0.1` has been run. Sweeping it maps the
  trade-off between prediction and isotropy directly, and is cheap.
- **Does it help a bigger backbone?** Everything used ViT-S/14. A GPU makes
  `dinov2:base`/`large` feasible.
- **Is the predictor doing anything?** The JEPA loss also has an MSE term.
  Train with SIGReg only, and with MSE only, to see which does the work.
- **The action vocabulary is degenerate.** The classroom walk yields
  `forward 116, stop 72, left 46, right 0` — the "right" one-hot slot is never
  trained. Any action-conditioned claim on this data has that hole in it.

---

## Rung 2 — feed the trained latent into the memory

This is built but unmeasured, and it is the obvious next experiment.

Everything in the main pipeline projects features into phasors with
`random_project_to_phasor`: a fixed Gaussian matrix `W ~ N(0, 1/d_in)` of shape
`(d_in, 2D)`, giving (I, Q) pairs normalised onto the unit circle. The input can
be *any* real vector — raw DINOv2, whitened DINOv2, or the SIGReg latent.

So the full chain is already assembled:

```
image → DINOv2 → SIGReg head (128-d) → random_project_to_phasor → ⊗ S(x,y) → bundle → M
```

To measure it, drop the trained latent in where a crop embedding file goes and
run the standard evaluations — `isotropy_ladder`, `crosstalk_scaling`,
`heldout_eval`. The comparison that matters is *learned projection vs whitening
vs z-scoring, scored through the memory rather than in raw cosine space*, which
nobody has run.

Two design questions worth being deliberate about:

**Where should the isotropy live?** SIGReg shapes the 128-d latent, but the
memory actually cares about the *phasor* after random projection. Regularising
the latent is a proxy. A loss applied after the projection — on the phase
distribution itself — is a different and arguably more honest target.

**Should the projection be learned too?** `W` is currently a fixed random draw.
Learning it jointly with the head, under a loss that includes bundling
interference, would be a genuine "train the encoder for the algebra" result
rather than a two-stage approximation.

---

## Rung 3 — put the algebra in the loss

The direction, not a plan.

Every rung above optimises a *proxy* for what the memory needs (isotropy,
predictability) and hopes it transfers. The honest objective is the thing you
actually care about: **after bundling N of these, can you still read one back?**

That is differentiable. Bundling is a sum, unbinding is a division, and the
readout is a correlation — all of it is autograd-friendly. So a loss can be
written directly over a simulated memory:

```
sample N events → bind each to a random place → bundle → unbind one
→ correlate against the grid → penalise the margin between the true
   place and the best competitor
```

Train the encoder to maximise that margin and you are no longer guessing which
geometric property matters; you are optimising retrieval under superposition
directly. The margin has a known ideal form — `√(2D/k)` for k bundled items —
so there is a theoretical curve to compare against.

Why it might not work, stated up front: the gradient has to pass through a
bundle of many terms where the signal is 1/k of the total, so it will be noisy;
and a loss that only rewards retrievability may collapse to a degenerate code
that is retrievable but semantically useless. Both are testable — the second is
exactly what the held-out place metric would catch.

---

## What to run first, if you want a result

In rough order of value per unit effort:

1. **Rung 2 on existing data.** The chain is built; nobody has scored the
   SIGReg latent *through the memory*. Days, not weeks.
2. **A λ sweep on rung 1.** Cheap, and it produces a curve rather than a point.
3. **Rung 1 with a larger backbone** on a GPU — and check whether the
   encoder-independent whitening floor survives.
4. **Rung 3 as a toy.** Do it on synthetic vectors first, where you control the
   geometry and know the right answer, before touching real features.

Before any of it, read the corrections in [RESULTS_SO_FAR.md](RESULTS_SO_FAR.md).
Several of the obvious baselines in this repo were measured wrongly the first
time, and the failure modes are subtle: leakage through a random split, a
constant predictor scoring 73%, and a whitening multiplier that turned out to
depend on which encoder you measured it with.
