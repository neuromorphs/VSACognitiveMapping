# Concepts

No prior VSA knowledge assumed. This explains what the memory is, why learned
image features fight it, and what all the metrics in this repo actually measure.

---

## 1. The memory is one vector

A conventional map stores a list: chair at (2.1, 4.4), laptop at (−1.0, 3.2),
and so on. This memory stores **one fixed-size vector** for all of it, and the
vector is the same size whether you saw ten objects or ten thousand.

The vectors are **phasors**: every component is a complex number of magnitude 1,
so a vector of dimension D is really a list of D angles. Two operations:

**Binding (`⊗`)** — elementwise multiply, which *adds the angles*. It combines
two things into one vector that resembles neither. Used to attach a *what* to a
*where*:

```
event = CHAIR ⊗ S(2.1, 4.4)
```

Binding is invertible: divide by `CHAIR` and you recover `S(2.1, 4.4)`.

**Bundling (`+`)** — add the vectors and renormalise. It makes a superposition
that resembles *all* its inputs. This is how many events become one memory:

```
M = event₁ + event₂ + … + eventₙ
```

To ask "where are the chairs?", divide `M` by `CHAIR` and correlate the residue
against a grid of candidate positions. Bright spots are the answer.

The whole system is that. Nothing is searched, nothing is scanned.

## 2. Position is encoded so that geometry is arithmetic

Position uses **fractional power encoding**: pick a random base vector `B` and
raise it to a real power.

```
S(x, y) = Bₓ^(x/ℓ) ⊗ B_y^(y/ℓ)          ℓ = 0.75 m
```

The important property is that this is a **group homomorphism**:

```
S(a) ⊗ S(b) = S(a + b)      exactly
```

Moving a belief by a displacement is one bind. No decoding, no grid, no
approximation. That is why the Kalman filter in `vsa_kalman.py` has a *predict*
step that costs one multiply — see [ADVANCED_SIGREG_VSA.md](ADVANCED_SIGREG_VSA.md).

Similarity between two positions falls off as a sinc-like bump whose width is
set by `ℓ`: half height at ≈0.30 ℓ, first zero at ℓ/2. That is why every heat
map in the demos is a *blob* rather than a lit pixel.

## 3. What limits it: crosstalk

Bundling is lossy. When you divide `M` by `CHAIR`, you get the chair's position
*plus a smear of every other stored event*. That smear is **crosstalk**, and it
sets how much a memory of fixed size can hold.

Crosstalk depends on how the stored vectors are arranged. Think of each vector
as an arrow from the origin:

- If the arrows point in **all directions evenly** — an **isotropic** cloud, a
  fuzzy ball — their interference has random signs and largely cancels.
- If they **bunch into a narrow cone** — **anisotropic** — the interference adds
  up coherently and swamps the answer.

Measure it with cosine: 0 means perpendicular, 1 means identical direction.
Random high-dimensional vectors are near 0. **Learned image features are not.**

## 4. The problem, in numbers

Mean pairwise cosine between frame descriptors from the classroom walk:

| encoder | mean \|cos\| |
|---|---|
| random vectors (D = 8192) | 0.011 |
| DINOv2 ViT-S/14 | 0.359 |
| ResNet-50 | 0.404 |
| YOLOv8n | 0.887 |
| **untrained** ResNet-50 | **0.997** |

YOLOv8n's features are so aligned that every image points nearly the same way.
An untrained network is almost degenerate — which is itself informative, because
it means **extreme anisotropy is the architectural default and training reduces
it**, rather than training causing it.

## 5. The vocabulary of the measurements

**Effective rank** — your vectors have 256 or 384 numbers, so they *could* use
that many independent directions. Effective rank asks how many they actually
use. Measured here: about 7. The data lives in a 7-dimensional pancake inside a
256-dimensional room.

**IsoScore** — the same idea as a 0–1 grade. 0 is a line, 1 is a perfect ball.
Raw features score 0.02–0.12.

**χ (chi)** — how loud the crosstalk is relative to one stored item's signal.
Lower is better. Scales roughly with the number of stored items, so χ values are
only comparable at the same N.

**Whitening** — the standard fix. Measure which directions the data spreads out
in, then rescale so every direction has equal spread. It turns the cone back
into a ball.

Note what whitening actually *is*: a rotation onto the principal axes, then a
**non-uniform** rescale. The rotation is provably inert — a random orthogonal
map reproduces every cosine measurement to twelve decimal places. All the work,
and all the damage, is in the rescale.

**Monotonic** — moves one way only, never turns back. Used a lot here because
several relationships that "should" have an optimum in the middle turn out not
to.

## 6. The thing this project found

Two results that reframe the above, both measured several independent ways.

**Isotropy predicts capacity, not key quality.** Making the cloud rounder lets
you *store* more, but makes the vectors *worse at identifying things*. Push
IsoScore to 1.0 by whitening and retrieval degrades monotonically. Frame
embeddings are rounder than object crops and retrieve far worse. An encoder can
be more isotropic and strictly less useful.

**Crosstalk is driven by the shared mean, not by the spectrum's shape.**
Subtracting the mean leaves IsoScore and effective rank *bit-identical* — they
are computed on the centred covariance, so centring cannot change them — yet it
halves crosstalk. Two representations, same spectrum, 2× different crosstalk.
So the coherent interference comes from every vector sharing a common component
that adds in phase; flattening the spectrum attacks the wrong term.

The practical consequence is blunt: **centre or z-score your features; do not
whiten them.** Per-dimension z-scoring improves retrieval *and* cuts crosstalk
2.3× — there is no trade-off to negotiate. Whitening to 128 components scores
*below chance* on held-out data.

## 7. What the system is not

**It does not estimate pose.** Robot pose comes from LIO-SAM (LiDAR + inertial);
object positions come from depth-camera measurements deprojected through that
pose. The memory is *handed* a pose and stores appearance bound to it, so a
query returns *the pose stored with the most similar stored appearance*. That is
associative recall of a recorded pose, not visual localisation.

This matters for interpreting results. Under closed-set scoring an **untrained**
network matched DINOv2's median recall — because the task only needs the
descriptor to be *consistent*, not *good*. Store "arbitrary vector ↔ pose",
query with that same vector, get the pose back. Any evaluation that does not
hold data out will fail to notice this.

---

Next: [RESULTS_SO_FAR.md](RESULTS_SO_FAR.md) for the measurements and the
corrections, or [ADVANCED_SIGREG_VSA.md](ADVANCED_SIGREG_VSA.md) for where the
research goes from here.
