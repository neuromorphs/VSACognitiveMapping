# Associative memory math

The math behind the VSA/HD associative-memory pipeline: the `Phasor` algebra
(`src/vsa_cognitive_mapping/vsa.py`), the three content encoders compared in
`scripts/encoder_sweep.py`, the memory-trace construction in
`scripts/associative_memory.py`, and the metrics used to compare encoders.

## 1. Phasor algebra primitives

A `Phasor` is a vector of unit-modulus complex numbers, seeded independently
per instance:

$$B \in \mathbb{C}^d, \quad B_m = e^{i\phi_m}, \quad \phi_m \sim \mathrm{Unif}(0, 2\pi) \text{ i.i.d.}$$

**Bind** $\otimes$ (elementwise complex multiply — the frequency-domain form
of circular convolution):

$$(A \otimes B)_m = A_m B_m$$

**Unbind** $\oslash$ (elementwise complex divide — the approximate inverse of bind):

$$(A \oslash B)_m = A_m / B_m$$

**Bundle** (elementwise mean of $k$ phasors — superposes them into one trace
that still correlates with each input):

$$\mathrm{bundle}(A^{(1)}, \dots, A^{(k)})_m = \frac{1}{k}\sum_{j=1}^{k} A^{(j)}_m$$

**Fractional power encoding (FPE)** — raising a base phasor to a continuous
scalar exponent $v$:

$$B^{v}_m = e^{iv\phi_m}$$

This is a group homomorphism in $v$:

$$B^{a} \otimes B^{b} = B^{a+b} \qquad \text{since } e^{ia\phi_m}e^{ib\phi_m} = e^{i(a+b)\phi_m}$$

which gives a bell/sinc-shaped similarity kernel over the encoded scalar —
the standard VSA scheme for embedding continuous quantities (position, time,
or any other scalar) into a bind/bundle-able vector.

**Two distinct similarity definitions** are used in the codebase — easy to
conflate, so worth stating explicitly:

- `Phasor.similarity(A, B)` — the *average per-dimension phase-difference
  cosine*, used to check algebraic identities (e.g. self-similarity = 1):

$$\mathrm{sim}(A, B) = \frac{1}{d}\mathrm{Re}\left(\sum_{m=1}^{d} \bar{A}_m B_m\right)$$

- `phasor_correlation_matrix(Z)` — L2-normalize each whole content vector
  first, then take the real part of the Hermitian inner product. This is
  standard cosine similarity treating $\mathbb{C}^d \cong \mathbb{R}^{2d}$,
  and it's what `encoder_sweep.py` actually uses to compare encoded content
  vectors pairwise:

$$\mathrm{corr}_{ij} = \mathrm{Re}\left(\hat z_i \cdot \overline{\hat z_j}\right), \qquad \hat z_i = z_i / \lVert z_i \rVert_2$$

## 2. The three content encoders

All three map a batch of JEPA latents $z \in \mathbb{R}^{N \times d_{in}}$ to
content phasors $z^{\text{content}} \in \mathbb{C}^{N \times d}$, where $d$
is `hd_dim`.

### random-proj

A random Gaussian projection matrix, scaled to keep projected magnitudes
$O(1)$ regardless of $d_{in}$:

$$W \in \mathbb{R}^{d_{in} \times 2d}, \quad W_{jk} \sim \mathcal{N}\!\left(0, \tfrac{1}{d_{in}}\right) \text{ i.i.d.}$$

Project and split into in-phase/quadrature halves:

$$[I \mid Q] = zW \in \mathbb{R}^{N \times 2d}, \qquad I, Q \in \mathbb{R}^{N \times d}$$

Normalize each $(I, Q)$ pair onto the unit circle — keep only the phase,
discard the magnitude:

$$z^{\text{content}}_{n,k} = \frac{I_{n,k} + iQ_{n,k}}{\sqrt{I_{n,k}^2 + Q_{n,k}^2 + \varepsilon}} = e^{i\theta_{n,k}}, \qquad \theta_{n,k} = \mathrm{atan2}(Q_{n,k}, I_{n,k})$$

Because $W$'s entries are i.i.d. $\mathcal{N}(0, 1/d_{in})$, this is a
Johnson–Lindenstrauss-style random projection: pairwise angles between rows
of $z$ are approximately preserved by $[I\mid Q]$ before the phase-only
normalization discards magnitude. Nothing about $z$'s geometry is thrown
away except that one magnitude — which is why random-proj's fidelity is so
much higher than either FPE-based encoder below.

### pca-fpe

**Step 1 — PCA via SVD, standardized scores.** Center $z$, take the economy
SVD, and read off explained variance directly from the singular values:

$$\tilde z = z - \bar z, \qquad \tilde z = USV^\top, \qquad \mathrm{evr}_j = \frac{s_j^2 / (N-1)}{\sum_{j'} s_{j'}^2 / (N-1)}$$

Project onto the top-$K$ right-singular vectors and standardize each
resulting column to unit variance:

$$\hat s_{n,c} = \left(\tilde z\, V_{:,1:K}\right)_{n,c}, \qquad s_{n,c} = \hat s_{n,c} / \sigma_c, \qquad \sigma_c = \mathrm{std}_n(\hat s_{n,c}), \quad c = 1,\dots,K$$

**Step 2 — one independent base phasor per component:** $B_1,\dots,B_K \in \mathbb{C}^d$.

**Step 3 — FPE-encode each frame's component score, then bundle (mean)
across the $K$ components:**

$$z^{\text{content}}_n = \frac{1}{K}\sum_{c=1}^{K} B_c^{\,s_{n,c}/\ell}, \qquad \ell = \texttt{length\_scale}$$

This bundle is a **raw complex mean, not renormalized** — its magnitude
shrinks toward 0 as the $K$ per-axis phases spread apart (destructive
interference). That shrinkage is the mechanism behind the "everything looks
the same" / "everything looks orthogonal" collapse at the extremes of $K$
and $\ell$ discussed earlier: large $\ell$ pushes every exponent toward 0
(all $K$ terms point the same direction, magnitude stays near 1, but every
frame converges to the same vector); large $K$ at fixed $\ell$ dilutes any
one axis's distinguishing signal via averaging.

### full-fpe

Identical FPE + bundle step to pca-fpe, but skipping the PCA rotation —
every raw latent dimension is standardized and used as its own axis
($K = d_{in}$ instead of a PCA subspace):

$$s_{n,j} = \frac{z_{n,j} - \bar z_j}{\sigma_j + \varepsilon}, \qquad j = 1,\dots,d_{in}$$

$$z^{\text{content}}_n = \frac{1}{d_{in}}\sum_{j=1}^{d_{in}} B_j^{\,s_{n,j}/\ell}$$

## 3. Memory-trace construction

`scripts/associative_memory.py`'s `build_memory` binds each frame's content
encoding (random-proj, via `random_project_to_phasor`) to independent base
phasors for time and 2D position, then bundles every frame into one trace.
Given frame index $t_n$ and position $(x_n, y_n)$:

$$\mathrm{trace}_n = z^{\text{content}}_n \otimes B_t^{\,t_n} \otimes B_x^{\,x_n} \otimes B_y^{\,y_n}$$

$$M = \frac{1}{N}\sum_{n=1}^{N} \mathrm{trace}_n$$

Querying $M$ back out (unbind by a candidate time or position, then sweep
FPE-similarity against candidates) isn't implemented as library code in this
repo yet — see the "Associative Memory 1/2" sections of
`notebooks/loading_sample_blender_data.ipynb` for the worked pattern.

## 4. Comparison metrics

`scripts/encoder_sweep.py` compares encoders using two metrics, both derived
from the encoded content's pairwise similarity matrix
$\mathrm{corr}^{\text{content}}$ (via `phasor_correlation_matrix`, §1 above)
and the raw latent's cosine-similarity matrix
$\mathrm{corr}^{\text{raw}}_{ij} = z_i \cdot z_j / (\lVert z_i\rVert_2 \lVert z_j\rVert_2)$:

**Fidelity** — Pearson correlation between the two similarity matrices'
strict upper triangles (does the encoding preserve which frames look alike?):

$$\mathrm{fidelity} = \mathrm{corr}\Big(\{\mathrm{corr}^{\text{raw}}_{ij}\}_{i<j},\ \{\mathrm{corr}^{\text{content}}_{ij}\}_{i<j}\Big)$$

**Orthogonality** — how close the encoded content vectors are to mutually
orthogonal (1.0 = every pair orthogonal, no interference when bundled
together; 0.0 = every pair identical, total collapse):

$$\mathrm{orthogonality} = 1 - \frac{1}{\binom{N}{2}}\sum_{i<j} \left|\mathrm{corr}^{\text{content}}_{ij}\right|$$
