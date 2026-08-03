"""Self-contained CPU demo of a VSA associative-memory cognitive map.

No external data, no download, no GPU. It exercises the ported phasor core
(``vsa_cognitive_mapping.vsa``) end to end on a small synthetic robot walk,
reproducing the mechanism from ``neuromorphs/VSACognitiveMapping``'s
associative-memory pipeline in miniature:

    1. A robot visits N frames along a trajectory; each frame has a position
       (x, y), a capture time t, and an "appearance" embedding (here random
       256-d vectors standing in for YOLO/JEPA features).
    2. Each appearance is projected to a unit-modulus phasor "content" vector
       (``random_project_to_phasor``), the same content encoder the upstream
       "random-proj" path uses.
    3. Two associative memories are built by binding content to a context
       phasor and bundling (the classroom design keeps one memory per axis so
       each only has to discriminate along its own dimension):
           M_pos  = mean_n  content_n  (x) ctx_pos(x_n, y_n)
           M_time = mean_n  content_n  (x) ctx_time(t_n)
       where ctx_pos(x, y) = Bx**(x/l) (x) By**(y/l)  and  ctx_time(t) = Bt**(t/l)
       are fractional-power-encoded (FPE) context phasors.
    4. Queries unbind a memory by a candidate context and clean up against the
       stored content codebook:
           "what did I see at (x, y)?"  ->  M_pos (/) ctx_pos(x, y)  ->  argmax similarity
       Self-recall error is reported in physical units (metres, frames), the
       same way the upstream ``evaluate`` command scores itself.

Run:
    python vsa_cognitive_mapping/demo_associative_memory.py
    python vsa_cognitive_mapping/demo_associative_memory.py --n-frames 32 --hd-dim 4096
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import Phasor, random_project_to_phasor  # noqa: E402


# --------------------------------------------------------------------------
# Scene: a synthetic robot walk with per-frame position, time, and appearance
# --------------------------------------------------------------------------

def make_scene(n_frames: int, embed_dim: int, seed: int):
    """Return (positions (N,2), times (N,), appearance embeddings (N, embed_dim)).

    Positions trace a Lissajous-ish loop so frames are well spread in (x, y);
    appearances are distinct random vectors (a clean stand-in for a real
    detector's features, one unique landmark per frame)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    positions = np.stack([3.0 * np.cos(theta), 2.0 * np.sin(2.0 * theta)], axis=1)
    embeddings = rng.normal(size=(n_frames, embed_dim)).astype(np.float32)
    return positions, t, embeddings


# --------------------------------------------------------------------------
# Context encoders (FPE) -- position (two bound axes) and time (one axis)
# --------------------------------------------------------------------------

class ContextEncoders:
    def __init__(self, hd_dim: int, seed: int, pos_length_scale: float, time_length_scale: float):
        self.Bx = Phasor(dim=hd_dim, seed=seed + 0)
        self.By = Phasor(dim=hd_dim, seed=seed + 1)
        self.Bt = Phasor(dim=hd_dim, seed=seed + 2)
        self.pos_l = pos_length_scale
        self.time_l = time_length_scale

    def ctx_pos(self, x: float, y: float) -> Phasor:
        return (self.Bx ** float(x / self.pos_l)) * (self.By ** float(y / self.pos_l))

    def ctx_time(self, t: float) -> Phasor:
        return self.Bt ** float(t / self.time_l)


# --------------------------------------------------------------------------
# Memory build / query
# --------------------------------------------------------------------------

def encode_content(embeddings: np.ndarray, hd_dim: int, seed: int) -> list[Phasor]:
    """Project real appearance embeddings to unit-modulus phasor content."""
    x = torch.from_numpy(embeddings)
    z, _ = random_project_to_phasor(x, d=hd_dim, seed=seed)
    z = z.numpy()
    return [Phasor(data=z[i]) for i in range(z.shape[0])]


def build_memory(content: list[Phasor], contexts: list[Phasor]) -> Phasor:
    """Bundle (mean) of content_n bound to context_n into one trace."""
    bound = np.stack([(c * ctx).values for c, ctx in zip(content, contexts)])
    return Phasor(data=bound.mean(axis=0))


def cleanup(residual: Phasor, codebook: list[Phasor]) -> np.ndarray:
    """Similarity of a residual against every stored content vector."""
    return np.array([residual.similarity(c) for c in codebook])


# --------------------------------------------------------------------------
# Demo sections
# --------------------------------------------------------------------------

def evaluate_position_recall(positions, content, enc: ContextEncoders):
    """For every stored frame, query by its own (x, y) and measure how far
    the top-1 recalled frame's position is from the true one."""
    M = build_memory(content, [enc.ctx_pos(x, y) for x, y in positions])
    errors, exact = [], 0
    for n, (x, y) in enumerate(positions):
        residual = M / enc.ctx_pos(x, y)
        best = int(np.argmax(cleanup(residual, content)))
        exact += int(best == n)
        errors.append(float(np.linalg.norm(positions[best] - positions[n])))
    errors = np.array(errors)
    return {
        "exact_frame_acc": exact / len(positions),
        "mean_pos_err_m": float(errors.mean()),
        "median_pos_err_m": float(np.median(errors)),
    }


def evaluate_time_recall(times, content, enc: ContextEncoders):
    M = build_memory(content, [enc.ctx_time(t) for t in times])
    errors, exact = [], 0
    for n, t in enumerate(times):
        residual = M / enc.ctx_time(t)
        best = int(np.argmax(cleanup(residual, content)))
        exact += int(best == n)
        errors.append(abs(float(times[best] - times[n])))
    errors = np.array(errors)
    return {
        "exact_frame_acc": exact / len(times),
        "mean_time_err_frames": float(errors.mean()),
    }


def demo_position_kernel(positions, content, enc: ContextEncoders, frame: int, span=4.0, steps=17):
    """Reverse query: unbind M_pos by one frame's *content* and sweep a
    candidate x across the map -- similarity should peak at the true x,
    showing the FPE similarity kernel is intact through bind + bundle."""
    M = build_memory(content, [enc.ctx_pos(x, y) for x, y in positions])
    true_x, true_y = positions[frame]
    residual = M / content[frame]  # content -> where did I see this?
    xs = np.linspace(true_x - span, true_x + span, steps)
    sims = np.array([residual.similarity(enc.ctx_pos(x, true_y)) for x in xs])
    peak_x = xs[int(np.argmax(sims))]
    return xs, sims, true_x, peak_x


def capacity_sweep(positions, embeddings, seed, dims=(256, 512, 1024, 2048)):
    """How position-recall accuracy grows with hd_dim as N traces compete in
    one bundled trace -- the bundling-capacity point from the upstream docs."""
    rows = []
    for d in dims:
        content = encode_content(embeddings, hd_dim=d, seed=seed)
        enc = ContextEncoders(hd_dim=d, seed=seed + 100, pos_length_scale=1.0, time_length_scale=4.0)
        res = evaluate_position_recall(positions, content, enc)
        rows.append((d, res["exact_frame_acc"], res["mean_pos_err_m"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-frames", type=int, default=24)
    ap.add_argument("--hd-dim", type=int, default=2048)
    ap.add_argument("--embed-dim", type=int, default=256, help="appearance embedding dim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    positions, times, embeddings = make_scene(args.n_frames, args.embed_dim, args.seed)
    content = encode_content(embeddings, hd_dim=args.hd_dim, seed=args.seed)
    enc = ContextEncoders(hd_dim=args.hd_dim, seed=args.seed + 100,
                          pos_length_scale=1.0, time_length_scale=4.0)

    print("=" * 70)
    print("VSA associative-memory cognitive map -- CPU demo")
    print(f"frames={args.n_frames}  hd_dim={args.hd_dim}  embed_dim={args.embed_dim}  seed={args.seed}")
    print("=" * 70)

    pos = evaluate_position_recall(positions, content, enc)
    print("\n[1] Position self-recall  (query 'what did I see at (x, y)?')")
    print(f"    exact-frame accuracy : {pos['exact_frame_acc']:.3f}")
    print(f"    mean position error  : {pos['mean_pos_err_m']:.4f} m")
    print(f"    median position error: {pos['median_pos_err_m']:.4f} m")

    tim = evaluate_time_recall(times, content, enc)
    print("\n[2] Time self-recall  (query 'what did I see at time t?')")
    print(f"    exact-frame accuracy : {tim['exact_frame_acc']:.3f}")
    print(f"    mean time error      : {tim['mean_time_err_frames']:.3f} frames")

    frame = args.n_frames // 3
    xs, sims, true_x, peak_x = demo_position_kernel(positions, content, enc, frame)
    print(f"\n[3] Position similarity kernel  (reverse query for frame {frame})")
    print(f"    true x = {true_x:+.2f}   kernel peak x = {peak_x:+.2f}   "
          f"(|error| = {abs(peak_x - true_x):.3f} m)")
    bar_max = sims.max() if sims.max() > 0 else 1.0
    for x, s in zip(xs, sims):
        bar = "#" * int(40 * max(s, 0) / bar_max)
        marker = "  <- true x" if abs(x - true_x) < (xs[1] - xs[0]) / 2 else ""
        print(f"    x={x:+5.2f} | {bar:<40} {s:+.3f}{marker}")

    print("\n[4] Bundling capacity sweep  (position exact-frame accuracy vs hd_dim)")
    print("    hd_dim | exact acc | mean pos err (m)")
    print("    -------+-----------+-----------------")
    for d, acc, err in capacity_sweep(positions, embeddings, args.seed):
        print(f"    {d:6d} |   {acc:5.3f}   |   {err:.4f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
