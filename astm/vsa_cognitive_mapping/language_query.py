"""Language -> VSA: query the cognitive map with free text via LM latents.

Two mechanisms, compared side by side (the question: how easy is it to parse
a language-model latent into VSA space and query the map?):

  A. SOFT GROUNDING (the safe route). Embed the query text and all class
     names with a sentence encoder; route the query to its nearest class
     atom(s) by text cosine; run the ordinary VSA where-query. The LM is a
     front-end to the codebook; the memory itself is untouched.

  B. DIRECT LATENT BINDING (the wedge). Rebuild the what<->where memory with
     class atoms DERIVED FROM TEXT EMBEDDINGS: whiten the embeddings over a
     noun vocabulary (the isotropy medicine, applied to the language
     modality), random-project to unit phasors, bind to object positions,
     bundle. Then a free-text query ("something to sit on") is encoded by
     the SAME text->phasor map and unbinds the memory DIRECTLY — no codebook
     routing. If the LM's semantic geometry survives the projection, the
     position field lights up at the right furniture even for phrases that
     were never stored. Whether it survives is exactly what we measure.

Both run against the object-position event stream (events_object.csv), so
"where" means the furniture location, not the robot's vantage.

Usage (first run downloads the MiniLM encoder, ~90 MB):
    python vsa_cognitive_mapping/language_query.py            # battery + figure
    python vsa_cognitive_mapping/language_query.py --query "something to sit on"
    python vsa_cognitive_mapping/language_query.py --mechanism direct --query "cold storage"

Prior art this speaks to: CLIP-Fields / VLMaps / NLMap (language-queryable
maps via VLM embeddings, no VSA); SPA language + HDC text encodings (VSA, no
robot map). The combination — LM latent as an unbinding key over a spatial
VSA memory — is the research programme's zero-shot-modality wedge.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vsa_cognitive_mapping.vsa import random_project_to_phasor, pca_components  # noqa: E402
from vsa_cognitive_mapping.astm_traces import (  # noqa: E402
    load_events, ExactEventTable, TraceSet, DEFAULT_OUT, EVENTS_OBJECT_CSV,
)
from vsa_cognitive_mapping.classroom_pipeline import ClassroomEncoders  # noqa: E402

# a small noun vocabulary for whitening statistics (isotropy over language
# space needs more samples than our 38 class names; ~180 common object nouns)
VOCAB_EXTRA = """table desk shelf cabinet drawer lamp light window door wall floor ceiling
carpet rug curtain blind whiteboard blackboard projector screen monitor keyboard mouse
computer printer scanner phone tablet camera speaker headphones microphone cable plug
socket switch button handle knob lock key box bag basket bin bucket jar bottle cup mug
glass plate bowl fork knife spoon pan pot kettle toaster oven stove sink tap towel soap
brush broom mop vacuum duster sponge cloth paper notebook book folder file pen pencil
marker eraser ruler scissors stapler tape glue clip pin board poster picture painting
frame mirror clock watch calendar map globe plant flower vase pot tree leaf branch stone
rock chair seat stool bench sofa couch armchair cushion pillow blanket sheet mattress
bed cot crib wardrobe closet hanger hook rack stand tripod ladder step stair rail fence
gate sign label sticker badge card ticket coin wallet purse backpack suitcase luggage
umbrella coat jacket hat glove scarf shoe boot sock shirt trousers dress skirt belt
helmet mask goggles tool hammer screwdriver wrench pliers drill saw nail screw bolt nut
washer spring wire rope chain hose pipe valve pump motor engine wheel tyre pedal brake
handlebar frame seatpost extinguisher alarm detector sensor router modem server rack
laptop television remote controller console cartridge disc drive memory chip circuit
battery charger adapter transformer generator panel meter gauge dial display""".split()


def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


class LanguageMap:
    """Text->phasor map + the two query mechanisms over one event stream."""

    def __init__(self, out_dir=DEFAULT_OUT, hd=8192, seed=0, pos_l=0.75, grid=72):
        self.ev = load_events(out_dir, events_name=EVENTS_OBJECT_CSV) \
            if _load_events_supports_name() else _load_events_fallback(out_dir)
        self.table = ExactEventTable(self.ev)
        self.classes = sorted(set(self.ev["class"]))
        self.hd, self.seed = hd, seed

        print("loading text encoder (MiniLM)...")
        self.enc_text = load_encoder()
        # embed class names + noun vocabulary; whiten over the union
        vocab = list(dict.fromkeys(self.classes + VOCAB_EXTRA))
        emb = self.enc_text.encode(vocab, normalize_embeddings=False,
                                   show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float64)
        white, _ = pca_components(emb, n_components=min(128, emb.shape[1]))
        self.vocab = vocab
        self.vocab_white = white
        self.class_rows = {c: vocab.index(c) for c in self.classes}
        # fit stats reused for out-of-vocab queries: recompute projection basis
        self._mu = emb.mean(axis=0, keepdims=True)
        centered = emb - self._mu
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = white.shape[1]
        self._vt, self._s = vt[:k], np.maximum(s[:k] / np.sqrt(max(len(vocab) - 1, 1)), 1e-8)

        # text -> phasor projection (fixed W, same for atoms and queries)
        z, self._W = random_project_to_phasor(
            torch.from_numpy(np.ascontiguousarray(white)).float(), d=hd, seed=seed)
        self.vocab_phasor = z.numpy().astype(np.complex128)

        # spatial side: position codes + memory built with TEXT-DERIVED atoms
        self.enc_pos = ClassroomEncoders(hd, seed + 100, pos_l, 20.0)
        pad = 1.0
        xs, ys = self.ev["x"], self.ev["y"]
        self.gx = np.linspace(xs.min() - pad, xs.max() + pad, grid)
        ratio = (ys.max() - ys.min() + 2 * pad) / (xs.max() - xs.min() + 2 * pad)
        self.gy = np.linspace(ys.min() - pad, ys.max() + pad, max(8, int(grid * ratio)))
        G = np.empty((len(self.gy), len(self.gx), hd), np.complex64)
        for j, yy in enumerate(self.gy):
            for i, xx in enumerate(self.gx):
                G[j, i] = self.enc_pos.ctx_pos(xx, yy).values
        self.G_flat = G.reshape(-1, hd)

        print(f"building text-atom memory over {len(self.ev['t'])} object-position events...")
        acc = np.zeros(hd, np.complex128)
        # log class-weighting (the frequency-bias fix) so rare classes survive
        counts = defaultdict(float)
        for c, cf in zip(self.ev["class"], self.ev["conf"]):
            counts[c] += cf
        kcnt = defaultdict(int)
        for c in self.ev["class"]:
            kcnt[c] += 1
        for c, x, y, cf in zip(self.ev["class"], xs, ys, self.ev["conf"]):
            w = cf * np.log1p(kcnt[c]) / kcnt[c]
            atom = self.vocab_phasor[self.class_rows[c]]
            acc += w * atom * self.enc_pos.ctx_pos(float(x), float(y)).values
        self.M = acc / max(sum(counts.values()), 1e-9)

    # ---- text -> whitened latent -> phasor (same map as the atoms) --------
    def text_phasor(self, text):
        e = np.asarray(self.enc_text.encode([text], show_progress_bar=False),
                       dtype=np.float64)
        w = ((e - self._mu) @ self._vt.T) / self._s
        z, _ = random_project_to_phasor(torch.from_numpy(w).float(),
                                        d=self.hd, W=self._W)
        return z.numpy().astype(np.complex128)[0], w[0]

    def text_sims(self, text):
        """Cosine of the query's whitened latent vs every class name's."""
        _, w = self.text_phasor(text)
        cw = self.vocab_white[[self.class_rows[c] for c in self.classes]]
        a = w / (np.linalg.norm(w) + 1e-9)
        b = cw / (np.linalg.norm(cw, axis=1, keepdims=True) + 1e-9)
        return dict(zip(self.classes, b @ a))

    # ---- mechanism A: soft grounding --------------------------------------
    def query_grounded(self, text, top_k=3):
        sims = self.text_sims(text)
        ranked = sorted(sims.items(), key=lambda kv: -kv[1])[:top_k]
        # mixture probe weighted by text similarity (softmax-ish, temp 0.1)
        wts = np.exp(np.array([s for _, s in ranked]) / 0.1)
        wts /= wts.sum()
        residual = np.zeros(self.hd, np.complex128)
        for (c, _), w in zip(ranked, wts):
            residual += w * (self.M / self.vocab_phasor[self.class_rows[c]])
        field = self._decode(residual)
        return field, ranked

    # ---- mechanism B: direct latent unbind ---------------------------------
    def query_direct(self, text):
        q, _ = self.text_phasor(text)
        return self._decode(self.M / q)

    def _decode(self, residual):
        s = np.real(self.G_flat @ np.conj(residual.astype(np.complex64))) / self.hd
        return s.reshape(len(self.gy), len(self.gx))

    def peak(self, field):
        j, i = np.unravel_index(int(np.argmax(field)), field.shape)
        return float(self.gx[i]), float(self.gy[j]), float(field.max())


def _load_events_supports_name():
    import inspect
    try:
        return "events_name" in inspect.signature(load_events).parameters
    except (TypeError, ValueError):
        return False


def _load_events_fallback(out_dir):
    import csv as _csv
    path = os.path.join(out_dir, EVENTS_OBJECT_CSV)
    ev = {"class": [], "x": [], "y": [], "t": [], "conf": []}
    with open(path) as f:
        for r in _csv.DictReader(f):
            ev["class"].append(r["class"])
            for k in ("x", "y", "t", "conf"):
                ev[k].append(float(r[k]))
    for k in ("x", "y", "t", "conf"):
        ev[k] = np.array(ev[k])
    return ev


# battery: (query text, intended class) — paraphrases never stored as atoms
BATTERY = [
    ("chair", "chair"), ("tv", "tv"), ("person", "person"),
    ("something to sit on", "chair"),
    ("a screen for watching videos", "tv"),
    ("a human being", "person"),
    ("seating furniture", "chair"),
    ("a computer display", "tv"),
    ("place to put my coffee mug", "dining table"),
    ("luggage", "suitcase"),
    ("cold storage for food", "refrigerator"),
    ("a portable computer", "laptop"),
    ("cutting tool for paper", "scissors"),
    ("wall clock", "clock"),
    ("bag you carry", "handbag"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--query", default=None, help="single free-text query")
    ap.add_argument("--mechanism", choices=["grounded", "direct", "both"], default="both")
    ap.add_argument("--hd-dim", type=int, default=8192)
    args = ap.parse_args()

    lm = LanguageMap(out_dir=args.out_dir, hd=args.hd_dim)
    table = lm.table

    def eval_one(text, intended=None, verbose=True):
        out = {}
        truth = table.where(what=intended) if intended else None
        if args.mechanism in ("grounded", "both"):
            f, ranked = lm.query_grounded(text)
            px, py, sim = lm.peak(f)
            err = np.hypot(px - truth[0], py - truth[1]) if truth else float("nan")
            out["grounded"] = (px, py, sim, err, ranked)
        if args.mechanism in ("direct", "both"):
            f = lm.query_direct(text)
            px, py, sim = lm.peak(f)
            err = np.hypot(px - truth[0], py - truth[1]) if truth else float("nan")
            out["direct"] = (px, py, sim, err, None)
        if verbose:
            print(f"\nquery: '{text}'" + (f"   (intended: {intended})" if intended else ""))
            for mech, (px, py, sim, err, ranked) in out.items():
                line = f"  [{mech:>8}] peak ({px:+.2f},{py:+.2f}) sim {sim:+.4f}"
                if truth:
                    line += f"  err {err:.2f} m"
                if ranked:
                    line += "   grounded to: " + ", ".join(
                        f"{c}({s:+.2f})" for c, s in ranked)
                print(line)
        return out

    if args.query:
        eval_one(args.query)
        return

    results = {"grounded": [], "direct": []}
    ok_ground = 0
    ground_hits = {}
    for text, intended in BATTERY:
        out = eval_one(text, intended)
        for mech in results:
            if mech in out:
                results[mech].append((text, intended, out[mech][3]))
        if "grounded" in out:
            ground_hits[text] = out["grounded"][4][0][0] == intended
            if ground_hits[text]:
                ok_ground += 1

    print("\n===== SUMMARY =====")
    if results["grounded"]:
        errs = np.array([r[2] for r in results["grounded"]])
        print(f"grounded:  top-1 text-grounding accuracy {ok_ground}/{len(BATTERY)}"
              f" | position err mean {errs.mean():.2f} m, median {np.median(errs):.2f} m,"
              f" <=1 m: {(errs <= 1.0).sum()}/{len(errs)}")
    if results["direct"]:
        errs = np.array([r[2] for r in results["direct"]])
        print(f"direct:    position err mean {errs.mean():.2f} m, median {np.median(errs):.2f} m,"
              f" <=1 m: {(errs <= 1.0).sum()}/{len(errs)}")
        worst = sorted(results["direct"], key=lambda r: -r[2])[:3]
        print("  hardest for direct:", "; ".join(f"'{t}'→{c} ({e:.1f} m)" for t, c, e in worst))

    # paraphrase-only split: exclude the exact-name queries (text == class
    # name — identity freebies for both mechanisms); the paraphrases are the
    # actual zero-shot language claim.
    n_exact = sum(1 for t, c in BATTERY if t == c)
    para_texts = [t for t, c in BATTERY if t != c]
    print(f"paraphrase-only split (n={len(para_texts)}; excludes {n_exact} "
          f"exact-name queries):")
    for mech in ("grounded", "direct"):
        if not results[mech]:
            continue
        pe = np.array([e for t, c, e in results[mech] if t != c])
        line = (f"  {mech:>8}: position err mean {pe.mean():.2f} m, "
                f"median {np.median(pe):.2f} m, <=1 m: "
                f"{(pe <= 1.0).sum()}/{len(pe)}")
        if mech == "grounded":
            ph = sum(1 for t in para_texts if ground_hits.get(t))
            line += f" | top-1 text-grounding {ph}/{len(para_texts)}"
        print(line)

    # figure: one worked example, both mechanisms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    text = "something to sit on"
    fg, ranked = lm.query_grounded(text)
    fd = lm.query_direct(text)
    truth = table.where(what="chair")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, f, title in ((axs[0], fg, f"soft grounding → {ranked[0][0]}"),
                         (axs[1], fd, "direct latent unbind")):
        lim = np.max(np.abs(f)) or 1.0
        ax.imshow(f, origin="lower", cmap="coolwarm", vmin=-lim, vmax=lim,
                  extent=[lm.gx[0], lm.gx[-1], lm.gy[0], lm.gy[-1]], aspect="equal")
        px, py, _ = lm.peak(f)
        ax.scatter([px], [py], marker="x", s=130, color="black", linewidth=2.4,
                   label="decoded peak")
        ax.scatter([truth[0]], [truth[1]], s=150, facecolor="none", edgecolor="black",
                   linewidth=2, label="chair (modal truth)")
        ax.set_title(f"'{text}' — {title}", fontsize=11)
        ax.legend(fontsize=8)
    path = os.path.join(args.out_dir, "language_query.png")
    fig.savefig(path, dpi=140)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
