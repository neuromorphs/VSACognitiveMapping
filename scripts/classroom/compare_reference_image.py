"""Run a new, out-of-dataset image through the YOLO detector/embedder and
check whether its embedding is more similar to frames where a given class
was actually detected than to frames where it wasn't.

    python scripts/classroom/compare_reference_image.py --image data/scissors --reference-class scissors

Loads the same YOLO model detect_and_embed_classroom.py uses, runs
detection on the new image (reporting what YOLO itself sees in it, as a
sanity check) and embedding extraction (the same penultimate-layer feature,
via model.embed(), as detect_and_embed_classroom.py's run_embed_yolo), then
scores the new image's cosine similarity against every frame's embedding in
--embeddings, split into two groups by whether --detections records
--reference-class in that frame.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
RED = "#e34948"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_yolo_model(yolo_model: str):
    """Lazily import ultralytics, same as detect_and_embed_classroom.py."""
    from ultralytics import YOLO
    return YOLO(yolo_model)


def cosine_similarity_to_all(query: np.ndarray, embeddings: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """query: (D,), embeddings: (N, D) -> (N,) cosine similarities."""
    query_unit = query / (np.linalg.norm(query) + eps)
    emb_unit = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + eps)
    return emb_unit @ query_unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="path to the new, out-of-dataset image")
    parser.add_argument("--reference-class", default="scissors",
                        help="COCO class to check the new image's embedding against, e.g. 'scissors'")
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold for reporting what the detector itself sees "
                             "in the new image (does not affect the embedding, which is extracted "
                             "regardless of any detection)")
    parser.add_argument("--embeddings", default="outputs/classroom_detections/embeddings.pt")
    parser.add_argument("--detections", default="outputs/classroom_detections/detections.csv")
    parser.add_argument("--top-k", type=int, default=10, help="how many nearest-neighbor frames to report")
    parser.add_argument("--out-path", default=None,
                        help="default: outputs/classroom_detections/"
                             "reference_image_similarity_<reference-class>.png")
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"))
    print(f"loaded {args.image}: {image.shape[1]}x{image.shape[0]}")

    device = resolve_device(args.device)
    print(f"loading {args.yolo_model} on device={device}...")
    model = load_yolo_model(args.yolo_model)

    print("running detection on the new image...")
    results = model.predict([image], device=device, conf=args.conf, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        print(f"YOLO found no detections in {args.image} (>= confidence {args.conf})")
    else:
        for k in range(len(boxes)):
            cls_id = int(boxes.cls[k])
            print(f"  detected: {model.names[cls_id]} (confidence={float(boxes.conf[k]):.3f})")

    print("extracting the new image's embedding...")
    new_embedding = model.embed([image], device=device, verbose=False)[0].cpu().numpy()

    data = torch.load(args.embeddings)
    frame_idx = data["frame_idx"].numpy()
    embeddings = data["embedding"].numpy()
    if embeddings.shape[1] != new_embedding.shape[0]:
        raise SystemExit(f"embedding dim mismatch: {args.embeddings} has {embeddings.shape[1]}-dim "
                         f"embeddings but the new image's YOLO embedding is {new_embedding.shape[0]}-dim "
                         f"-- are --embeddings and --yolo-model from the same backbone?")

    det = pd.read_csv(args.detections)
    class_frame_idx = set(det[det["class_name"].str.lower() == args.reference_class.lower()]["frame_idx"])
    if not class_frame_idx:
        available = ", ".join(sorted(det["class_name"].unique()))
        raise SystemExit(f"no '{args.reference_class}' detections found in {args.detections}\n"
                         f"available classes: {available}")
    has_class = np.array([f in class_frame_idx for f in frame_idx])
    print(f"{has_class.sum()}/{len(has_class)} frames have >=1 '{args.reference_class}' detection")

    sims = cosine_similarity_to_all(new_embedding, embeddings)
    sims_with, sims_without = sims[has_class], sims[~has_class]

    print(f"\ncosine similarity to frames WITH '{args.reference_class}': "
         f"mean={sims_with.mean():.3f} median={np.median(sims_with):.3f} (n={len(sims_with)})")
    print(f"cosine similarity to frames WITHOUT '{args.reference_class}': "
         f"mean={sims_without.mean():.3f} median={np.median(sims_without):.3f} (n={len(sims_without)})")

    order = np.argsort(sims)[::-1][:args.top_k]
    n_top_k_with_class = int(has_class[order].sum())
    base_rate = 100 * has_class.mean()
    print(f"\ntop-{args.top_k} most similar frames: {n_top_k_with_class}/{args.top_k} contain "
         f"'{args.reference_class}' (base rate if similarity carried no information: {base_rate:.1f}%)")
    for rank, idx in enumerate(order, 1):
        tag = f"HAS {args.reference_class}" if has_class[idx] else "no match"
        print(f"  {rank}. frame_idx={int(frame_idx[idx])} similarity={sims[idx]:.3f} ({tag})")

    out_path = Path(args.out_path) if args.out_path else \
        Path(f"outputs/classroom_detections/reference_image_similarity_{args.reference_class}.png")

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE, constrained_layout=True)
    bins = np.linspace(sims.min(), sims.max(), 40)
    ax.hist(sims_without, bins=bins, density=True, color=BLUE, alpha=0.55,
           label=f"no {args.reference_class} (n={len(sims_without)})")
    ax.hist(sims_with, bins=bins, density=True, color=RED, alpha=0.55,
           label=f"{args.reference_class} detected (n={len(sims_with)})")
    ax.axvline(sims_without.mean(), color=BLUE, linestyle="--", linewidth=1.5)
    ax.axvline(sims_with.mean(), color=RED, linestyle="--", linewidth=1.5)
    ax.set_facecolor(SURFACE)
    ax.set_title(f"'{Path(args.image).name}' embedding similarity to every frame,\n"
                f"split by whether '{args.reference_class}' was detected there",
                color=INK_PRIMARY, fontsize=12, pad=10)
    ax.set_xlabel("cosine similarity", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("density", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    legend = ax.legend(loc="best", fontsize=8, facecolor=SURFACE, edgecolor=GRID)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nplot written to {out_path}")


if __name__ == "__main__":
    main()
