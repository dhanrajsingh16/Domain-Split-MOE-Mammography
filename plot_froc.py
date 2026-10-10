#!/usr/bin/env python3
"""
FROC curve generator for CBIS-DDSM 458-image test set.
Reuses the EXACT evaluation method from 'MoE Evaluation x 3 Seeds.ipynb'
(same test path, conf=0.001, iou=0.6, imgsz=1280, class-aware matching),
so the curve is guaranteed consistent with the paper's reported S@2:
  Universal 0.2021 | DS-MoE 0.2027 | Gated 0.1950

Produces: figures/froc_curves.pdf and .png  (publication quality).
Run on server:  conda activate torch_gap  &&  python -u plot_froc.py
"""

from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG (identical to the paper's evaluation notebook) ─────────────────────
BASE      = "/home/modss/Desktop/dhanraj"
HIT_IOU   = 0.30
RCALC     = 0            # router: alphabetical -> calc=0
RMASS     = 1           # mass=1
NC        = 4
NAMES     = ["benign_mass", "malignant_mass", "benign_calc", "malignant_calc"]
SEEDS     = [42, 1, 123]

TEST_DIR  = Path(f"{BASE}/canonical_4class/test")     # <-- 458-image test set
TEST_IMGS = sorted((TEST_DIR / "images").glob("*"))
TEST_LBL  = TEST_DIR / "labels"
OUT_DIR   = Path(f"{BASE}/figures")

# publication palette: distinct in colour AND line style (grayscale-safe)
COLORS    = {"Universal": "#2B6CB0", "DS-MoE": "#C05621", "Gated MoE": "#276749"}
STYLES    = {"Universal": "-",       "DS-MoE": "--",       "Gated MoE": (0, (1, 1))}
LWIDTH    = {"Universal": 2.2,       "DS-MoE": 2.8,        "Gated MoE": 2.2}

# ── HELPERS (copied verbatim from the notebook) ───────────────────────────────
def load_gt(img_path):
    lbl = TEST_LBL / (Path(img_path).stem + ".txt")
    if not lbl.exists():
        return [], []
    with Image.open(img_path) as im:
        W, H = im.size
    boxes, labels = [], []
    for line in lbl.read_text().strip().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = map(float, line.split())
        x1 = (cx - bw/2) * W;  y1 = (cy - bh/2) * H
        x2 = (cx + bw/2) * W;  y2 = (cy + bh/2) * H
        boxes.append([x1, y1, x2, y2]);  labels.append(int(c))
    return boxes, labels

def box_iou_np(b1, b2):
    if not b1 or not b2:
        return np.zeros((len(b1), len(b2)))
    b1 = np.array(b1, dtype=np.float32)
    b2 = np.array(b2, dtype=np.float32)
    ix1 = np.maximum(b1[:,0:1], b2[:,0]);  iy1 = np.maximum(b1[:,1:2], b2[:,1])
    ix2 = np.minimum(b1[:,2:3], b2[:,2]);  iy2 = np.minimum(b1[:,3:4], b2[:,3])
    inter = np.maximum(ix2-ix1, 0) * np.maximum(iy2-iy1, 0)
    a1 = (b1[:,2]-b1[:,0]) * (b1[:,3]-b1[:,1])
    a2 = (b2[:,2]-b2[:,0]) * (b2[:,3]-b2[:,1])
    return inter / np.maximum(a1[:,None] + a2[None,:] - inter, 1e-7)

def predict_detect(model, img_path, imgsz=1280):
    res = model.predict(str(img_path), imgsz=imgsz, conf=0.001, iou=0.6,
                        verbose=False, device=0)[0]
    preds = []
    if res.boxes is not None and len(res.boxes):
        for box in res.boxes:
            c  = int(box.cls.item())
            cf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            preds.append((c, x1, y1, x2, y2, cf))
    return preds

def predict_router(model, img_path):
    res = model.predict(str(img_path), imgsz=640, verbose=False, device=0)[0]
    probs = res.probs.data.cpu().numpy()
    return float(probs[RCALC]), float(probs[RMASS])

# ── OVERALL FROC as a FULL CURVE (same matching as notebook's froc_eval) ──────
def froc_curve_overall(all_preds, all_gt_boxes, all_gt_cls, n_imgs):
    """Returns (fpi, sens) step arrays for the class-aware overall FROC."""
    flat = [(cf, i, c, x1, y1, x2, y2)
            for i, preds in enumerate(all_preds)
            for (c, x1, y1, x2, y2, cf) in preds]
    flat.sort(key=lambda x: -x[0])
    total_gt = sum(len(g) for g in all_gt_cls)
    if total_gt == 0 or not flat:
        return np.array([0.0]), np.array([0.0])
    matched = [set() for _ in range(n_imgs)]
    tp, fp = [], []
    for (cf, i, c, x1, y1, x2, y2) in flat:
        gt_b = [b for b, gc in zip(all_gt_boxes[i], all_gt_cls[i]) if gc == c]
        gt_j = [j for j, gc in enumerate(all_gt_cls[i]) if gc == c]
        hit = False
        if gt_b:
            ious = box_iou_np([[x1,y1,x2,y2]], gt_b)[0]
            best = int(np.argmax(ious))
            if ious[best] >= HIT_IOU and gt_j[best] not in matched[i]:
                matched[i].add(gt_j[best]);  hit = True
        tp.append(1 if hit else 0);  fp.append(0 if hit else 1)
    tpc = np.cumsum(tp);  fpc = np.cumsum(fp)
    sens = tpc / total_gt;  fpi = fpc / n_imgs
    return fpi, sens

def s_at_2(fpi, sens):
    """S@2 exactly as the notebook computes it (searchsorted, no interp)."""
    if len(fpi) == 0:            return 0.0
    if fpi[-1] < 2.0:            return float(sens[-1])
    if fpi[0]  > 2.0:            return 0.0
    return float(sens[np.searchsorted(fpi, 2.0)])

# ── PLOT (publication quality) ────────────────────────────────────────────────
def plot_froc(curves, s2_means, x_axis, out_path):
    plt.rcParams.update({
        "font.size": 12, "font.family": "DejaVu Sans",
        "axes.linewidth": 1.1, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ["Universal", "DS-MoE", "Gated MoE"]:
        arr  = np.array(curves[name])          # (n_seeds, n_pts)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        color = COLORS[name]
        s2 = s2_means[name]
        ax.plot(x_axis, mean, color=color, linestyle=STYLES[name],
                linewidth=LWIDTH[name], zorder=3,
                label=f"{name}  (S@2 = {s2:.3f})")
        ax.fill_between(x_axis, np.clip(mean-std, 0, 1), np.clip(mean+std, 0, 1),
                        color=color, alpha=0.12, zorder=1)
        ax.plot(2.0, s2, marker="o", markersize=7, color=color,
                markeredgecolor="white", markeredgewidth=1.3, zorder=5)

    YMAX = 0.40    # data tops out ~0.27; zoom in so the curves are readable
    ax.axvline(x=2.0, color="#718096", linestyle=(0, (4, 3)),
               linewidth=1.2, zorder=2)
    ax.text(2.06, 0.008, "2 FP/image", color="#4A5568", fontsize=9,
            ha="left", va="bottom")
    ax.set_xlabel("False positives per image", fontsize=13)
    ax.set_ylabel("Sensitivity", fontsize=13)
    ax.set_xlim(0, 4); ax.set_ylim(0, YMAX)
    ax.set_xticks(np.arange(0, 4.1, 0.5))
    ax.set_yticks(np.arange(0, YMAX + 0.001, 0.05))
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.96,
              edgecolor="#CBD5E0", title="Mean over 3 seeds (42/1/123)",
              title_fontsize=10)
    plt.tight_layout()
    fig.savefig(str(out_path) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_path) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}.pdf / .png")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fast re-style path: REPLOT=1 skips inference, re-plots from cached data.
    cache = OUT_DIR / "froc_curve_data.npz"
    if os.environ.get("REPLOT") == "1" and cache.exists():
        print("REPLOT mode: re-plotting from cached curve data ...")
        d = np.load(cache, allow_pickle=True)
        plot_froc(d["curves"].item(), d["s2_means"].item(), d["x_axis"],
                  OUT_DIR / "froc_curves")
        sys.exit(0)

    from ultralytics import YOLO

    n_imgs = len(TEST_IMGS)
    print(f"Test images: {n_imgs}")
    all_gt_boxes, all_gt_cls = [], []
    for img in TEST_IMGS:
        gb, gc = load_gt(img)
        all_gt_boxes.append(gb);  all_gt_cls.append(gc)
    print(f"Total GT boxes: {sum(len(g) for g in all_gt_cls)}")

    x_axis = np.linspace(0, 4, 500)
    curves = {"Universal": [], "DS-MoE": [], "Gated MoE": []}
    s2_all = {"Universal": [], "DS-MoE": [], "Gated MoE": []}

    for SEED in SEEDS:
        print(f"\n{'='*50}\nSEED = {SEED}\n{'='*50}")
        mass_m   = YOLO(f"{BASE}/canonical_mass_clean/runs/mass_clean_seed{SEED}/weights/best.pt")
        calc_m   = YOLO(f"{BASE}/canonical_calc_clean/runs/calc_clean_seed{SEED}/weights/best.pt")
        uni_m    = YOLO(f"{BASE}/canonical_4class/runs/universal_seed{SEED}/weights/best.pt")
        router_m = YOLO(f"{BASE}/canonical_router_clean/runs/router_clean_seed{SEED}/weights/best.pt")

        print("Running inference on 458 test images...")
        mass_raw, calc_raw, uni_preds, rprobs = [], [], [], []
        for i, img in enumerate(TEST_IMGS):
            if i % 100 == 0:
                print(f"  {i}/{n_imgs}...")
            mass_raw.append(predict_detect(mass_m, img))
            calc_raw.append(predict_detect(calc_m, img))
            uni_preds.append(predict_detect(uni_m, img))
            rprobs.append(predict_router(router_m, img))

        mass_preds = mass_raw
        calc_preds = [[(c+2, x1, y1, x2, y2, cf) for (c, x1, y1, x2, y2, cf) in p]
                      for p in calc_raw]
        ds_preds = [m + c for m, c in zip(mass_preds, calc_preds)]
        gated_preds = []
        for m, c, (p_calc, p_mass) in zip(mass_preds, calc_preds, rprobs):
            gm = [(cls, x1, y1, x2, y2, cf*p_mass) for (cls, x1, y1, x2, y2, cf) in m]
            gc = [(cls, x1, y1, x2, y2, cf*p_calc) for (cls, x1, y1, x2, y2, cf) in c]
            gated_preds.append(gm + gc)

        for name, preds in [("Universal", uni_preds),
                            ("DS-MoE",    ds_preds),
                            ("Gated MoE", gated_preds)]:
            fpi, sens = froc_curve_overall(preds, all_gt_boxes, all_gt_cls, n_imgs)
            # interpolate step curve onto the common x-grid (prepend origin)
            xf = np.concatenate([[0.0], fpi])
            yf = np.concatenate([[0.0], sens])
            y_interp = np.interp(x_axis, xf, yf, right=yf[-1])
            curves[name].append(y_interp)
            s2 = s_at_2(fpi, sens)
            s2_all[name].append(s2)
            print(f"  {name:<12} S@2 = {s2:.4f}")

    s2_means = {k: float(np.mean(v)) for k, v in s2_all.items()}

    # cache the computed curves so future styling tweaks need no inference
    np.savez(cache,
             curves=np.array(curves, dtype=object),
             s2_means=np.array(s2_means, dtype=object),
             x_axis=x_axis)
    print(f"Cached curve data -> {cache}")

    print("\nGenerating FROC plot ...")
    plot_froc(curves, s2_means, x_axis, OUT_DIR / "froc_curves")

    print("\n--- FROC S@2 Summary (mean +/- std over 3 seeds) ---")
    for name in ["Universal", "DS-MoE", "Gated MoE"]:
        v = np.array(s2_all[name])
        print(f"  {name:<12} {v.mean():.4f} +/- {v.std():.4f}")
    print("\nExpected (paper):  Universal 0.2021 | DS-MoE 0.2027 | Gated 0.1950")
