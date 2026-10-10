#!/usr/bin/env python3
"""
MIAS zero-shot evaluation: Universal vs Domain-Split MoE (3 seeds)
Metrics: FROC S@2 (hit_iou=0.30), mAP50 (iou=0.50)
Run:  python mias_eval.py
"""

import os, sys, tarfile, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE     = Path("/home/modss/Desktop/dhanraj")
MIAS_DIR = Path("/home/modss/Desktop/dhanraj/mias_eval")
SEEDS    = [42, 1, 123]

MASS_W = {s: BASE/f"canonical_mass_clean/runs/mass_clean_seed{s}/weights/best.pt" for s in SEEDS}
CALC_W = {s: BASE/f"canonical_calc_clean/runs/calc_clean_seed{s}/weights/best.pt"  for s in SEEDS}
UNI_W  = {s: BASE/f"canonical_4class/runs/universal_seed{s}/weights/best.pt"       for s in SEEDS}

HIT_IOU   = 0.30   # FROC match threshold
AP_IOU    = 0.50   # mAP50 match threshold
CONF      = 0.01   # low conf → capture full FROC curve
NMS_IOU   = 0.45
IMG_SIZE  = 1280
FP_TARGET = 2.0    # S@2

MIAS_URL  = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"

# MIAS lesion types → 0=benign_mass, 1=malig_mass, 2=benign_calc, 3=malig_calc
MASS_TYPES = {"CIRC", "SPIC", "MISC", "ARCH", "ASYM"}
CALC_TYPES = {"CALC"}
IMG_H = 1024   # MIAS standard image height (for y-axis flip)

# ── STEP 1: DOWNLOAD & CONVERT ───────────────────────────────────────────────
def prepare_images():
    MIAS_DIR.mkdir(parents=True, exist_ok=True)
    pgm_dir = MIAS_DIR / "pgm"
    png_dir = MIAS_DIR / "png"
    pgm_dir.mkdir(exist_ok=True)
    png_dir.mkdir(exist_ok=True)

    # Download if needed
    if not any(pgm_dir.glob("mdb*.pgm")):
        tar_path = MIAS_DIR / "all-mias.tar.gz"
        print(f"Downloading MIAS from {MIAS_URL} ...")
        try:
            urllib.request.urlretrieve(MIAS_URL, tar_path,
                reporthook=lambda b,bs,t: print(f"  {min(b*bs,t)//1024}/{t//1024} KB", end="\r"))
            print()
        except Exception as e:
            sys.exit(f"Download failed: {e}\n"
                     f"Manually download and place PGM files in: {pgm_dir}\n"
                     f"Also place all-mias.txt in: {MIAS_DIR}")
        print("Extracting ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(pgm_dir)
        print(f"Extracted. PGM count: {len(list(pgm_dir.rglob('*.pgm')))}")

    # Convert PGM → PNG RGB (only once)
    pgms = sorted(pgm_dir.rglob("mdb*.pgm"))
    if not pgms:
        sys.exit(f"No PGM files found in {pgm_dir}. Check extraction.")

    existing_pngs = set(p.stem for p in png_dir.glob("*.png"))
    to_convert = [p for p in pgms if p.stem not in existing_pngs]

    if to_convert:
        print(f"Converting {len(to_convert)} PGM → PNG ...")
        for pgm in to_convert:
            img = Image.open(pgm).convert("RGB")
            img.save(png_dir / f"{pgm.stem}.png")
        print("Done.")
    else:
        print(f"All {len(pgms)} PNGs already converted.")

    return sorted(png_dir.glob("mdb*.png")), pgm_dir

# ── STEP 2: PARSE ANNOTATIONS ────────────────────────────────────────────────
def find_ann_file(pgm_dir):
    candidates = list(MIAS_DIR.rglob("all-mias.txt")) + list(pgm_dir.rglob("all-mias.txt"))
    if candidates:
        return candidates[0]
    # also check common alternative name
    alt = list(MIAS_DIR.rglob("*.txt"))
    for f in alt:
        if "mias" in f.name.lower() or "mdb" in open(f).read(50).lower():
            return f
    sys.exit(
        f"Annotation file not found.\n"
        f"Download 'all-mias.txt' from: http://peipa.essex.ac.uk/info/mias.html\n"
        f"Save to: {MIAS_DIR / 'all-mias.txt'}"
    )

def parse_annotations(ann_path):
    """Returns dict: stem → list of (global_cls, x1, y1, x2, y2) in pixel coords."""
    anns = {}
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            ref = parts[0]                 # e.g. mdb001
            abn = parts[2].upper()         # CIRC / CALC / NORM / ...
            anns.setdefault(ref, [])
            if abn == "NORM" or len(parts) < 7:
                continue
            sev  = parts[3].upper()        # B or M
            x_c  = int(parts[4])
            y_c  = int(parts[5])
            r    = int(parts[6])
            # MIAS y-axis: origin at bottom-left → flip to top-left
            y_c  = IMG_H - y_c
            x1   = max(0,      x_c - r)
            y1   = max(0,      y_c - r)
            x2   = min(IMG_H,  x_c + r)
            y2   = min(IMG_H,  y_c + r)
            if x2 <= x1 or y2 <= y1:
                continue
            if abn in CALC_TYPES:
                cls = 2 if sev == "B" else 3
            elif abn in MASS_TYPES:
                cls = 0 if sev == "B" else 1
            else:
                cls = 0 if sev == "B" else 1  # unknown → treat as mass
            anns[ref].append((cls, x1, y1, x2, y2))
    return anns

# ── STEP 3: INFERENCE ────────────────────────────────────────────────────────
def predict(model, img_path, cls_offset=0):
    """Returns list of (cls, conf, x1, y1, x2, y2) in original image coords."""
    res = model.predict(str(img_path), conf=CONF, iou=NMS_IOU,
                        imgsz=IMG_SIZE, verbose=False)[0]
    preds = []
    if res.boxes is not None and len(res.boxes):
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        for b, c, cl in zip(boxes, confs, clses):
            preds.append((cl + cls_offset, float(c),
                          float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    return preds

def ds_moe_predict(mass_mdl, calc_mdl, img_path):
    return predict(mass_mdl, img_path, cls_offset=0) + \
           predict(calc_mdl,  img_path, cls_offset=2)

# ── STEP 4: METRICS ──────────────────────────────────────────────────────────
def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-9)

def froc_s_at_fp(all_preds, all_gts, n_imgs):
    """FROC sensitivity at FP_TARGET FP/image. all_preds: (img_i,cls,conf,x1,y1,x2,y2)."""
    total_gt = sum(len(v) for v in all_gts.values())
    if total_gt == 0:
        return 0.0
    sorted_preds = sorted(all_preds, key=lambda x: -x[2])
    matched = {i: [False] * len(g) for i, g in all_gts.items()}
    tp_cum, fp_cum = 0, 0
    fp_pts, sens_pts = [0.0], [0.0]

    for img_i, cls, conf, x1, y1, x2, y2 in sorted_preds:
        gts = all_gts.get(img_i, [])
        best_iou, best_j = 0.0, -1
        for j, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
            if gc != cls:
                continue
            iou = box_iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= HIT_IOU and best_j >= 0 and not matched[img_i][best_j]:
            matched[img_i][best_j] = True
            tp_cum += 1
        else:
            fp_cum += 1
        sens_pts.append(tp_cum / total_gt)
        fp_pts.append(fp_cum / n_imgs)

    fp_arr   = np.array(fp_pts)
    sens_arr = np.array(sens_pts)
    if fp_arr[-1] < FP_TARGET:
        return float(sens_arr[-1])
    idx = np.searchsorted(fp_arr, FP_TARGET)
    if idx == 0:
        return 0.0
    t = (FP_TARGET - fp_arr[idx-1]) / (fp_arr[idx] - fp_arr[idx-1] + 1e-12)
    return float(sens_arr[idx-1] + t * (sens_arr[idx] - sens_arr[idx-1]))

def class_ap(all_preds, all_gts, cls):
    """AP@IoU=0.50 for a single class."""
    preds = [(i, cf, x1, y1, x2, y2)
             for i, c, cf, x1, y1, x2, y2 in all_preds if c == cls]
    gts   = {i: [(x1,y1,x2,y2) for c,x1,y1,x2,y2 in g if c == cls]
             for i, g in all_gts.items()}
    total = sum(len(v) for v in gts.values())
    if total == 0 or not preds:
        return 0.0
    preds_s = sorted(preds, key=lambda x: -x[1])
    matched = {i: [False]*len(g) for i, g in gts.items()}
    tp_list, fp_list = [], []
    for img_i, conf, x1, y1, x2, y2 in preds_s:
        g = gts.get(img_i, [])
        best_iou, best_j = 0.0, -1
        for j, (gx1, gy1, gx2, gy2) in enumerate(g):
            iou = box_iou((x1,y1,x2,y2), (gx1,gy1,gx2,gy2))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= AP_IOU and best_j >= 0 and not matched[img_i][best_j]:
            matched[img_i][best_j] = True
            tp_list.append(1); fp_list.append(0)
        else:
            tp_list.append(0); fp_list.append(1)
    tp_c = np.cumsum(tp_list)
    fp_c = np.cumsum(fp_list)
    rec  = tp_c / total
    prec = tp_c / (tp_c + fp_c + 1e-9)
    ap   = sum(prec[rec >= t].max() if (rec >= t).any() else 0.0
               for t in np.linspace(0, 1, 101)) / 101
    return float(ap)

def compute_metrics(all_preds, all_gts, n_imgs):
    s2   = froc_s_at_fp(all_preds, all_gts, n_imgs)
    map50 = np.mean([class_ap(all_preds, all_gts, c) for c in range(4)])
    return float(s2), float(map50)

# ── STEP 5: RUN ───────────────────────────────────────────────────────────────
def run_eval(png_files, ann_dict):
    results = {"uni": {}, "moe": {}}
    n_imgs  = len(png_files)

    # build GT lookup by image index
    full_gts = {}
    for idx, png in enumerate(png_files):
        boxes = ann_dict.get(png.stem, [])
        full_gts[idx] = [(c, x1, y1, x2, y2) for c, x1, y1, x2, y2 in boxes]

    total_gt = sum(len(v) for v in full_gts.values())
    print(f"\n{'='*60}")
    print(f"Images: {n_imgs}  |  GT boxes: {total_gt}")
    gt_per_cls = [sum(1 for g in full_gts.values() for c,*_ in g if c==i) for i in range(4)]
    labels = ["benign_mass","malig_mass","benign_calc","malig_calc"]
    for l, n in zip(labels, gt_per_cls):
        print(f"  {l}: {n}")
    print('='*60)

    for seed in SEEDS:
        print(f"\nSeed {seed} — loading models ...")
        uni_mdl  = YOLO(str(UNI_W[seed]))
        mass_mdl = YOLO(str(MASS_W[seed]))
        calc_mdl = YOLO(str(CALC_W[seed]))

        uni_preds, moe_preds = [], []

        for idx, png in enumerate(png_files):
            if (idx+1) % 50 == 0:
                print(f"  [{idx+1}/{n_imgs}]", end="\r")
            up = predict(uni_mdl, png, cls_offset=0)
            mp = ds_moe_predict(mass_mdl, calc_mdl, png)
            for c, cf, x1, y1, x2, y2 in up:
                uni_preds.append((idx, c, cf, x1, y1, x2, y2))
            for c, cf, x1, y1, x2, y2 in mp:
                moe_preds.append((idx, c, cf, x1, y1, x2, y2))

        u_s2, u_map = compute_metrics(uni_preds, full_gts, n_imgs)
        m_s2, m_map = compute_metrics(moe_preds, full_gts, n_imgs)

        results["uni"][seed] = (u_s2, u_map)
        results["moe"][seed] = (m_s2, m_map)

        print(f"\n  Seed {seed} Universal  | FROC S@2: {u_s2:.4f} | mAP50: {u_map:.4f}")
        print(f"  Seed {seed} DS-MoE     | FROC S@2: {m_s2:.4f} | mAP50: {m_map:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("MIAS ZERO-SHOT SUMMARY  (mean ± std, seeds 42/1/123)")
    print(f"{'='*60}")
    print(f"{'Model':<14} | {'FROC S@2':>16} | {'mAP50':>14}")
    print("-"*50)
    for name, key in [("Universal", "uni"), ("DS-MoE", "moe")]:
        s2s  = [results[key][s][0] for s in SEEDS]
        maps = [results[key][s][1] for s in SEEDS]
        print(f"{name:<14} | {np.mean(s2s):.4f} ± {np.std(s2s):.4f}  "
              f"| {np.mean(maps):.4f} ± {np.std(maps):.4f}")
    print(f"{'='*60}")

    # per-seed table
    print("\nPer-seed detail:")
    print(f"{'Seed':<6} {'Model':<14} {'FROC S@2':>10} {'mAP50':>10}")
    for s in SEEDS:
        u = results["uni"][s]
        m = results["moe"][s]
        print(f"{s:<6} {'Universal':<14} {u[0]:>10.4f} {u[1]:>10.4f}")
        print(f"{s:<6} {'DS-MoE':<14} {m[0]:>10.4f} {m[1]:>10.4f}")

# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    png_files, pgm_dir = prepare_images()
    ann_path  = find_ann_file(pgm_dir)
    print(f"Annotation file: {ann_path}")
    ann_dict  = parse_annotations(ann_path)
    n_boxes   = sum(len(v) for v in ann_dict.values())
    print(f"Parsed {n_boxes} boxes from {len(ann_dict)} annotated images")
    run_eval(png_files, ann_dict)
