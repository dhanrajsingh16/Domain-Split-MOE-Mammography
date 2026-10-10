#!/usr/bin/env python3
"""
Mass-only evaluation on CBIS-DDSM test set.
Compares: Mass Expert vs Universal (mass predictions only)
Metrics:  FROC S@2 (2 FP/img), mAP50  — both for mass classes only (0, 1)
Run:      python -u mass_eval.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

BASE      = Path("/home/modss/Desktop/dhanraj")
TEST_IMGS = BASE / "canonical_4class/images/test"
TEST_LBLS = BASE / "canonical_4class/labels/test"
SEEDS     = [42, 1, 123]
MASS_W    = {s: BASE/f"canonical_mass_clean/runs/mass_clean_seed{s}/weights/best.pt" for s in SEEDS}
UNI_W     = {s: BASE/f"canonical_4class/runs/universal_seed{s}/weights/best.pt"      for s in SEEDS}

CONF      = 0.01
NMS_IOU   = 0.45
IMG_SIZE  = 1280
HIT_IOU   = 0.30
AP_IOU    = 0.50
FP_TARGET = 2.0
B         = 2000
RNG       = np.random.default_rng(0)

MASS_CLASSES = {0, 1}   # benign_mass=0, malig_mass=1


def load_gt():
    img_paths = sorted(TEST_IMGS.glob("*.jpg")) + sorted(TEST_IMGS.glob("*.png"))
    gt_list = []
    for ip in img_paths:
        lbl = TEST_LBLS / (ip.stem + ".txt")
        boxes = []
        if lbl.exists():
            with PILImage.open(ip) as im: W, H = im.size
            for ln in lbl.read_text().strip().splitlines():
                p = ln.split()
                if len(p) < 5: continue
                cls = int(p[0])
                if cls not in MASS_CLASSES: continue
                cx, cy, bw, bh = map(float, p[1:5])
                x1 = (cx - bw/2)*W; y1 = (cy - bh/2)*H
                x2 = (cx + bw/2)*W; y2 = (cy + bh/2)*H
                boxes.append((cls, x1, y1, x2, y2))
        gt_list.append(boxes)
    n_mass_imgs  = sum(1 for g in gt_list if len(g) > 0)
    n_mass_boxes = sum(len(g) for g in gt_list)
    print(f"Test set: {len(img_paths)} images, {n_mass_boxes} mass GT boxes "
          f"({n_mass_imgs} images contain mass)")
    return img_paths, gt_list


def infer_mass_only(model, ip):
    try:
        res = model.predict(str(ip), conf=CONF, iou=NMS_IOU,
                            imgsz=IMG_SIZE, verbose=False)[0]
    except Exception:
        return []
    preds = []
    if res.boxes is not None and len(res.boxes):
        for b, c, cl in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy(),
                             res.boxes.cls.cpu().numpy().astype(int)):
            if cl in MASS_CLASSES:
                preds.append((int(cl), float(c),
                              float(b[0]), float(b[1]), float(b[2]), float(b[3])))
    return preds


def run_inference(img_paths, seed):
    from ultralytics import YOLO
    print(f"  Seed {seed} — loading models ...")
    mass_mdl = YOLO(str(MASS_W[seed]))
    uni_mdl  = YOLO(str(UNI_W[seed]))
    mass_preds, uni_preds = [], []
    for idx, ip in enumerate(img_paths):
        if (idx+1) % 50 == 0:
            print(f"    [{idx+1}/{len(img_paths)}]", end="\r")
        mass_preds.append(infer_mass_only(mass_mdl, ip))
        uni_preds.append(infer_mass_only(uni_mdl, ip))
    print()
    return mass_preds, uni_preds


def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    i = (ix2-ix1)*(iy2-iy1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i/(u+1e-9)


def froc_s2(pred_list, gt_list, indices):
    n = len(indices)
    flat_p = []
    for new_i, orig_i in enumerate(indices):
        for cls, cf, x1, y1, x2, y2 in pred_list[orig_i]:
            flat_p.append((new_i, cls, cf, x1, y1, x2, y2))
    flat_p.sort(key=lambda x: -x[2])
    gt_sub    = [gt_list[i] for i in indices]
    total_gt  = sum(len(g) for g in gt_sub)
    if total_gt == 0: return 0.0
    matched = [[False]*len(g) for g in gt_sub]
    tp, fp = 0, 0
    for new_i, cls, cf, x1, y1, x2, y2 in flat_p:
        biou, bj = 0.0, -1
        for j, (gc, gx1, gy1, gx2, gy2) in enumerate(gt_sub[new_i]):
            if gc != cls: continue
            iou = box_iou((x1,y1,x2,y2),(gx1,gy1,gx2,gy2))
            if iou > biou: biou, bj = iou, j
        if biou >= HIT_IOU and bj >= 0 and not matched[new_i][bj]:
            matched[new_i][bj] = True; tp += 1
        else:
            fp += 1
        if fp/n > FP_TARGET: break
    return float(tp/total_gt)


def map50_mass(pred_list, gt_list, indices):
    flat_p = []
    for new_i, orig_i in enumerate(indices):
        for cls, cf, x1, y1, x2, y2 in pred_list[orig_i]:
            flat_p.append((new_i, cls, cf, x1, y1, x2, y2))
    gt_sub = [gt_list[i] for i in indices]
    aps = []
    for c in sorted(MASS_CLASSES):
        cp = [(i, cf, x1,y1,x2,y2) for i,cl,cf,x1,y1,x2,y2 in flat_p if cl==c]
        cg = [[b[1:] for b in g if b[0]==c] for g in gt_sub]
        total = sum(len(g) for g in cg)
        if total == 0 or not cp:
            aps.append(0.0); continue
        cp.sort(key=lambda x: -x[1])
        matched = [[False]*len(g) for g in cg]
        tp_l, fp_l = [], []
        for i, cf, x1,y1,x2,y2 in cp:
            biou, bj = 0.0, -1
            for j, (gx1,gy1,gx2,gy2) in enumerate(cg[i]):
                iou = box_iou((x1,y1,x2,y2),(gx1,gy1,gx2,gy2))
                if iou > biou: biou, bj = iou, j
            if biou >= AP_IOU and bj >= 0 and not matched[i][bj]:
                matched[i][bj] = True; tp_l.append(1); fp_l.append(0)
            else:
                tp_l.append(0); fp_l.append(1)
        tc = np.cumsum(tp_l); fc = np.cumsum(fp_l)
        rec = tc/total; prec = tc/(tc+fc+1e-9)
        ap = sum(prec[rec>=t].max() if (rec>=t).any() else 0.0
                 for t in np.linspace(0,1,101)) / 101
        aps.append(float(ap))
    return float(np.mean(aps))


def bootstrap(pred_list, gt_list, metric_fn, n_imgs):
    scores = []
    for _ in range(B):
        idx = RNG.integers(0, n_imgs, size=n_imgs)
        scores.append(metric_fn(pred_list, gt_list, idx))
    return np.array(scores)


if __name__ == "__main__":
    img_paths, gt_list = load_gt()
    n       = len(img_paths)
    all_idx = list(range(n))

    results = {}
    for seed in SEEDS:
        mass_p, uni_p = run_inference(img_paths, seed)

        for mname, preds in [("MassExpert", mass_p), ("Universal", uni_p)]:
            s2_pt  = froc_s2(preds, gt_list, all_idx)
            map_pt = map50_mass(preds, gt_list, all_idx)

            print(f"\nBootstrapping {mname} seed {seed} (B={B}) ...")
            bs_s2  = bootstrap(preds, gt_list, froc_s2, n)
            bs_map = bootstrap(preds, gt_list, map50_mass, n)
            ci_s2  = (np.percentile(bs_s2, 2.5), np.percentile(bs_s2, 97.5))
            ci_map = (np.percentile(bs_map, 2.5), np.percentile(bs_map, 97.5))

            results.setdefault(mname, {})[seed] = {
                "s2": s2_pt, "map50": map_pt,
                "s2_ci": ci_s2, "map50_ci": ci_map
            }
            print(f"  FROC S@2 = {s2_pt:.4f}  95% CI [{ci_s2[0]:.4f}, {ci_s2[1]:.4f}]")
            print(f"  mAP50    = {map_pt:.4f}  95% CI [{ci_map[0]:.4f}, {ci_map[1]:.4f}]")

    print(f"\n{'='*74}")
    print("MASS-ONLY SUMMARY  (mean ± std, 3 seeds | HIT_IOU=0.30 | S@2=2FP/img)")
    print(f"Classes: benign_mass=0, malig_mass=1")
    print(f"{'='*74}")
    for mname in ["MassExpert", "Universal"]:
        d    = results[mname]
        s2s  = [d[s]["s2"]    for s in SEEDS]
        maps = [d[s]["map50"] for s in SEEDS]
        lo_s2  = np.mean([d[s]["s2_ci"][0]   for s in SEEDS])
        hi_s2  = np.mean([d[s]["s2_ci"][1]   for s in SEEDS])
        lo_map = np.mean([d[s]["map50_ci"][0] for s in SEEDS])
        hi_map = np.mean([d[s]["map50_ci"][1] for s in SEEDS])
        print(f"\n{mname}")
        print(f"  FROC S@2 : {np.mean(s2s):.4f} ± {np.std(s2s):.4f}  "
              f"95% CI [{lo_s2:.4f}, {hi_s2:.4f}]")
        print(f"  mAP50    : {np.mean(maps):.4f} ± {np.std(maps):.4f}  "
              f"95% CI [{lo_map:.4f}, {hi_map:.4f}]")
    print('='*74)
