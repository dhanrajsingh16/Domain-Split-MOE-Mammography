#!/usr/bin/env python3
"""
CMMD zero-shot evaluation: image-level detection sensitivity
No bounding boxes in CMMD — evaluates whether model fires the correct class
on each image given patient-level ground truth (abnormality + classification).
Metric: per-class image-level sensitivity (conf > 0.25 threshold)
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE     = Path("/home/modss/Desktop/dhanraj")
CMMD_DIR = Path("/home/modss/Desktop/dhanraj/cmmd_eval")
DB_DIR   = CMMD_DIR / "TheChineseMammographyDatabase"
SEEDS    = [42, 1, 123]

MASS_W = {s: BASE/f"canonical_mass_clean/runs/mass_clean_seed{s}/weights/best.pt" for s in SEEDS}
CALC_W = {s: BASE/f"canonical_calc_clean/runs/calc_clean_seed{s}/weights/best.pt"  for s in SEEDS}
UNI_W  = {s: BASE/f"canonical_4class/runs/universal_seed{s}/weights/best.pt"       for s in SEEDS}

CONF     = 0.25   # confidence threshold for "model detected something"
NMS_IOU  = 0.45
IMG_SIZE = 1280

CLASS_NAMES = ["benign_mass", "malig_mass", "benign_calc", "malig_calc"]


# ── STEP 1: BUILD IMAGE → CLASS MAP ──────────────────────────────────────────
def build_image_label_map():
    """
    Returns list of (png_path, cls_id) for all convertible DICOM images.
    cls_id: 0=benign_mass, 1=malig_mass, 2=benign_calc, 3=malig_calc
    """
    xlsx = DB_DIR / "CMMD_clinicaldata_revision.xlsx"
    df = pd.read_excel(xlsx)
    print(f"Clinical data: {len(df)} rows, columns: {list(df.columns)}")

    # Build patient → class mapping
    patient_cls = {}
    for _, row in df.iterrows():
        pid     = str(row["ID1"]).strip()
        abn     = str(row["abnormality"]).lower()
        cls_str = str(row["classification"]).lower()
        is_calc = "calc" in abn
        is_mal  = "malignant" in cls_str
        if is_calc:
            cls = 3 if is_mal else 2
        else:
            cls = 1 if is_mal else 0
        patient_cls[pid] = cls   # if duplicate ID, last row wins (same label anyway)

    print(f"Patient labels loaded: {len(patient_cls)}")
    for c, name in enumerate(CLASS_NAMES):
        count = sum(1 for v in patient_cls.values() if v == c)
        print(f"  {name}: {count} patients")

    # Find DICOM files and convert to PNG
    png_dir = CMMD_DIR / "png"
    png_dir.mkdir(exist_ok=True)

    try:
        import pydicom
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pydicom", "-q"])
        import pydicom

    cmmd_root  = DB_DIR / "CMMD"
    image_list = []   # (png_path, cls_id)
    skipped    = 0
    convert_errors = 0

    for pid, cls in patient_cls.items():
        patient_dir = cmmd_root / pid
        if not patient_dir.exists():
            skipped += 1
            continue
        dcm_files = sorted(patient_dir.rglob("*.dcm"))
        for dcm_path in dcm_files:
            stem     = f"{pid}_{dcm_path.stem}"
            png_path = png_dir / f"{stem}.png"
            if not png_path.exists():
                try:
                    ds  = pydicom.dcmread(str(dcm_path))
                    arr = ds.pixel_array.astype(np.float32)
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255
                    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                    img.save(png_path)
                except Exception as e:
                    convert_errors += 1
                    continue
            image_list.append((png_path, cls))

    print(f"\nTotal images ready: {len(image_list)}"
          f"  (skipped {skipped} missing dirs, {convert_errors} convert errors)")
    for c, name in enumerate(CLASS_NAMES):
        count = sum(1 for _, cl in image_list if cl == c)
        print(f"  {name}: {count} images")

    return image_list


# ── STEP 2: INFERENCE ─────────────────────────────────────────────────────────
def predict_classes(model, img_path, cls_offset=0):
    """Returns set of detected class IDs (shifted by cls_offset) above CONF."""
    try:
        res = model.predict(str(img_path), conf=CONF, iou=NMS_IOU,
                            imgsz=IMG_SIZE, verbose=False)[0]
    except Exception:
        return set()
    classes = set()
    if res.boxes is not None and len(res.boxes):
        for cl in res.boxes.cls.cpu().numpy().astype(int):
            classes.add(cl + cls_offset)
    return classes


def ds_moe_classes(mass_mdl, calc_mdl, img_path):
    return predict_classes(mass_mdl, img_path, 0) | predict_classes(calc_mdl, img_path, 2)


# ── STEP 3: EVALUATE ──────────────────────────────────────────────────────────
def evaluate(image_list, seed):
    from ultralytics import YOLO
    print(f"\nSeed {seed} — loading models ...")
    uni_mdl  = YOLO(str(UNI_W[seed]))
    mass_mdl = YOLO(str(MASS_W[seed]))
    calc_mdl = YOLO(str(CALC_W[seed]))

    uni_hit = [0] * 4
    moe_hit = [0] * 4
    cls_cnt = [0] * 4

    for i, (png_path, gt_cls) in enumerate(image_list):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(image_list)}]", end="\r")

        uni_preds = predict_classes(uni_mdl, png_path, 0)
        moe_preds = ds_moe_classes(mass_mdl, calc_mdl, png_path)

        cls_cnt[gt_cls] += 1
        if gt_cls in uni_preds:
            uni_hit[gt_cls] += 1
        if gt_cls in moe_preds:
            moe_hit[gt_cls] += 1

    print()
    return uni_hit, moe_hit, cls_cnt


def sensitivity(hits, counts):
    return [h / c if c > 0 else 0.0 for h, c in zip(hits, counts)]


def mean_sens(sens_list):
    valid = [s for s in sens_list if s >= 0]
    return np.mean(valid) if valid else 0.0


# ── ENTRY ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    image_list = build_image_label_map()
    if not image_list:
        sys.exit("No images found. Check CMMD directory structure.")

    all_uni_sens = []
    all_moe_sens = []

    for seed in SEEDS:
        uni_hit, moe_hit, cls_cnt = evaluate(image_list, seed)
        u_sens = sensitivity(uni_hit, cls_cnt)
        m_sens = sensitivity(moe_hit, cls_cnt)
        all_uni_sens.append(u_sens)
        all_moe_sens.append(m_sens)

        print(f"\nSeed {seed} (conf>{CONF}):")
        print(f"  {'Class':<16} {'N':>6} {'Universal':>11} {'DS-MoE':>10}")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  {name:<16} {cls_cnt[c]:>6} {u_sens[c]:>11.4f} {m_sens[c]:>10.4f}")
        print(f"  {'Mean':<16} {'':>6} {mean_sens(u_sens):>11.4f} {mean_sens(m_sens):>10.4f}")

    # Summary
    print(f"\n{'='*62}")
    print("CMMD ZERO-SHOT IMAGE-LEVEL SENSITIVITY  (mean ± std, 3 seeds)")
    print(f"Metric: P(correct class predicted | image contains that class)")
    print(f"Confidence threshold: {CONF}")
    print(f"{'='*62}")
    print(f"{'Class':<16} {'Universal':>22} {'DS-MoE':>22}")
    print("-" * 62)
    for c, name in enumerate(CLASS_NAMES):
        u_vals = [s[c] for s in all_uni_sens]
        m_vals = [s[c] for s in all_moe_sens]
        print(f"{name:<16} {np.mean(u_vals):>8.4f} ± {np.std(u_vals):.4f}"
              f"    {np.mean(m_vals):>8.4f} ± {np.std(m_vals):.4f}")
    u_overall = [mean_sens(s) for s in all_uni_sens]
    m_overall = [mean_sens(s) for s in all_moe_sens]
    print("-" * 62)
    print(f"{'Mean':<16} {np.mean(u_overall):>8.4f} ± {np.std(u_overall):.4f}"
          f"    {np.mean(m_overall):>8.4f} ± {np.std(m_overall):.4f}")
    print('=' * 62)
