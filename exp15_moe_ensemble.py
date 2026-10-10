# ════════════════════════════════════════════════════════
# EXP 15 — Architecture-aware MoE Ensemble (EVAL ONLY)
# YOLO11l Mass Expert (Exp 6) + YOLOv8l Calc Expert (Exp 12)
#
# HOW TO SET UP ON KAGGLE:
#   1. Open a new notebook
#   2. Click "Add data" → "Notebook Outputs"
#   3. Add your Exp 6 notebook output (YOLO11l Mass Expert best.pt)
#   4. Add your Exp 12 notebook output (YOLOv8l Calc Expert best.pt)
#   5. Add the CBIS-DDSM dataset (datasets/awsaf49/...)
#   6. Run all 4 cells in order
#
# No training needed — this is inference + evaluation only.
# ════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════
# CELL 1  |  Install
# ════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "ultralytics", "-q", "--upgrade"], check=True)
print("Done. Run Cell 2.")


# ════════════════════════════════════════════════════════
# CELL 2  |  Build 4-class TEST dataset
# Rebuilds Exp 4 style combined dataset (seed=42 → same split)
# Only writes TEST split (skips train/val for speed)
# ════════════════════════════════════════════════════════
import cv2, yaml, shutil, hashlib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

INPUT_ROOT  = Path("/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset")
WORK_DIR    = Path("/kaggle/working/exp15_moe")
JPEG_ROOT   = INPUT_ROOT / "jpeg"
SEED        = 42
CLASS_NAMES = ["benign_mass", "malignant_mass", "benign_calc", "malignant_calc"]
MIN_BBOX_PX = 8
MIN_AREA    = 1e-4

WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── Find all CSVs ───────────────────────────────────────
all_csvs = sorted(INPUT_ROOT.rglob("*.csv"))
mass_train_csv = next((f for f in all_csvs if "mass" in f.name.lower() and "train" in f.name.lower()), None)
mass_test_csv  = next((f for f in all_csvs if "mass" in f.name.lower() and "test"  in f.name.lower()), None)
calc_train_csv = next((f for f in all_csvs if "calc" in f.name.lower() and "train" in f.name.lower()), None)
calc_test_csv  = next((f for f in all_csvs if "calc" in f.name.lower() and "test"  in f.name.lower()), None)
print(f"Mass CSVs: {mass_train_csv is not None}, Calc CSVs: {calc_train_csv is not None}")

# ── Load & combine all 4 types ──────────────────────────
def load_combined():
    records = []
    for fp, cls_offset, label in [
        (mass_train_csv, 0, "mass"), (mass_test_csv, 0, "mass"),
        (calc_train_csv, 2, "calc"), (calc_test_csv,  2, "calc"),
    ]:
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-]+","_",regex=True)
        df["pathology"] = df["pathology"].str.strip().str.upper()
        for _, row in df.iterrows():
            p = str(row.get("pathology","")).strip().upper()
            if p in ("BENIGN","BENIGN_WITHOUT_CALLBACK"):
                records.append({"pathology_raw": p, "class_id": cls_offset + 0,
                                 "lesion": label, "_row": row, "_df_cols": list(df.columns)})
            elif p == "MALIGNANT":
                records.append({"pathology_raw": p, "class_id": cls_offset + 1,
                                 "lesion": label, "_row": row, "_df_cols": list(df.columns)})
    return records

raw_records = load_combined()
print(f"Raw records: {len(raw_records)}")

# ── Build JPEG index ────────────────────────────────────
def build_jpeg_index():
    idx = {"level0": {}, "level1": {}, "level2": {}}
    for p in JPEG_ROOT.rglob("*.jpg"):
        parts = p.parts
        idx["level0"].setdefault(p.parent.name, []).append(p)
        if len(parts) >= 3:
            idx["level1"].setdefault(f"{parts[-3]}/{parts[-2]}", []).append(p)
        pid  = next((x for x in parts if x.startswith("P_")), None)
        view = next((x for x in parts if any(v in x for v in ("_CC","_MLO","CC","MLO"))), None)
        if pid and view:
            idx["level2"].setdefault(f"{pid}_{view}", []).append(p)
    return idx

def resolve(csv_path, idx):
    if not isinstance(csv_path, str) or not csv_path.strip():
        return None
    rel = csv_path.replace("\\","/").lstrip("/").replace(".dcm",".jpg")
    c = JPEG_ROOT / rel
    if c.exists(): return c
    parts = Path(csv_path).parts
    for part in reversed(parts[:-1]):
        if part in idx["level0"]:
            hits = sorted(idx["level0"][part])
            if hits: return hits[0]
    if len(parts) >= 3:
        key = f"{parts[-3]}/{parts[-2]}"
        if key in idx["level1"]:
            hits = sorted(idx["level1"][key])
            if hits: return hits[0]
    pid  = next((p for p in parts if p.startswith("P_")), None)
    view = next((p for p in parts if any(v in p for v in ("_CC","_MLO","LEFT","RIGHT","CC","MLO"))), None)
    if pid and view and f"{pid}_{view}" in idx["level2"]:
        hits = sorted(idx["level2"][f"{pid}_{view}"])
        if hits: return hits[0]
    return None

jpeg_idx = build_jpeg_index()

def find_col(cols, *must_have, exclude=()):
    for c in cols:
        if all(k in c for k in must_have) and not any(e in c for e in exclude):
            return c
    return None

# ── Build resolved records ──────────────────────────────
def mask_to_bboxes(mask_path, img_w, img_h):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: return []
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    if binary.max() == 0: return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_BBOX_PX or h < MIN_BBOX_PX: continue
        if (w * h) / (img_w * img_h) < MIN_AREA: continue
        xc = np.clip((x + w/2) / img_w, 0.0, 1.0)
        yc = np.clip((y + h/2) / img_h, 0.0, 1.0)
        bboxes.append((xc, yc, np.clip(w/img_w,0,1), np.clip(h/img_h,0,1)))
    return bboxes

records = []
n_skip = 0
for rec in tqdm(raw_records, desc="Resolving paths"):
    row  = rec["_row"]
    cols = rec["_df_cols"]
    img_col  = find_col(cols, "image", "file", "path", exclude=("cropped","roi","mask"))
    mask_col = find_col(cols, "roi", "mask") or find_col(cols, "mask")
    pid_col  = find_col(cols, "patient") or "patient_id"
    if not img_col or not mask_col:
        n_skip += 1; continue
    img_p  = resolve(str(row[img_col]),  jpeg_idx)
    mask_p = resolve(str(row[mask_col]), jpeg_idx)
    if img_p is None or mask_p is None:
        n_skip += 1; continue
    records.append({
        "img_path":   str(img_p),
        "mask_path":  str(mask_p),
        "class_id":   rec["class_id"],
        "patient_id": str(row.get(pid_col, f"UNK_{len(records)}")),
    })

rec_df = pd.DataFrame(records)
print(f"\nValid records: {len(rec_df)}  (skipped {n_skip})")
for i, name in enumerate(CLASS_NAMES):
    n = (rec_df["class_id"] == i).sum()
    print(f"  {name}: {n}")

# ── Patient-level 70/15/15 split (same seed as Exp 4) ──
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, rest_idx = next(gss1.split(rec_df, groups=rec_df["patient_id"].values))
rest_df  = rec_df.iloc[rest_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["patient_id"].values))
test_df = rest_df.iloc[test_idx].reset_index(drop=True)

print(f"\nTest split: {len(test_df)} records, {test_df['patient_id'].nunique()} patients")
for i, name in enumerate(CLASS_NAMES):
    n = (test_df["class_id"] == i).sum()
    print(f"  {name}: {n}")

# ── Write TEST images + labels only ────────────────────
img_out = WORK_DIR / "images" / "test"
lbl_out = WORK_DIR / "labels" / "test"
img_out.mkdir(parents=True, exist_ok=True)
lbl_out.mkdir(parents=True, exist_ok=True)

written = 0; skipped = 0
for img_str, group in tqdm(test_df.groupby("img_path"), desc="Writing test set"):
    img = cv2.imread(img_str)
    if img is None: skipped += 1; continue
    h, w = img.shape[:2]
    lines = []
    for _, row in group.iterrows():
        for (xc, yc, bw, bh) in mask_to_bboxes(Path(row["mask_path"]), w, h):
            lines.append(f"{int(row['class_id'])} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    if not lines: skipped += 1; continue
    stem = f"{Path(img_str).stem}_{hashlib.md5(img_str.encode()).hexdigest()[:8]}"
    shutil.copy2(img_str, img_out / f"{stem}.jpg")
    (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    written += 1

print(f"\nTest images written: {written}  skipped: {skipped}")

# ── Write data.yaml (needed for reference) ─────────────
DATA_YAML = WORK_DIR / "data.yaml"
yaml_cfg = {
    "path":  str(WORK_DIR),
    "train": "images/test",   # placeholder — not used
    "val":   "images/test",
    "test":  "images/test",
    "nc":    4,
    "names": CLASS_NAMES,
}
with open(DATA_YAML, "w") as f:
    yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)

print(f"data.yaml written. Proceed to Cell 3.")


# ════════════════════════════════════════════════════════
# CELL 3  |  Find Weights + Run MoE Inference
# ════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "ultralytics", "torchmetrics", "-q", "--upgrade"], check=True)

import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm.auto import tqdm

# ── Find weights ────────────────────────────────────────
# Mass expert: search for Exp 6 (YOLO11l) or Exp 11 (YOLOv8l) weights
# Attach your Exp 6 notebook output as a dataset in Kaggle
# Path pattern: /kaggle/input/{your-exp6-output}/...
MASS_PT = None
CALC_PT = None

for search_root in [Path("/kaggle/input"), Path("/kaggle/working")]:
    if not search_root.exists(): continue
    for pt in search_root.rglob("best.pt"):
        name = str(pt).lower()
        if "mass" in name and MASS_PT is None:
            MASS_PT = pt
        if "calc" in name and CALC_PT is None:
            CALC_PT = pt

print(f"Mass model : {MASS_PT}")
print(f"Calc model : {CALC_PT}")

if MASS_PT is None or CALC_PT is None:
    print("\n⚠  Weights not found. Check that you attached Exp 6 and Exp 12 outputs.")
    print("   If running on Exp 12's account, CALC_PT should auto-resolve.")
    print("   For MASS_PT, attach the Exp 6 notebook output as a dataset.\n")
    raise FileNotFoundError("Set MASS_PT and CALC_PT paths manually below:")
    # MANUAL OVERRIDE — uncomment and set paths if auto-search fails:
    # MASS_PT = Path("/kaggle/input/YOUR-EXP6-DATASET/exp6_yolo11l_mass/runs/.../best.pt")
    # CALC_PT = Path("/kaggle/input/YOUR-EXP12-DATASET/exp12_yolov8l_calc/runs/.../best.pt")

# ── Load both expert models ─────────────────────────────
print("\nLoading mass expert...")
mass_model = YOLO(str(MASS_PT))
print("Loading calc expert...")
calc_model = YOLO(str(CALC_PT))

# ── Get test images and labels ──────────────────────────
WORK_DIR   = Path("/kaggle/working/exp15_moe")
test_img_dir = WORK_DIR / "images" / "test"
test_lbl_dir = WORK_DIR / "labels" / "test"
test_images  = sorted(test_img_dir.glob("*.jpg"))
print(f"\nTest images: {len(test_images)}")

CLASS_NAMES = ["benign_mass", "malignant_mass", "benign_calc", "malignant_calc"]

# ── Run inference — batch predict for speed ─────────────
print("\nRunning mass expert inference...")
img_paths_str = [str(p) for p in test_images]
mass_results = mass_model.predict(
    img_paths_str, imgsz=1024, device=0, verbose=False, conf=0.001, iou=0.6, batch=4
)

print("Running calc expert inference...")
calc_results = calc_model.predict(
    img_paths_str, imgsz=1024, device=0, verbose=False, conf=0.001, iou=0.6, batch=4
)

# ── Build predictions + targets for torchmetrics ────────
preds_list   = []
targets_list = []

for i, img_path in enumerate(tqdm(test_images, desc="Collecting predictions")):
    lbl_path = test_lbl_dir / f"{img_path.stem}.txt"
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    # Ground truth
    gt_boxes, gt_labels = [], []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().split("\n"):
            if not line.strip(): continue
            cls, xc, yc, bw, bh = map(float, line.split())
            x1 = (xc - bw/2) * W; y1 = (yc - bh/2) * H
            x2 = (xc + bw/2) * W; y2 = (yc + bh/2) * H
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(int(cls))

    targets_list.append({
        "boxes":  torch.tensor(gt_boxes,  dtype=torch.float32) if gt_boxes  else torch.zeros((0, 4)),
        "labels": torch.tensor(gt_labels, dtype=torch.int64)   if gt_labels else torch.zeros(0, dtype=torch.int64),
    })

    # Combined predictions
    all_boxes, all_scores, all_classes = [], [], []

    # Mass predictions (class 0=benign_mass, 1=malignant_mass — keep as-is)
    m = mass_results[i]
    if m.boxes is not None and len(m.boxes) > 0:
        for j in range(len(m.boxes)):
            x1, y1, x2, y2 = m.boxes.xyxy[j].cpu().numpy()
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(float(m.boxes.conf[j].cpu()))
            all_classes.append(int(m.boxes.cls[j].cpu()))   # 0 or 1

    # Calc predictions (class 0→2=benign_calc, 1→3=malignant_calc)
    c = calc_results[i]
    if c.boxes is not None and len(c.boxes) > 0:
        for j in range(len(c.boxes)):
            x1, y1, x2, y2 = c.boxes.xyxy[j].cpu().numpy()
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(float(c.boxes.conf[j].cpu()))
            all_classes.append(int(c.boxes.cls[j].cpu()) + 2)  # 0→2, 1→3

    preds_list.append({
        "boxes":  torch.tensor(all_boxes,   dtype=torch.float32) if all_boxes   else torch.zeros((0, 4)),
        "scores": torch.tensor(all_scores,  dtype=torch.float32) if all_scores  else torch.zeros(0),
        "labels": torch.tensor(all_classes, dtype=torch.int64)   if all_classes else torch.zeros(0, dtype=torch.int64),
    })

print(f"\nCollected {len(preds_list)} prediction sets. Proceed to Cell 4.")


# ════════════════════════════════════════════════════════
# CELL 4  |  Compute mAP and Print Final Results
# ════════════════════════════════════════════════════════
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np

CLASS_NAMES = ["benign_mass", "malignant_mass", "benign_calc", "malignant_calc"]

# mAP50
metric50 = MeanAveragePrecision(
    iou_type="bbox",
    iou_thresholds=[0.50],
    class_metrics=True,
)
metric50.update(preds_list, targets_list)
res50 = metric50.compute()

# mAP50-95
metric_full = MeanAveragePrecision(
    iou_type="bbox",
    iou_thresholds=None,   # default: 0.50:0.05:0.95
    class_metrics=True,
)
metric_full.update(preds_list, targets_list)
res_full = metric_full.compute()

map50     = float(res50["map_50"].item())
map5095   = float(res_full["map"].item())
per_class_ap50   = res50.get("map_per_class",   None)
per_class_ap5095 = res_full.get("map_per_class", None)

print("\n" + "═"*62)
print("EXP 15 — Architecture-aware MoE — 4-CLASS TEST RESULTS")
print(f"  Mass expert: {MASS_PT.parent.parent.name}")
print(f"  Calc expert: {CALC_PT.parent.parent.name}")
print("═"*62)
print(f"mAP50:     {map50:.4f}")
print(f"mAP50-95:  {map5095:.4f}")
print()
print("Per-class AP50:")
if per_class_ap50 is not None:
    for i, name in enumerate(CLASS_NAMES):
        ap = float(per_class_ap50[i].item()) if i < len(per_class_ap50) else 0
        print(f"  {name:22s}  AP50={ap:.4f}")
print()
print("Per-class AP50-95:")
if per_class_ap5095 is not None:
    for i, name in enumerate(CLASS_NAMES):
        ap = float(per_class_ap5095[i].item()) if i < len(per_class_ap5095) else 0
        print(f"  {name:22s}  AP50-95={ap:.4f}")
print("═"*62)

# ── Comparison table ────────────────────────────────────
print("\nCOMPARISON vs previous MoE experiments:")
print(f"  Exp 10 (YOLO11l Mass + YOLO11l Calc):  mAP50 = 0.2582")
print(f"  Exp 15 (YOLO11l Mass + YOLOv8l Calc):  mAP50 = {map50:.4f}  ← this run")
delta = (map50 - 0.2582) / 0.2582 * 100
print(f"  Improvement: {delta:+.1f}%")
print("\n>>> IMMEDIATELY: File → Save Version → Quick Save <<<")
