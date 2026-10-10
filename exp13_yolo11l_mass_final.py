# ════════════════════════════════════════════════════════
# EXP 13 — YOLO11l Mass Expert FINAL
# Goal: Beat Exp 6 (mAP50 = 0.447)
# Changes vs Exp 6: patience=50, epochs=200, batch=auto
# Backbone: YOLO11l (proven best for masses)
# ════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════
# CELL 1  |  Install
# ════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "ultralytics", "-q", "--upgrade"], check=True)
print("Done. Run Cell 2.")


# ════════════════════════════════════════════════════════
# CELL 2  |  Dataset Preparation — MASS ONLY
# ════════════════════════════════════════════════════════
import cv2, yaml, shutil, hashlib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Auto-detect dataset path (handles both Kaggle path styles)
INPUT_ROOT = None
for candidate in [
    Path("/kaggle/input/cbis-ddsm-breast-cancer-image-dataset"),
    Path("/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset"),
]:
    if (candidate / "jpeg").exists():
        INPUT_ROOT = candidate
        break
if INPUT_ROOT is None:
    raise RuntimeError("CBIS-DDSM dataset not found. Add it via 'Add Data' → search 'cbis-ddsm'.")
print(f"Dataset root: {INPUT_ROOT}")

WORK_DIR    = Path("/kaggle/working/exp13_yolo11l_mass_final")
RUNS_DIR    = WORK_DIR / "runs"
JPEG_ROOT   = INPUT_ROOT / "jpeg"
SEED        = 42
CLASS_NAMES = ["benign_mass", "malignant_mass"]
CLASS_MAP   = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}
MIN_BBOX_PX = 8
MIN_AREA    = 1e-4

for d in [WORK_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

all_csvs = sorted(INPUT_ROOT.rglob("*.csv"))
print(f"Found {len(all_csvs)} CSV files:")
for f in all_csvs:
    print(f"  {f.relative_to(INPUT_ROOT)}")

mass_train_csv = next((f for f in all_csvs if "mass" in f.name.lower() and "train" in f.name.lower()), None)
mass_test_csv  = next((f for f in all_csvs if "mass" in f.name.lower() and "test"  in f.name.lower()), None)
print(f"\nMass train CSV : {mass_train_csv}")
print(f"Mass test  CSV : {mass_test_csv}")
assert mass_train_csv and mass_test_csv, "Mass CSVs not found!"

def load_annotations():
    dfs = []
    for fp in [mass_train_csv, mass_test_csv]:
        df = pd.read_csv(fp)
        df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
        dfs.append(df)
        print(f"  Loaded: {fp.name} ({len(df)} rows)")
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined["pathology"] = combined["pathology"].str.strip().str.upper()
    print(f"\nTotal mass records: {len(combined)}")
    print(combined["pathology"].value_counts().to_string())
    return combined

df = load_annotations()

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
    n = sum(len(v) for v in idx["level0"].values())
    print(f"JPEG index: {n} images in {len(idx['level0'])} folders")
    return idx

def resolve(csv_path, idx):
    if not isinstance(csv_path, str) or not csv_path.strip(): return None
    rel = csv_path.replace("\\", "/").lstrip("/").replace(".dcm", ".jpg")
    c = JPEG_ROOT / rel
    if c.exists(): return c
    parts = Path(rel).parts
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
        if all(k in c for k in must_have) and not any(e in c for e in exclude): return c
    return None

cols     = list(df.columns)
img_col  = find_col(cols, "image", "file", "path", exclude=("cropped","roi","mask"))
mask_col = find_col(cols, "roi", "mask") or find_col(cols, "mask")
pid_col  = find_col(cols, "patient") or "patient_id"
print(f"\nColumns: img='{img_col}'  mask='{mask_col}'  patient='{pid_col}'")
assert img_col and mask_col, "Column detection failed!"

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
n_skip  = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building records"):
    pathology = str(row.get("pathology","")).strip().upper()
    if pathology not in CLASS_MAP: n_skip += 1; continue
    img_p  = resolve(str(row[img_col]),  jpeg_idx)
    mask_p = resolve(str(row[mask_col]), jpeg_idx)
    if img_p is None or mask_p is None: n_skip += 1; continue
    records.append({
        "img_path":   str(img_p),
        "mask_path":  str(mask_p),
        "class_id":   CLASS_MAP[pathology],
        "patient_id": str(row.get(pid_col, f"UNK_{len(records)}")),
    })

rec_df = pd.DataFrame(records)
print(f"\nValid records: {len(rec_df)}  (skipped {n_skip})")
print(rec_df["class_id"].value_counts().rename(lambda x: CLASS_NAMES[x]).to_string())

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, rest_idx = next(gss1.split(rec_df, groups=rec_df["patient_id"].values))
train_df = rec_df.iloc[train_idx].reset_index(drop=True)
rest_df  = rec_df.iloc[rest_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["patient_id"].values))
val_df  = rest_df.iloc[val_idx].reset_index(drop=True)
test_df = rest_df.iloc[test_idx].reset_index(drop=True)

print("\nSplit summary:")
for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
    dist = dict(sdf["class_id"].value_counts().rename(lambda x: CLASS_NAMES[x]))
    print(f"  {name:5s}: {len(sdf):4d} images | {sdf['patient_id'].nunique():3d} patients | {dist}")

def write_split(split_name, split_df):
    img_out = WORK_DIR / "images" / split_name
    lbl_out = WORK_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    written = 0; skipped = 0
    for img_str, group in tqdm(split_df.groupby("img_path"), desc=f"Writing {split_name}"):
        img = cv2.imread(img_str)
        if img is None: skipped += 1; continue
        h, w  = img.shape[:2]
        lines = []
        for _, row in group.iterrows():
            for (xc, yc, bw, bh) in mask_to_bboxes(Path(row["mask_path"]), w, h):
                lines.append(f"{int(row['class_id'])} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        if not lines: skipped += 1; continue
        stem = f"{Path(img_str).stem}_{hashlib.md5(img_str.encode()).hexdigest()[:8]}"
        shutil.copy2(img_str, img_out / f"{stem}.jpg")
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1
    print(f"  [{split_name}] written={written}  skipped={skipped}")
    return written

n_tr = write_split("train", train_df)
n_va = write_split("val",   val_df)
n_te = write_split("test",  test_df)

DATA_YAML = WORK_DIR / "data.yaml"
yaml_cfg  = {
    "path":  str(WORK_DIR),
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc":    len(CLASS_NAMES),
    "names": CLASS_NAMES,
}
with open(DATA_YAML, "w") as f:
    yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False)

print(f"\ndata.yaml : {DATA_YAML}")
print(f"Dataset   : train={n_tr}  val={n_va}  test={n_te}")
print("Done. Run Cell 3.")


# ════════════════════════════════════════════════════════
# CELL 3  |  Training — YOLO11l Mass Final  (v4 — PUSH FOR 0.50)
# Strategy to cross mAP50 0.50:
#   imgsz 1024 (max res -> better localization -> higher mAP)
#   batch 4   (minimum HEALTHY BatchNorm; batch=1 was fatal)
#   expandable_segments -> lets batch=4 fit the T4 (Kaggle's own
#   OOM message recommended this exact flag).
#   mosaic 0.0 (proven: mass HATES mosaic; baseline used 0.0).
# If Cell 3 OOMs in the first seconds -> change batch=4 to batch=2.
# ════════════════════════════════════════════════════════
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduces fragmentation -> batch=4 fits @1024

from ultralytics import YOLO
from pathlib import Path

WORK_DIR  = Path("/kaggle/working/exp13_yolo11l_mass_final")
RUNS_DIR  = WORK_DIR / "runs"
DATA_YAML = WORK_DIR / "data.yaml"

model = YOLO("yolo11l.pt")
results = model.train(
    data          = str(DATA_YAML),
    epochs        = 150,
    patience      = 40,
    imgsz         = 1024,     # v4: max resolution (matches 0.447 baseline)
    batch         = 8,        # v4: bumped 4->8 (batch=4 used only 6.43G -> headroom). Better BatchNorm.
    device        = 0,
    workers       = 2,
    project       = str(RUNS_DIR),
    name          = "yolo11l_mass_final",
    exist_ok      = True,
    optimizer     = "AdamW",
    lr0           = 0.0008,   # v4: slightly lower for stable small-batch training
    lrf           = 0.01,
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 5,
    cos_lr        = True,
    nbs           = 64,       # gradient accumulation -> effective batch 64 (stabilises optimizer)
    mosaic        = 0.0,      # KEY: mass HATES mosaic (proven). Baseline used 0.0.
    close_mosaic  = 0,
    copy_paste    = 0.0,
    mixup         = 0.0,
    flipud        = 0.5,
    fliplr        = 0.5,
    degrees       = 0.0,
    scale         = 0.5,
    translate     = 0.1,
    seed          = 42,
    amp           = True,
    val           = True,
    plots         = True,
    verbose       = True,
)

best_pts = sorted(RUNS_DIR.glob("yolo11l_mass_final*/weights/best.pt"))
if not best_pts: best_pts = sorted(RUNS_DIR.glob("*/weights/best.pt"))
BEST_PT = best_pts[-1]
print(f"\nDone — val mAP50 = {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"Best weights: {BEST_PT}")
print("Run Cell 4 → then IMMEDIATELY: File → Save Version → Quick Save")


# ════════════════════════════════════════════════════════
# CELL 4  |  Test Evaluation
# ════════════════════════════════════════════════════════
import subprocess
subprocess.run(["pip", "install", "ultralytics", "-q", "--upgrade"], check=True)
import numpy as np
from ultralytics import YOLO
from pathlib import Path

RUNS_DIR  = Path("/kaggle/working/exp13_yolo11l_mass_final/runs")
DATA_YAML = Path("/kaggle/working/exp13_yolo11l_mass_final/data.yaml")

best_pts = sorted(RUNS_DIR.glob("yolo11l_mass_final*/weights/best.pt"))
if not best_pts: best_pts = sorted(RUNS_DIR.glob("*/weights/best.pt"))
BEST_PT = best_pts[-1]
print(f"Loading: {BEST_PT}")

model   = YOLO(str(BEST_PT))
metrics = model.val(data=str(DATA_YAML), split="test",
                    imgsz=1024, batch=4, device=0, workers=2,
                    augment=True)   # TTA: +1-3% mAP at test time (imgsz matches training)

p_arr  = np.array(metrics.box.p).flatten()
r_arr  = np.array(metrics.box.r).flatten()
ap50   = np.array(metrics.box.ap50).flatten()
ap5095 = np.array(metrics.box.ap).flatten()

print("\n" + "═"*62)
print("EXP 13 — YOLO11l MASS FINAL — TEST RESULTS")
print("═"*62)
print(f"mAP50:     {metrics.box.map50:.4f}  (target: 0.480+  excellent: 0.500+)")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print()
for i, name in enumerate(["benign_mass", "malignant_mass"]):
    p  = float(p_arr[i])  if i < len(p_arr)  else 0
    r  = float(r_arr[i])  if i < len(r_arr)  else 0
    a  = float(ap50[i])   if i < len(ap50)   else 0
    a2 = float(ap5095[i]) if i < len(ap5095) else 0
    print(f"  {name:20s}  P={p:.4f}  R={r:.4f}  AP50={a:.4f}  AP50-95={a2:.4f}")
print()
baseline = 0.447
delta    = (metrics.box.map50 - baseline) / baseline * 100
print(f"Baseline (Exp 6):  mAP50 = {baseline}")
print(f"This run:          mAP50 = {metrics.box.map50:.4f}  ({delta:+.1f}%)")
print("═"*62)
print("\n>>> IMMEDIATELY: File → Save Version → Quick Save <<<")
