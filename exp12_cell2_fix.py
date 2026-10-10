# ════════════════════════════════════════════════════════
# CELL 2A  |  DIAGNOSTIC — Run this FIRST to find CSV paths
# Paste and run this cell before Cell 2
# ════════════════════════════════════════════════════════
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input/cbis-ddsm-breast-cancer-image-dataset")

print("=== INPUT ROOT ===")
print(f"Exists: {INPUT_ROOT.exists()}")
if INPUT_ROOT.exists():
    for item in sorted(INPUT_ROOT.iterdir()):
        print(f"  {item.name}/")
        if item.is_dir():
            for sub in sorted(item.iterdir())[:8]:
                print(f"    {sub.name}")

print("\n=== ALL CSV FILES (recursive search) ===")
all_csvs = sorted(INPUT_ROOT.rglob("*.csv"))
for f in all_csvs:
    print(f"  {f.relative_to(INPUT_ROOT)}")

print(f"\nTotal CSVs found: {len(all_csvs)}")


# ════════════════════════════════════════════════════════
# CELL 2  |  Dataset Preparation — CALC ONLY (FIXED)
# Replace your previous Cell 2 with this entire block
# ════════════════════════════════════════════════════════
import cv2, yaml, shutil, hashlib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Paths & config ─────────────────────────────────────
INPUT_ROOT  = Path("/kaggle/input/cbis-ddsm-breast-cancer-image-dataset")
WORK_DIR    = Path("/kaggle/working/exp12_yolov8l_calc")
RUNS_DIR    = WORK_DIR / "runs"
JPEG_ROOT   = INPUT_ROOT / "jpeg"
SEED        = 42
CLASS_NAMES = ["benign_calc", "malignant_calc"]
CLASS_MAP   = {"BENIGN": 0, "BENIGN_WITHOUT_CALLBACK": 0, "MALIGNANT": 1}
MIN_BBOX_PX = 8
MIN_AREA    = 1e-4

for d in [WORK_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Robust CSV discovery (recursive, no hardcoded subdir) ─
all_csvs = sorted(INPUT_ROOT.rglob("*.csv"))
print(f"Found {len(all_csvs)} CSV files:")
for f in all_csvs:
    print(f"  {f.relative_to(INPUT_ROOT)}")

calc_train_csv = next((f for f in all_csvs if "calc" in f.name.lower() and "train" in f.name.lower()), None)
calc_test_csv  = next((f for f in all_csvs if "calc" in f.name.lower() and "test"  in f.name.lower()), None)

print(f"\nCalc train CSV : {calc_train_csv}")
print(f"Calc test  CSV : {calc_test_csv}")
assert calc_train_csv and calc_test_csv, "Calc CSVs not found! Check the dataset is attached."

# ── Load and combine ────────────────────────────────────
def load_annotations():
    dfs = []
    for fp in [calc_train_csv, calc_test_csv]:
        df = pd.read_csv(fp)
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(r"[\s\-]+", "_", regex=True))
        dfs.append(df)
        print(f"  Loaded: {fp.name} ({len(df)} rows)")

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined["pathology"] = combined["pathology"].str.strip().str.upper()
    print(f"\nTotal calc records: {len(combined)}")
    print(combined["pathology"].value_counts().to_string())
    return combined

df = load_annotations()

# ── Build 3-tier JPEG index ─────────────────────────────
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
    if not isinstance(csv_path, str) or not csv_path.strip():
        return None
    rel     = csv_path.replace("\\", "/").lstrip("/")
    rel_jpg = rel.replace(".dcm", ".jpg")
    c = JPEG_ROOT / rel_jpg
    if c.exists():
        return c
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
    if pid and view:
        key2 = f"{pid}_{view}"
        if key2 in idx["level2"]:
            hits = sorted(idx["level2"][key2])
            if hits: return hits[0]
    return None

jpeg_idx = build_jpeg_index()

# ── Column detection ────────────────────────────────────
def find_col(cols, *must_have, exclude=()):
    for c in cols:
        if all(k in c for k in must_have) and not any(e in c for e in exclude):
            return c
    return None

cols     = list(df.columns)
img_col  = find_col(cols, "image", "file", "path", exclude=("cropped","roi","mask"))
mask_col = find_col(cols, "roi", "mask") or find_col(cols, "mask")
pid_col  = find_col(cols, "patient") or "patient_id"
print(f"\nColumns: img='{img_col}'  mask='{mask_col}'  patient='{pid_col}'")
print(f"All columns: {cols}")
assert img_col and mask_col, f"Column detection failed! See all columns above."

# ── Mask → YOLO bboxes ──────────────────────────────────
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

# ── Build records ────────────────────────────────────────
records = []
n_skip  = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building records"):
    pathology = str(row.get("pathology","")).strip().upper()
    if pathology not in CLASS_MAP:
        n_skip += 1; continue
    img_p  = resolve(str(row[img_col]),  jpeg_idx)
    mask_p = resolve(str(row[mask_col]), jpeg_idx)
    if img_p is None or mask_p is None:
        n_skip += 1; continue
    records.append({
        "img_path":   str(img_p),
        "mask_path":  str(mask_p),
        "class_id":   CLASS_MAP[pathology],
        "patient_id": str(row.get(pid_col, f"UNK_{len(records)}")),
    })

rec_df = pd.DataFrame(records)
print(f"\nValid records: {len(rec_df)}  (skipped {n_skip})")
print(rec_df["class_id"].value_counts().rename(lambda x: CLASS_NAMES[x]).to_string())

# ── Patient-level 70 / 15 / 15 split ────────────────────
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

# ── Write YOLO images + labels ───────────────────────────
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

# ── Write data.yaml ──────────────────────────────────────
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
print("Done. Proceed to Cell 3.")
