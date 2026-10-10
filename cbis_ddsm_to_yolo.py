#!/usr/bin/env python3
"""
CBIS-DDSM Mammogram Dataset → YOLO Format
Stage 1: Dataset Preparation for Lesion Detection Research
Dataset : CBIS-DDSM Breast Cancer Image Dataset (Kaggle: awsaf49/cbis-ddsm-breast-cancer-image-dataset)
Platform: Kaggle Notebook
"""

# ══════════════════════════════════════════════════════════════════════════
# CELL 1 — INSTALL / IMPORT
# ══════════════════════════════════════════════════════════════════════════
import os, sys, cv2, yaml, logging, shutil, hashlib, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════
# CELL 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
INPUT_ROOT = Path("/kaggle/input/cbis-ddsm-breast-cancer-image-dataset")
WORK_DIR   = Path("/kaggle/working/yolo_cbis")
CSV_DIR    = INPUT_ROOT / "csv"
JPEG_ROOT  = INPUT_ROOT / "jpeg"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

MIN_BBOX_PX        = 8      # discard bboxes smaller than this in either dimension
MIN_BBOX_AREA_FRAC = 1e-4   # discard bboxes smaller than 0.01 % of image area

# Two-class detection: benign / malignant
CLASS_MAP = {
    "BENIGN":                  0,
    "BENIGN_WITHOUT_CALLBACK": 0,
    "MALIGNANT":               1,
}
CLASS_NAMES = ["benign", "malignant"]

# ══════════════════════════════════════════════════════════════════════════
# CELL 3 — LOGGING
# ══════════════════════════════════════════════════════════════════════════
WORK_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("cbis2yolo")
log.setLevel(logging.DEBUG)
log.handlers.clear()

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
_ch  = logging.StreamHandler(sys.stdout);       _ch.setLevel(logging.INFO);  _ch.setFormatter(_fmt)
_fh  = logging.FileHandler(WORK_DIR / "prep.log", mode="w"); _fh.setLevel(logging.DEBUG); _fh.setFormatter(_fmt)
log.addHandler(_ch)
log.addHandler(_fh)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — INSPECT DATASET STRUCTURE
# ══════════════════════════════════════════════════════════════════════════
def inspect_dataset() -> dict:
    log.info("━" * 60)
    log.info("STEP 1 │ Dataset Inspection")
    log.info("━" * 60)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {INPUT_ROOT}\n"
            "→ Ensure the Kaggle dataset is attached: "
            "awsaf49/cbis-ddsm-breast-cancer-image-dataset"
        )

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    jpegs     = list(JPEG_ROOT.rglob("*.jpg"))

    log.info(f"CSV files   : {len(csv_files)}")
    for f in csv_files:
        log.info(f"  └─ {f.name}")

    log.info(f"JPEG images : {len(jpegs)}")

    # Compact directory tree (3 levels)
    def _tree(root: Path, prefix: str = "", depth: int = 3):
        if depth == 0:
            return
        entries = sorted(root.iterdir())[:6]
        for i, e in enumerate(entries):
            conn = "└─" if i == len(entries) - 1 else "├─"
            log.info(f"{prefix}{conn} {e.name}")
            if e.is_dir():
                ext = "   " if i == len(entries) - 1 else "│  "
                _tree(e, prefix + ext, depth - 1)

    log.info(f"\n  Tree: {INPUT_ROOT}")
    _tree(INPUT_ROOT)

    info = {"csv_count": len(csv_files), "jpeg_count": len(jpegs)}
    log.info(f"\n  Summary: {info}")
    return info


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD LESION ANNOTATIONS FROM CSV
# ══════════════════════════════════════════════════════════════════════════
def load_annotations() -> pd.DataFrame:
    log.info("━" * 60)
    log.info("STEP 2 │ Load Lesion Annotations")
    log.info("━" * 60)

    csv_specs = [
        ("mass_case_description_train_set.csv", "mass"),
        ("mass_case_description_test_set.csv",  "mass"),
        ("calc_case_description_train_set.csv", "calc"),
        ("calc_case_description_test_set.csv",  "calc"),
    ]

    dfs = []
    for fname, atype in csv_specs:
        fp = CSV_DIR / fname
        if not fp.exists():
            # Case-insensitive fallback
            hits = list(CSV_DIR.glob(f"*{fname[:4]}*"))
            if not hits:
                log.warning(f"  MISSING : {fname}")
                continue
            fp = hits[0]
            log.info(f"  Resolved: {fp.name}")

        df = pd.read_csv(fp)
        # Normalise column names
        df.columns = (df.columns
                        .str.strip()
                        .str.lower()
                        .str.replace(r"[\s\-]+", "_", regex=True))
        df["abnormality_source"] = atype
        dfs.append(df)
        log.info(f"  {fp.name}: {len(df):4d} rows")

    if not dfs:
        raise FileNotFoundError(
            "No annotation CSVs found. Verify CSV_DIR path and dataset attachment."
        )

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined["pathology"] = combined["pathology"].str.strip().str.upper()

    log.info(f"\n  Total rows : {len(combined)}")
    log.info(f"  Columns    : {list(combined.columns)}")
    log.info(f"  Pathology  :\n{combined['pathology'].value_counts().to_string()}")

    return combined


# ══════════════════════════════════════════════════════════════════════════
# PATH RESOLUTION HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _build_jpeg_index(jpeg_root: Path) -> dict:
    """
    Three-tier index for robust path resolution:
      level0 : parent_folder_name → [Path]  (series-level lookup)
      level1 : grandparent/parent → [Path]  (study/series lookup)
      level2 : patient_view_key  → [Path]   (P_XXXXX_VIEW lookup)
    """
    log.info("  Building JPEG index …")
    idx = {"level0": {}, "level1": {}, "level2": {}}

    for p in jpeg_root.rglob("*.jpg"):
        parts = p.parts

        # level0: immediate parent
        key0 = p.parent.name
        idx["level0"].setdefault(key0, []).append(p)

        # level1: grandparent/parent
        if len(parts) >= 2:
            key1 = f"{parts[-3]}/{parts[-2]}" if len(parts) >= 3 else parts[-2]
            idx["level1"].setdefault(key1, []).append(p)

        # level2: detect P_XXXXX + CC/MLO pattern
        pid_part  = next((x for x in parts if x.startswith("P_")), None)
        view_part = next((x for x in parts if any(v in x for v in ("_CC", "_MLO", "CC", "MLO"))), None)
        if pid_part and view_part:
            key2 = f"{pid_part}_{view_part}"
            idx["level2"].setdefault(key2, []).append(p)

    total = sum(len(v) for v in idx["level0"].values())
    log.info(f"  Indexed {total} images │ {len(idx['level0'])} level-0 keys")
    return idx


def _resolve(csv_path: str, idx: dict) -> Path | None:
    """
    Map a DICOM-style CSV path string to the actual JPEG file.
    Tries four strategies in order.
    """
    if not isinstance(csv_path, str) or not csv_path.strip():
        return None

    rel     = csv_path.replace("\\", "/").lstrip("/")
    rel_jpg = rel.replace(".dcm", ".jpg")

    # ── Strategy 1: direct substitution ──────────────────────────────────
    candidate = JPEG_ROOT / rel_jpg
    if candidate.exists():
        return candidate

    parts = Path(rel).parts

    # ── Strategy 2: series-folder lookup (innermost matching folder) ──────
    for part in reversed(parts[:-1]):
        if part in idx["level0"]:
            hits = sorted(idx["level0"][part])
            if hits:
                return hits[0]

    # ── Strategy 3: grandparent/parent compound key ───────────────────────
    if len(parts) >= 2:
        key = f"{parts[-3]}/{parts[-2]}" if len(parts) >= 3 else parts[-2]
        if key in idx["level1"]:
            hits = sorted(idx["level1"][key])
            if hits:
                return hits[0]

    # ── Strategy 4: patient_id + view glob ───────────────────────────────
    pid_part  = next((p for p in parts if p.startswith("P_")), None)
    view_part = next((p for p in parts
                      if any(v in p for v in ("_CC", "_MLO", "LEFT", "RIGHT", "CC", "MLO"))), None)
    if pid_part and view_part:
        key2 = f"{pid_part}_{view_part}"
        if key2 in idx["level2"]:
            hits = sorted(idx["level2"][key2])
            if hits:
                return hits[0]

    return None


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — CONVERT ROI MASKS → YOLO BOUNDING BOXES
# ══════════════════════════════════════════════════════════════════════════
def mask_to_bboxes(mask_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """
    Read binary ROI mask → list of (x_c, y_c, w, h) normalised [0,1].
    Handles multiple disconnected ROIs per mask.
    Returns empty list on failure or empty mask.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        log.debug(f"mask_read_fail: {mask_path}")
        return []

    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    if binary.max() == 0:
        log.debug(f"empty_mask: {mask_path}")
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < MIN_BBOX_PX or h < MIN_BBOX_PX:
            continue
        if (w * h) / (img_w * img_h) < MIN_BBOX_AREA_FRAC:
            continue
        x_c = np.clip((x + w / 2) / img_w, 0.0, 1.0)
        y_c = np.clip((y + h / 2) / img_h, 0.0, 1.0)
        n_w = np.clip(w / img_w, 0.0, 1.0)
        n_h = np.clip(h / img_h, 0.0, 1.0)
        bboxes.append((x_c, y_c, n_w, n_h))

    return bboxes


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — BUILD RECORD TABLE
# ══════════════════════════════════════════════════════════════════════════
def build_records(df: pd.DataFrame) -> pd.DataFrame:
    log.info("━" * 60)
    log.info("STEP 4 │ Build Record Table (resolve paths)")
    log.info("━" * 60)

    jpeg_idx = _build_jpeg_index(JPEG_ROOT)

    # Robust column detection
    def _find_col(cols, *must_have, exclude=()):
        for c in cols:
            if all(k in c for k in must_have) and not any(e in c for e in exclude):
                return c
        return None

    cols = list(df.columns)
    img_col  = _find_col(cols, "image", "file", "path", exclude=("cropped", "roi", "mask"))
    mask_col = (_find_col(cols, "roi", "mask") or
                _find_col(cols, "mask"))
    pid_col  = _find_col(cols, "patient") or "patient_id"

    if not img_col or not mask_col:
        log.error(f"Column detection failed. Available: {cols}")
        raise ValueError(
            f"Cannot identify image col ({img_col}) or mask col ({mask_col}).\n"
            f"All columns: {cols}"
        )

    log.info(f"  Image col  : {img_col}")
    log.info(f"  Mask col   : {mask_col}")
    log.info(f"  Patient col: {pid_col}")

    records        = []
    n_bad_pathology = 0
    n_missing_img  = 0
    n_missing_mask = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Resolving", unit="row"):
        pathology = str(row.get("pathology", "")).strip().upper()
        if pathology not in CLASS_MAP:
            n_bad_pathology += 1
            continue

        img_path  = _resolve(str(row[img_col]),  jpeg_idx)
        mask_path = _resolve(str(row[mask_col]), jpeg_idx)

        if img_path is None:
            n_missing_img += 1
            log.debug(f"img_not_found: {row[img_col]}")
            continue
        if mask_path is None:
            n_missing_mask += 1
            log.debug(f"mask_not_found: {row[mask_col]}")
            continue

        records.append({
            "img_path":   str(img_path),
            "mask_path":  str(mask_path),
            "class_id":   CLASS_MAP[pathology],
            "patient_id": str(row.get(pid_col, f"UNK_{len(records)}")),
            "pathology":  pathology,
            "abn_source": str(row.get("abnormality_source", "")),
        })

    rec_df = pd.DataFrame(records)

    log.info(f"\n  Input rows       : {len(df)}")
    log.info(f"  Unknown pathology: {n_bad_pathology}")
    log.info(f"  Image missing    : {n_missing_img}")
    log.info(f"  Mask missing     : {n_missing_mask}")
    log.info(f"  Valid records    : {len(rec_df)}")

    if len(rec_df) == 0:
        raise RuntimeError(
            "Zero valid records built.\n"
            "Likely causes:\n"
            "  1. JPEG_ROOT path is wrong — check INPUT_ROOT\n"
            "  2. CSV column names differ from expected — check log for debug lines\n"
            "  3. Dataset version mismatch"
        )

    class_dist = rec_df["class_id"].value_counts().rename(lambda x: CLASS_NAMES[x])
    log.info(f"\n  Class distribution:\n{class_dist.to_string()}")

    return rec_df


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — PATIENT-LEVEL TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════
def split_records(rec_df: pd.DataFrame) -> dict:
    log.info("━" * 60)
    log.info("STEP 5 │ Patient-Level Train / Val / Test Split")
    log.info("━" * 60)

    test_val_frac = VAL_RATIO + TEST_RATIO

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_val_frac, random_state=SEED)
    train_idx, rest_idx = next(gss1.split(rec_df, groups=rec_df["patient_id"].values))
    train_df = rec_df.iloc[train_idx].reset_index(drop=True)
    rest_df  = rec_df.iloc[rest_idx].reset_index(drop=True)

    test_frac = TEST_RATIO / test_val_frac
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=SEED)
    val_idx2, test_idx2 = next(gss2.split(rest_df, groups=rest_df["patient_id"].values))
    val_df  = rest_df.iloc[val_idx2].reset_index(drop=True)
    test_df = rest_df.iloc[test_idx2].reset_index(drop=True)

    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        cls_dist = dict(sdf["class_id"].value_counts().rename(lambda x: CLASS_NAMES[x]))
        log.info(
            f"  {name:5s}: {len(sdf):4d} rows │ "
            f"{sdf['patient_id'].nunique():3d} patients │ {cls_dist}"
        )

    return {"train": train_df, "val": val_df, "test": test_df}


# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — WRITE YOLO IMAGE + LABEL FILES
# ══════════════════════════════════════════════════════════════════════════
def write_split(split_name: str, split_df: pd.DataFrame) -> dict:
    img_out = WORK_DIR / "images" / split_name
    lbl_out = WORK_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    stats = {"written": 0, "skipped": 0, "total_bboxes": 0,
             "class_counts": {n: 0 for n in CLASS_NAMES}}

    # Group by image path — one mammogram can have multiple lesion rows
    for img_str, group in tqdm(
        split_df.groupby("img_path"),
        desc=f"  Writing {split_name}",
        unit="img"
    ):
        img_path = Path(img_str)
        img = cv2.imread(str(img_path))
        if img is None:
            log.debug(f"cv2_read_fail: {img_path}")
            stats["skipped"] += 1
            continue

        h, w = img.shape[:2]
        label_lines = []

        for _, row in group.iterrows():
            bboxes = mask_to_bboxes(Path(row["mask_path"]), w, h)
            cls_id = int(row["class_id"])
            for (xc, yc, bw, bh) in bboxes:
                label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                stats["class_counts"][CLASS_NAMES[cls_id]] += 1

        if not label_lines:
            stats["skipped"] += 1
            continue

        # Unique stem avoids collisions when the same filename appears across subdirs
        stem = f"{img_path.stem}_{hashlib.md5(img_str.encode()).hexdigest()[:8]}"

        shutil.copy2(img_path, img_out / f"{stem}.jpg")
        (lbl_out / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")

        stats["written"]      += 1
        stats["total_bboxes"] += len(label_lines)

    log.info(
        f"  [{split_name}] written={stats['written']} "
        f"skipped={stats['skipped']} bboxes={stats['total_bboxes']} "
        f"classes={stats['class_counts']}"
    )
    return stats


# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — GENERATE data.yaml
# ══════════════════════════════════════════════════════════════════════════
def write_data_yaml() -> Path:
    log.info("━" * 60)
    log.info("STEP 7 │ Write data.yaml")
    log.info("━" * 60)

    cfg = {
        "path":  str(WORK_DIR),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = WORK_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    log.info(f"  Saved: {yaml_path}")
    log.info(f"  Content:\n{yaml.dump(cfg, default_flow_style=False, sort_keys=False)}")
    return yaml_path


# ══════════════════════════════════════════════════════════════════════════
# STEP 8 — VERIFY LABELS
# ══════════════════════════════════════════════════════════════════════════
def verify_labels() -> bool:
    log.info("━" * 60)
    log.info("STEP 8 │ Label Verification")
    log.info("━" * 60)

    all_ok = True
    report = {}

    for split in ("train", "val", "test"):
        img_dir = WORK_DIR / "images" / split
        lbl_dir = WORK_DIR / "labels" / split

        imgs = {p.stem for p in img_dir.glob("*.jpg")}
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}

        orphan_imgs = imgs - lbls   # images without a label file
        orphan_lbls = lbls - imgs   # labels without an image

        n_lines   = 0
        n_bad     = 0
        cls_count = {i: 0 for i in range(len(CLASS_NAMES))}

        for lp in lbl_dir.glob("*.txt"):
            for line in lp.read_text().strip().splitlines():
                n_lines += 1
                parts = line.split()
                ok = True
                if len(parts) != 5:
                    ok = False
                else:
                    try:
                        cls    = int(parts[0])
                        coords = [float(c) for c in parts[1:]]
                        if cls < 0 or cls >= len(CLASS_NAMES):
                            ok = False
                        if any(c < 0 or c > 1 for c in coords):
                            ok = False
                        if ok:
                            cls_count[cls] += 1
                    except ValueError:
                        ok = False
                if not ok:
                    n_bad += 1

        status = "✓ OK" if (n_bad == 0 and not orphan_imgs and not orphan_lbls) else "✗ ISSUES"
        log.info(
            f"  [{split}] imgs={len(imgs):4d} lbls={len(lbls):4d} "
            f"lines={n_lines:5d} bad={n_bad} orphan_imgs={len(orphan_imgs)} "
            f"orphan_lbls={len(orphan_lbls)} │ {status}"
        )
        log.info(f"    Class counts: { {CLASS_NAMES[k]: v for k, v in cls_count.items()} }")

        report[split] = {
            "images": len(imgs), "labels": len(lbls),
            "annotations": n_lines, "bad_lines": n_bad,
            "class_counts": {CLASS_NAMES[k]: v for k, v in cls_count.items()}
        }

        if n_bad or orphan_imgs or orphan_lbls:
            all_ok = False

    # Persist verification report
    rpt_path = WORK_DIR / "verification_report.json"
    rpt_path.write_text(json.dumps(report, indent=2))
    log.info(f"\n  Verification report saved: {rpt_path}")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# STEP 9 — PRINT FINAL SUMMARY + DIRECTORY STRUCTURE
# ══════════════════════════════════════════════════════════════════════════
def print_summary(all_stats: dict, yaml_path: Path):
    log.info("━" * 60)
    log.info("DATASET PREPARATION COMPLETE")
    log.info("━" * 60)

    total_imgs   = sum(s["written"]      for s in all_stats.values())
    total_bboxes = sum(s["total_bboxes"] for s in all_stats.values())

    log.info(f"  {'split':6s} {'images':>8s} {'bboxes':>8s}")
    log.info(f"  {'─'*6}─{'─'*8}─{'─'*8}")
    for split, s in all_stats.items():
        log.info(f"  {split:6s} {s['written']:8d} {s['total_bboxes']:8d}")
    log.info(f"  {'TOTAL':6s} {total_imgs:8d} {total_bboxes:8d}")

    log.info(f"\n  data.yaml  : {yaml_path}")
    log.info(f"  Output root: {WORK_DIR}")

    log.info("\n  Directory structure:")
    log.info(f"  {WORK_DIR.name}/")
    log.info(f"  ├─ data.yaml")
    log.info(f"  ├─ prep.log")
    log.info(f"  ├─ verification_report.json")
    for split in ("train", "val", "test"):
        n_img = len(list((WORK_DIR / "images" / split).glob("*.jpg")))
        n_lbl = len(list((WORK_DIR / "labels" / split).glob("*.txt")))
        log.info(f"  ├─ images/{split}/   ({n_img} .jpg)")
        log.info(f"  └─ labels/{split}/   ({n_lbl} .txt)")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    log.info("═" * 60)
    log.info("CBIS-DDSM → YOLO  │  Mammogram Lesion Detection")
    log.info("═" * 60)
    log.info(f"  Input  : {INPUT_ROOT}")
    log.info(f"  Output : {WORK_DIR}")
    log.info(f"  Classes: {CLASS_NAMES}")
    log.info(f"  Split  : train={TRAIN_RATIO} val={VAL_RATIO} test={TEST_RATIO}")

    inspect_dataset()

    df     = load_annotations()
    rec_df = build_records(df)
    splits = split_records(rec_df)

    all_stats = {}
    for name, sdf in splits.items():
        log.info(f"\n  ── Writing {name} ──")
        all_stats[name] = write_split(name, sdf)

    yaml_path = write_data_yaml()
    ok        = verify_labels()
    print_summary(all_stats, yaml_path)

    if not ok:
        log.warning("Verification flagged issues. Review prep.log and verification_report.json.")
        sys.exit(1)

    log.info("\n  Dataset ready. Awaiting next command.")


if __name__ == "__main__":
    main()
