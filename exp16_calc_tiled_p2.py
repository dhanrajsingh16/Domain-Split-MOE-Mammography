# ════════════════════════════════════════════════════════════════════════════
# EXP 16 — CALCIFICATION SPECIALIST via TILING + YOLOv8l-P2 + SAHI
# ════════════════════════════════════════════════════════════════════════════
# WHY: canonical_calc images are FULL-RES mammograms (~3-5k px). Downscaling the
#      whole image to 1280 destroys pixel-scale calc clusters -> calc stuck ~0.09.
# FIX: (1) slice mammograms into 1024px tiles at NATIVE res  -> calcs stay large
#      (2) train yolov8l-P2 (adds stride-4 head for tiny objects)
#      (3) SAHI inference on FULL test images -> merge -> eval vs ORIGINAL GT
#          (test set stays byte-identical to mass/universal/MoE -> fair compare)
#
# PREREQUISITE: /kaggle/working/canonical_calc must already exist (build it first,
#               or attach it via Add Input and point SRC at /kaggle/input/...).
# Run as 3 Kaggle cells (split on the "# === CELL N" markers).
# ════════════════════════════════════════════════════════════════════════════

# === CELL 1 — BUILD TILED DATASET (train + val) ============================
import os, cv2, yaml, math, random, shutil
from pathlib import Path
from tqdm.auto import tqdm

# ---- CONFIG ----------------------------------------------------------------
SRC       = Path("/kaggle/working/canonical_calc")        # existing calc dataset
DST       = Path("/kaggle/working/canonical_calc_tiled")  # new tiled dataset
TILE      = 1024          # tile size (px) at native resolution
OVERLAP   = 0.25          # fraction overlap between adjacent tiles
MIN_VIS   = 0.30          # keep a GT box if >=30% of its area falls inside the tile
NEG_RATIO = 0.30          # background tiles kept per positive tile in same image
NEG_CAP   = 32            # hard cap on background tiles per image
SEED      = 42
random.seed(SEED)

# --- guard: dataset must exist ---
assert (SRC / "data.yaml").exists(), (
    f"\n>>> {SRC}/data.yaml NOT FOUND. This is a fresh notebook.\n"
    f">>> Build canonical_calc first (run your cbis_ddsm_to_yolo step), OR\n"
    f">>> Add Input the saved canonical_calc dataset and set SRC to its /kaggle/input path.\n"
)

STRIDE = int(TILE * (1 - OVERLAP))
src_yaml = yaml.safe_load((SRC / "data.yaml").read_text())
CLASS_NAMES = src_yaml["names"]
if isinstance(CLASS_NAMES, dict):
    CLASS_NAMES = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)]
print(f"Classes: {CLASS_NAMES} | TILE={TILE} STRIDE={STRIDE} overlap={OVERLAP}")


def tile_origins(size, tile, stride):
    """Sliding-window start coords; last window flush to the edge."""
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, stride))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def yolo_to_xyxy(line, W, H):
    c, xc, yc, bw, bh = line.split()
    xc, yc, bw, bh = float(xc) * W, float(yc) * H, float(bw) * W, float(bh) * H
    return int(c), xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2


def build_split(split):
    img_dir = SRC / "images" / split
    lbl_dir = SRC / "labels" / split
    out_img = DST / "images" / split
    out_lbl = DST / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    n_pos = n_neg = n_box = 0

    for ip in tqdm(imgs, desc=f"tiling {split}"):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        H, W = img.shape[:2]
        lp = lbl_dir / f"{ip.stem}.txt"
        gts = []
        if lp.exists():
            for ln in lp.read_text().splitlines():
                if ln.strip():
                    gts.append(yolo_to_xyxy(ln, W, H))

        img_pos = 0
        neg_tiles = []
        for y0 in tile_origins(H, TILE, STRIDE):
            for x0 in tile_origins(W, TILE, STRIDE):
                x1, y1 = x0 + TILE, y0 + TILE
                lines = []
                for c, bx0, by0, bx1, by1 in gts:
                    ix0, iy0 = max(bx0, x0), max(by0, y0)
                    ix1, iy1 = min(bx1, x1), min(by1, y1)
                    iw, ih = ix1 - ix0, iy1 - iy0
                    if iw <= 0 or ih <= 0:
                        continue
                    box_area = max((bx1 - bx0) * (by1 - by0), 1e-6)
                    if (iw * ih) / box_area < MIN_VIS:
                        continue                      # too little of the box visible
                    cx = ((ix0 + ix1) / 2 - x0) / TILE
                    cy = ((iy0 + iy1) / 2 - y0) / TILE
                    lines.append(f"{c} {cx:.6f} {cy:.6f} {iw/TILE:.6f} {ih/TILE:.6f}")

                tile = img[y0:y1, x0:x1]
                if tile.shape[0] != TILE or tile.shape[1] != TILE:   # edge pad
                    tile = cv2.copyMakeBorder(tile, 0, TILE - tile.shape[0], 0,
                                              TILE - tile.shape[1], cv2.BORDER_CONSTANT, value=0)
                stem = f"{ip.stem}_{x0}_{y0}"
                if lines:
                    cv2.imwrite(str(out_img / f"{stem}.jpg"), tile)
                    (out_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                    n_pos += 1; img_pos += 1; n_box += len(lines)
                else:
                    neg_tiles.append((stem, tile))

        # keep a capped sample of background tiles from THIS image (suppress false positives)
        random.shuffle(neg_tiles)
        keep = min(len(neg_tiles), NEG_CAP, math.ceil(NEG_RATIO * img_pos))
        for stem, tile in neg_tiles[:keep]:
            cv2.imwrite(str(out_img / f"{stem}.jpg"), tile)
            (out_lbl / f"{stem}.txt").write_text("")
            n_neg += 1

    print(f"[{split}] positive_tiles={n_pos}  bg_tiles={n_neg}  boxes={n_box}")


for sp in ("train", "val"):
    build_split(sp)

tiled_yaml = {
    "path": str(DST), "train": "images/train", "val": "images/val",
    "nc": len(CLASS_NAMES), "names": CLASS_NAMES,
}
(DST / "data.yaml").write_text(yaml.dump(tiled_yaml, sort_keys=False))
print("Wrote", DST / "data.yaml")


# === CELL 2 — TRAIN yolov8l-P2 ON TILES ====================================
from ultralytics import YOLO

# P2 arch = extra stride-4 detection head for tiny objects; load v8l weights for transfer
model = YOLO("yolov8l-p2.yaml").load("yolov8l.pt")

model.train(
    data="/kaggle/working/canonical_calc_tiled/data.yaml",
    imgsz=1024,
    epochs=150,
    batch=4,                # keep >=4 (batch=1 collapses BatchNorm); raise to 6/8 if VRAM allows
    optimizer="AdamW",
    lr0=0.001, lrf=0.01,
    momentum=0.937, weight_decay=0.0005,
    warmup_epochs=5, cos_lr=True,
    patience=40,
    mosaic=0.0, copy_paste=0.0, mixup=0.0,   # calcs hate mosaic; tiles already enlarge objects
    fliplr=0.5, flipud=0.5, degrees=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
    seed=42, deterministic=True,
    project="/kaggle/working/canonical_calc_tiled/runs",
    name="calc_p2_tiled", exist_ok=True,
    plots=True,
)
print("BEST:", "/kaggle/working/canonical_calc_tiled/runs/calc_p2_tiled/weights/best.pt")
# >>> SAVE VERSION IN KAGGLE NOW so best.pt persists <<<


# === CELL 3 — SAHI INFERENCE ON FULL TEST IMAGES + mAP =====================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sahi", "torchmetrics"], check=False)

import cv2, yaml, torch
from pathlib import Path
from tqdm.auto import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# self-contained constants (in case Cell 1 state was lost)
SRC     = Path("/kaggle/working/canonical_calc")
TILE, OVERLAP = 1024, 0.25
BEST    = "/kaggle/working/canonical_calc_tiled/runs/calc_p2_tiled/weights/best.pt"
_names  = yaml.safe_load((SRC / "data.yaml").read_text())["names"]
CLASS_NAMES = [_names[k] for k in sorted(_names)] if isinstance(_names, dict) else _names

def yolo_to_xyxy(line, W, H):
    c, xc, yc, bw, bh = line.split()
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return int(c), xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2

det = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",        # use "yolov8" if your sahi version errors here
    model_path=BEST, confidence_threshold=0.10, device="cuda:0",
)
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
test_imgs = sorted([p for p in (SRC/"images"/"test").glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

for ip in tqdm(test_imgs, desc="SAHI test"):
    img = cv2.imread(str(ip)); H, W = img.shape[:2]
    res = get_sliced_prediction(
        str(ip), det, slice_height=TILE, slice_width=TILE,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
        postprocess_type="NMS", postprocess_match_threshold=0.5, verbose=0)
    pb, ps, pl = [], [], []
    for o in res.object_prediction_list:
        x0, y0, x1, y1 = o.bbox.to_xyxy()
        pb.append([x0, y0, x1, y1]); ps.append(o.score.value); pl.append(o.category.id)
    gb, gl = [], []
    lp = SRC/"labels"/"test"/f"{ip.stem}.txt"
    if lp.exists():
        for ln in lp.read_text().splitlines():
            if ln.strip():
                c, x0, y0, x1, y1 = yolo_to_xyxy(ln, W, H)
                gb.append([x0, y0, x1, y1]); gl.append(c)
    metric.update(
        [{"boxes": torch.tensor(pb or [], dtype=torch.float).reshape(-1,4),
          "scores": torch.tensor(ps or [], dtype=torch.float),
          "labels": torch.tensor(pl or [], dtype=torch.long)}],
        [{"boxes": torch.tensor(gb or [], dtype=torch.float).reshape(-1,4),
          "labels": torch.tensor(gl or [], dtype=torch.long)}])

m = metric.compute()
print("\n========  CALC (TILED P2 + SAHI) — CANONICAL TEST  ========")
print(f"  mAP50    = {m['map_50'].item():.4f}")
print(f"  mAP50-95 = {m['map'].item():.4f}")
if "map_per_class" in m and m["map_per_class"].numel() == len(CLASS_NAMES):
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:18s} AP50-95 = {m['map_per_class'][i].item():.4f}")
print("============================================================")
