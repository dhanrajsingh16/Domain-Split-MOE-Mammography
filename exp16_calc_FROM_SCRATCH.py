# ════════════════════════════════════════════════════════════════════════════
# EXP 16 — CALC SPECIALIST FROM SCRATCH  (new notebook, only CBIS-DDSM attached)
# ════════════════════════════════════════════════════════════════════════════
# 4 cells, run top to bottom:
#   CELL 1  build canonical_calc   (leakage-free patient split — VALIDATED builder)
#   CELL 2  tile mammograms        (1024px tiles at native res -> calcs stay large)
#   CELL 3  train yolov8l-p2       (stride-4 head for tiny objects)
#   CELL 4  SAHI eval on FULL test images  -> calc mAP50 (same test set = fair MoE compare)
# ════════════════════════════════════════════════════════════════════════════

# === CELL 1 — BUILD canonical_calc (leakage-free) ==========================
DATASET = "calc"   # this pipeline is calc-only

import hashlib, shutil, pandas as pd, cv2, yaml
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

# auto-detect the CBIS-DDSM mount (path differs by how the dataset is attached)
_cands = [
    Path('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'),
    Path('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset'),
]
CBIS = next((c for c in _cands if (c/'csv'/'dicom_info.csv').exists()), None)
assert CBIS is not None, f"CBIS-DDSM not found. Checked: {_cands}. Fix CBIS path."
ROOT = Path(f'/kaggle/working/canonical_{DATASET}'); ROOT.mkdir(parents=True, exist_ok=True)
SEED = 42
print("CBIS:", CBIS)

dicom = pd.read_csv(CBIS/'csv/dicom_info.csv')
full_img_map, mask_map = {}, {}
for _, r in dicom.iterrows():
    pid = str(r['PatientID']); desc = str(r['SeriesDescription']).lower()
    p = str(CBIS/str(r['image_path']).replace('CBIS-DDSM/', ''))
    if 'full mammogram' in desc: full_img_map.setdefault(pid, p)
    elif 'roi mask' in desc:     mask_map.setdefault(pid, p)

def build(csv, off):
    df = pd.read_csv(csv); out = []
    for _, row in df.iterrows():
        ip = full_img_map.get(str(row['image file path']).split('/')[0])
        mp = mask_map.get(str(row['ROI mask file path']).split('/')[0])
        if not ip or not mp or not Path(ip).exists() or not Path(mp).exists(): continue
        cls = off + (1 if 'MALIGNANT' in str(row['pathology']).upper() else 0)
        out.append({'img_path': ip, 'mask_path': mp, 'class_id': cls,
                    'patient_id': str(row['patient_id'])})
    return out

# build ALL records (mass off=0, calc off=2) so the split matches mass/universal notebooks
recs  = build(CBIS/'csv/mass_case_description_train_set.csv', 0) + build(CBIS/'csv/mass_case_description_test_set.csv', 0)
recs += build(CBIS/'csv/calc_case_description_train_set.csv', 2) + build(CBIS/'csv/calc_case_description_test_set.csv', 2)
rec = pd.DataFrame(recs).drop_duplicates(subset=['img_path', 'mask_path']).reset_index(drop=True)

# ── ONE canonical patient split (deterministic, identical on every account) ──
tr_i, rest_i = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(rec, groups=rec['patient_id']))
rest = rec.iloc[rest_i].reset_index(drop=True)
va_i, te_i = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(rest, groups=rest['patient_id']))
split_of = {}
for pid in rec.iloc[tr_i]['patient_id'].unique():  split_of[pid] = 'train'
for pid in rest.iloc[va_i]['patient_id'].unique(): split_of[pid] = 'val'
for pid in rest.iloc[te_i]['patient_id'].unique(): split_of[pid] = 'test'
rec['split'] = rec['patient_id'].map(split_of)

# ── PROVE zero leakage ──
P = {s: set(rec[rec.split == s].patient_id) for s in ['train', 'val', 'test']}
assert P['train'].isdisjoint(P['val']) and P['train'].isdisjoint(P['test']) and P['val'].isdisjoint(P['test']), "LEAKAGE!"
print("patients train/val/test:", len(P['train']), len(P['val']), len(P['test']), "— zero overlap ✅")

CFG = {
    "calc": dict(keep=[2, 3], remap={2: 0, 3: 1}, names=['benign_calc', 'malignant_calc']),
}[DATASET]

def boxes(mp, w, h, lesion):
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if m is None: return []
    if m.shape != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    _, b = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    if b.max() == 0: return []
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, ww, hh = cv2.boundingRect(c)
        if lesion == 'mass':
            if ww*hh < 100: continue
        else:
            if ww < 8 or hh < 8 or (ww*hh)/(w*h) < 1e-4: continue
        out.append(((x+ww/2)/w, (y+hh/2)/h, ww/w, hh/h))
    return out

sub = rec[rec['class_id'].isin(CFG['keep'])]
for s in ['train', 'val', 'test']:
    io = ROOT/'images'/s; lo = ROOT/'labels'/s
    io.mkdir(parents=True, exist_ok=True); lo.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_str, g in tqdm(sub[sub.split == s].groupby('img_path'), desc=f"{DATASET}/{s}"):
        img = cv2.imread(img_str)
        if img is None: continue
        h, w = img.shape[:2]; lines = []
        for _, row in g.iterrows():
            lesion = 'mass' if int(row['class_id']) < 2 else 'calc'
            for bb in boxes(Path(row['mask_path']), w, h, lesion):
                lines.append(f"{CFG['remap'][int(row['class_id'])]} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
        if not lines: continue
        stem = Path(img_str).stem + '_' + hashlib.md5(img_str.encode()).hexdigest()[:8]
        shutil.copy2(img_str, io/(stem+'.jpg')); (lo/(stem+'.txt')).write_text('\n'.join(lines)+'\n'); n += 1
    print(f"  {s}: {n} images")

yaml.dump({'path': str(ROOT), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
           'nc': len(CFG['names']), 'names': CFG['names']}, open(ROOT/'data.yaml', 'w'), sort_keys=False)
print("canonical_calc ready at", ROOT)


# === CELL 2 — TILE THE DATASET (train + val) ===============================
import os, cv2, yaml, math, random
from pathlib import Path
from tqdm.auto import tqdm

SRC       = Path("/kaggle/working/canonical_calc")
DST       = Path("/kaggle/working/canonical_calc_tiled")
TILE      = 1024
OVERLAP   = 0.25
MIN_VIS   = 0.30
NEG_RATIO = 0.30
NEG_CAP   = 32
SEED      = 42
random.seed(SEED)

STRIDE = int(TILE * (1 - OVERLAP))
src_yaml = yaml.safe_load((SRC/"data.yaml").read_text())
CLASS_NAMES = src_yaml["names"]
if isinstance(CLASS_NAMES, dict):
    CLASS_NAMES = [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)]
print(f"Classes: {CLASS_NAMES} | TILE={TILE} STRIDE={STRIDE}")

def tile_origins(size, tile, stride):
    if size <= tile: return [0]
    xs = list(range(0, size-tile+1, stride))
    if xs[-1] != size-tile: xs.append(size-tile)
    return xs

def yolo_to_xyxy(line, W, H):
    c, xc, yc, bw, bh = line.split()
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return int(c), xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2

def build_split(split):
    img_dir, lbl_dir = SRC/"images"/split, SRC/"labels"/split
    out_img, out_lbl = DST/"images"/split, DST/"labels"/split
    out_img.mkdir(parents=True, exist_ok=True); out_lbl.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    n_pos = n_neg = n_box = 0
    for ip in tqdm(imgs, desc=f"tiling {split}"):
        img = cv2.imread(str(ip))
        if img is None: continue
        H, W = img.shape[:2]
        lp = lbl_dir/f"{ip.stem}.txt"
        gts = [yolo_to_xyxy(ln, W, H) for ln in (lp.read_text().splitlines() if lp.exists() else []) if ln.strip()]
        img_pos = 0; neg_tiles = []
        for y0 in tile_origins(H, TILE, STRIDE):
            for x0 in tile_origins(W, TILE, STRIDE):
                x1, y1 = x0+TILE, y0+TILE
                lines = []
                for c, bx0, by0, bx1, by1 in gts:
                    ix0, iy0 = max(bx0,x0), max(by0,y0)
                    ix1, iy1 = min(bx1,x1), min(by1,y1)
                    iw, ih = ix1-ix0, iy1-iy0
                    if iw <= 0 or ih <= 0: continue
                    if (iw*ih)/max((bx1-bx0)*(by1-by0),1e-6) < MIN_VIS: continue
                    cx = ((ix0+ix1)/2 - x0)/TILE; cy = ((iy0+iy1)/2 - y0)/TILE
                    lines.append(f"{c} {cx:.6f} {cy:.6f} {iw/TILE:.6f} {ih/TILE:.6f}")
                tile = img[y0:y1, x0:x1]
                if tile.shape[0] != TILE or tile.shape[1] != TILE:
                    tile = cv2.copyMakeBorder(tile, 0, TILE-tile.shape[0], 0, TILE-tile.shape[1], cv2.BORDER_CONSTANT, value=0)
                stem = f"{ip.stem}_{x0}_{y0}"
                if lines:
                    cv2.imwrite(str(out_img/f"{stem}.jpg"), tile)
                    (out_lbl/f"{stem}.txt").write_text("\n".join(lines)+"\n")
                    n_pos += 1; img_pos += 1; n_box += len(lines)
                else:
                    neg_tiles.append((stem, tile))
        random.shuffle(neg_tiles)
        keep = min(len(neg_tiles), NEG_CAP, math.ceil(NEG_RATIO*img_pos))
        for stem, tile in neg_tiles[:keep]:
            cv2.imwrite(str(out_img/f"{stem}.jpg"), tile)
            (out_lbl/f"{stem}.txt").write_text(""); n_neg += 1
    print(f"[{split}] positive_tiles={n_pos}  bg_tiles={n_neg}  boxes={n_box}")

for sp in ("train", "val"):
    build_split(sp)

(DST/"data.yaml").write_text(yaml.dump(
    {"path": str(DST), "train":"images/train", "val":"images/val",
     "nc": len(CLASS_NAMES), "names": CLASS_NAMES}, sort_keys=False))
print("Wrote", DST/"data.yaml")


# === CELL 3 — TRAIN yolov8l-P2 ON TILES (2× T4) ===========================
# NOTE: if a previous run OOMs, do Run -> Restart (frees zombie GPU memory), then
#       rerun this cell. Tiled data persists on disk so Cells 1-2 needn't rerun.
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

from ultralytics import YOLO
model = YOLO("yolov8l-p2.yaml").load("yolov8l.pt")
model.train(
    data="/kaggle/working/canonical_calc_tiled/data.yaml",
    imgsz=1024, epochs=45, batch=8, device=0,   # 45 ep fits the 12h commit with margin; SINGLE GPU (P2 head can't use DDP)
    optimizer="AdamW", lr0=0.001, lrf=0.01,
    momentum=0.937, weight_decay=0.0005,
    warmup_epochs=5, cos_lr=True, patience=30,
    mosaic=0.0, copy_paste=0.0, mixup=0.0,
    fliplr=0.5, flipud=0.5, degrees=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
    seed=42, deterministic=True, workers=4,
    project="/kaggle/working/canonical_calc_tiled/runs",
    name="calc_p2_tiled", exist_ok=True, plots=True,
)
print("BEST:", "/kaggle/working/canonical_calc_tiled/runs/calc_p2_tiled/weights/best.pt")
# >>> SAVE VERSION IN KAGGLE right after this finishes <<<


# === CELL 4 — SAHI EVAL ON FULL TEST IMAGES + mAP ==========================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sahi", "torchmetrics"], check=False)
import cv2, yaml, torch, numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from torchmetrics.detection.mean_ap import MeanAveragePrecision

SRC = Path("/kaggle/working/canonical_calc")
TILE, OVERLAP = 1024, 0.25
HIT_IOU = 0.20   # FROC hit criterion for tiny calc clusters (mAP50 below still uses 0.5)
BEST = "/kaggle/working/canonical_calc_tiled/runs/calc_p2_tiled/weights/best.pt"
_names = yaml.safe_load((SRC/"data.yaml").read_text())["names"]
CLASS_NAMES = [_names[k] for k in sorted(_names)] if isinstance(_names, dict) else _names
NC = len(CLASS_NAMES)

def yolo_to_xyxy(line, W, H):
    c, xc, yc, bw, bh = line.split()
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return int(c), xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2

def iou(a, b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix1-ix0)*max(0,iy1-iy0)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0

# SAHI runs at a LOW conf so the FROC curve can sweep the full operating range
det = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",        # change to "yolov8" if your sahi version errors
    model_path=BEST, confidence_threshold=0.01, device="cuda:0")
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
test_imgs = sorted([p for p in (SRC/"images"/"test").glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
n_img = len(test_imgs)
all_preds, all_gts = [], []

for i, ip in enumerate(tqdm(test_imgs, desc="SAHI test")):
    img = cv2.imread(str(ip)); H, W = img.shape[:2]
    res = get_sliced_prediction(str(ip), det, slice_height=TILE, slice_width=TILE,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
        postprocess_type="NMS", postprocess_match_threshold=0.5, verbose=0)
    pb, ps, pl = [], [], []
    for o in res.object_prediction_list:
        x0, y0, x1, y1 = o.bbox.to_xyxy()
        box = [x0,y0,x1,y1]
        pb.append(box); ps.append(o.score.value); pl.append(o.category.id)
        all_preds.append((i, o.category.id, o.score.value, box))
    gb, gl = [], []
    lp = SRC/"labels"/"test"/f"{ip.stem}.txt"
    if lp.exists():
        for ln in lp.read_text().splitlines():
            if ln.strip():
                c, x0, y0, x1, y1 = yolo_to_xyxy(ln, W, H)
                box = [x0,y0,x1,y1]
                gb.append(box); gl.append(c); all_gts.append((i, c, box))
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
if "map_per_class" in m and m["map_per_class"].numel() == NC:
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:18s} AP50-95 = {m['map_per_class'][i].item():.4f}")
print("============================================================")

# ── FROC: sensitivity (recall) vs false-positives-per-image ──────────────────
def froc(cls=None, fp_points=(0.5,1,2,4)):
    P=sorted([p for p in all_preds if cls is None or p[1]==cls], key=lambda x:-x[2])
    G=[g for g in all_gts if cls is None or g[1]==cls]; total=len(G)
    if total==0: return None
    matched=set(); tp=fp=0; sens=[]; fppi=[]
    for (im,c,sc,box) in P:
        best=-1; bj=None
        for j,(gim,gc,gbox) in enumerate(G):
            if gim!=im or j in matched: continue
            v=iou(box,gbox)
            if v>=HIT_IOU and v>best: best=v; bj=j
        if bj is not None: matched.add(bj); tp+=1
        else: fp+=1
        sens.append(tp/total); fppi.append(fp/n_img)
    sens=np.array(sens); fppi=np.array(fppi); out={}
    for f in fp_points:
        idx=np.where(fppi<=f)[0]; out[f]=float(sens[idx[-1]]) if len(idx) else 0.0
    return out

print(f"\nFROC (hit IoU={HIT_IOU}) — sensitivity @ FP/image:")
for cls in [None]+list(range(NC)):
    nm="OVERALL" if cls is None else CLASS_NAMES[cls]
    r=froc(cls)
    if r: print(f"  {nm:18s} " + "  ".join(f"S@{f}={r[f]:.3f}" for f in (0.5,1,2,4)))
print("============================================================")
