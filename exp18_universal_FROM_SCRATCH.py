# ════════════════════════════════════════════════════════════════════════════
# EXP 18 — UNIVERSAL 4-CLASS BASELINE  (the model MoE must beat)
# ════════════════════════════════════════════════════════════════════════════
# One model detects ALL 4 classes: benign_mass, malignant_mass, benign_calc,
# malignant_calc — trained on the SAME canonical seed=42 split (test patients
# identical to mass & calc experts -> fair MoE-vs-Universal compare).
# Plain full-image YOLO11l, NO tiling (the standard "one model does everything"
# baseline). 3 cells: build -> train -> eval (mAP + FROC).
# Attach ONLY CBIS-DDSM. GPU T4 x2 is fine here (YOLO11l works with DDP).
# Run as Save & Run All (commit).
# ════════════════════════════════════════════════════════════════════════════

# === CELL 1 — BUILD canonical_universal (leakage-free, all 4 classes) ========
DATASET = "universal"
import hashlib, shutil, pandas as pd, cv2, yaml
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

_cands = [
    Path('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'),
    Path('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset'),
]
CBIS = next((c for c in _cands if (c/'csv'/'dicom_info.csv').exists()), None)
assert CBIS is not None, f"CBIS-DDSM not found. Checked: {_cands}"
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

recs  = build(CBIS/'csv/mass_case_description_train_set.csv', 0) + build(CBIS/'csv/mass_case_description_test_set.csv', 0)
recs += build(CBIS/'csv/calc_case_description_train_set.csv', 2) + build(CBIS/'csv/calc_case_description_test_set.csv', 2)
rec = pd.DataFrame(recs).drop_duplicates(subset=['img_path','mask_path']).reset_index(drop=True)

# ── SAME canonical patient split as mass/calc (deterministic seed=42) ──
tr_i, rest_i = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(rec, groups=rec['patient_id']))
rest = rec.iloc[rest_i].reset_index(drop=True)
va_i, te_i = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(rest, groups=rest['patient_id']))
split_of = {}
for pid in rec.iloc[tr_i]['patient_id'].unique():  split_of[pid] = 'train'
for pid in rest.iloc[va_i]['patient_id'].unique(): split_of[pid] = 'val'
for pid in rest.iloc[te_i]['patient_id'].unique(): split_of[pid] = 'test'
rec['split'] = rec['patient_id'].map(split_of)

P = {s: set(rec[rec.split==s].patient_id) for s in ['train','val','test']}
assert P['train'].isdisjoint(P['val']) and P['train'].isdisjoint(P['test']) and P['val'].isdisjoint(P['test']), "LEAKAGE!"
print("patients train/val/test:", len(P['train']), len(P['val']), len(P['test']), "— zero overlap ✅")

# keep ALL 4 classes, identity remap
CFG = dict(keep=[0,1,2,3], remap={0:0,1:1,2:2,3:3},
           names=['benign_mass','malignant_mass','benign_calc','malignant_calc'])

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
for s in ['train','val','test']:
    io = ROOT/'images'/s; lo = ROOT/'labels'/s
    io.mkdir(parents=True, exist_ok=True); lo.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_str, g in tqdm(sub[sub.split==s].groupby('img_path'), desc=f"{DATASET}/{s}"):
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

yaml.dump({'path': str(ROOT), 'train':'images/train', 'val':'images/val', 'test':'images/test',
           'nc': len(CFG['names']), 'names': CFG['names']}, open(ROOT/'data.yaml','w'), sort_keys=False)
print("canonical_universal ready at", ROOT)


# === CELL 2 — TRAIN UNIVERSAL (YOLO11l, 4-class, full image, 2x T4) ==========
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO

model = YOLO("yolo11l.pt")
model.train(
    data="/kaggle/working/canonical_universal/data.yaml",
    imgsz=1024, epochs=100, batch=8, device=[0,1],   # YOLO11l (no P2) is DDP-safe -> use both T4
    optimizer="AdamW", lr0=0.0005, lrf=0.01,
    momentum=0.937, weight_decay=0.0005,
    warmup_epochs=10, cos_lr=True, patience=30,
    mosaic=0.0, copy_paste=0.3, close_mosaic=10,
    fliplr=0.5, flipud=0.5, degrees=0.0,
    seed=42, deterministic=True, workers=4,
    project="/kaggle/working/canonical_universal/runs",
    name="universal", exist_ok=True, plots=True,
)
print("BEST:", "/kaggle/working/canonical_universal/runs/universal/weights/best.pt")
# >>> SAVE VERSION IN KAGGLE right after this finishes <<<


# === CELL 3 — UNIVERSAL EVAL: TWO PASSES (full-image baseline + SAHI) ========
# Pass A = Universal full-image  -> the standard baseline
# Pass B = Universal + SAHI       -> matched-inference baseline (same SAHI tool MoE's calc expert uses)
# Compare MoE against the STRONGER of the two. Same weights, two eval passes — no retraining.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sahi", "torchmetrics"], check=False)
import cv2, yaml, torch, numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

SRC  = Path("/kaggle/working/canonical_universal")
BEST = "/kaggle/working/canonical_universal/runs/universal/weights/best.pt"
HIT_IOU = 0.30          # FROC hit criterion (mAP50 uses strict 0.5)
TILE, OVERLAP = 1024, 0.25
names = yaml.safe_load((SRC/"data.yaml").read_text())["names"]
CLASS_NAMES = [names[k] for k in sorted(names)] if isinstance(names, dict) else names
NC = len(CLASS_NAMES)
test_imgs = sorted([p for p in (SRC/"images"/"test").glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
n_img = len(test_imgs)

def yl(line, W, H):
    c, xc, yc, bw, bh = line.split()
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return int(c), [xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2]
def iou(a, b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix1-ix0)*max(0,iy1-iy0)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0

def froc(all_preds, all_gts, cls=None, fp_points=(0.5,1,2,4)):
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

def evaluate(predict_fn, tag):
    """predict_fn(image_path, W, H) -> list of (cls:int, score:float, box:[x0,y0,x1,y1])."""
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    all_preds, all_gts = [], []
    for i, ip in enumerate(tqdm(test_imgs, desc=tag)):
        H, W = cv2.imread(str(ip)).shape[:2]
        preds = predict_fn(ip, W, H)
        pb, ps, pl = [], [], []
        for (c, sc, box) in preds:
            pb.append(box); ps.append(sc); pl.append(c)
            all_preds.append((i, c, sc, box))
        gb, gl = [], []
        lp = SRC/"labels"/"test"/f"{ip.stem}.txt"
        if lp.exists():
            for ln in lp.read_text().splitlines():
                if ln.strip():
                    c, box = yl(ln, W, H); gb.append(box); gl.append(c); all_gts.append((i, c, box))
        metric.update(
            [{"boxes": torch.tensor(pb or [], dtype=torch.float).reshape(-1,4),
              "scores": torch.tensor(ps or [], dtype=torch.float),
              "labels": torch.tensor(pl or [], dtype=torch.long)}],
            [{"boxes": torch.tensor(gb or [], dtype=torch.float).reshape(-1,4),
              "labels": torch.tensor(gl or [], dtype=torch.long)}])
    m = metric.compute()
    print(f"\n========  {tag}  — CANONICAL TEST  ========")
    print(f"  mAP50    = {m['map_50'].item():.4f}")
    print(f"  mAP50-95 = {m['map'].item():.4f}")
    if "map_per_class" in m and m["map_per_class"].numel() == NC:
        for k, nm in enumerate(CLASS_NAMES):
            print(f"    {nm:18s} AP50-95 = {m['map_per_class'][k].item():.4f}")
    print(f"  FROC (hit IoU={HIT_IOU}) — sensitivity @ FP/image:")
    for cls in [None]+list(range(NC)):
        nm="OVERALL" if cls is None else CLASS_NAMES[cls]
        r=froc(all_preds, all_gts, cls)
        if r: print(f"    {nm:18s} " + "  ".join(f"S@{f}={r[f]:.3f}" for f in (0.5,1,2,4)))
    print("="*60)
    return m['map_50'].item()

# ---- Pass A: Universal full-image (BASELINE) -------------------------------
ymodel = YOLO(BEST)
def predict_full(ip, W, H):
    r = ymodel.predict(str(ip), imgsz=1024, conf=0.01, iou=0.6, device=0, augment=True, verbose=False)[0]
    return [(int(b.cls[0]), float(b.conf[0]), b.xyxy[0].tolist()) for b in r.boxes]
map_full = evaluate(predict_full, "UNIVERSAL (full-image) [BASELINE]")

# ---- Pass B: Universal + SAHI (matched inference) --------------------------
det = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",        # change to "yolov8" if your sahi version errors
    model_path=BEST, confidence_threshold=0.01, device="cuda:0")
def predict_sahi(ip, W, H):
    res = get_sliced_prediction(str(ip), det, slice_height=TILE, slice_width=TILE,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
        postprocess_type="NMS", postprocess_match_threshold=0.5, verbose=0)
    out = []
    for o in res.object_prediction_list:
        out.append((o.category.id, o.score.value, list(o.bbox.to_xyxy())))
    return out
map_sahi = evaluate(predict_sahi, "UNIVERSAL + SAHI [matched inference]")

print(f"\n>>> Universal best mAP50 = {max(map_full, map_sahi):.4f} "
      f"(full={map_full:.4f}, sahi={map_sahi:.4f}) — MoE must beat THIS.")
