# ════════════════════════════════════════════════════════════════════════════
# EXP 22 — GATED MoE EVAL (router-gated mass+calc experts)
# ════════════════════════════════════════════════════════════════════════════
# Computes THREE rows of the paper's ablation table in one run:
#   (1) Domain-Split MoE  — plain union, no gating   (should match ~0.2612)
#   (2) Gated MoE         — router scales each expert's confidences
#   vs Universal baseline 0.2308 (already measured, reported in comments)
#
# Router class order is ALPHABETICAL: calc=0, mass=1
# Mass expert:  cls 0=benign_mass,  1=malignant_mass  → 4-class ids 0,1
# Calc expert:  cls 0=benign_calc,  1=malignant_calc  → 4-class ids 2,3
#
# Attach: CBIS-DDSM + weight-moe dataset (mass_best.pt, calc_best.pt, router_best.pt)
# GPU T4. ~30-45 min. Run as Save & Run All (commit).
# ════════════════════════════════════════════════════════════════════════════

# ── WEIGHT PATHS — update if your dataset name differs ──────────────────────
MASS_BEST   = "/kaggle/input/weight-moe/mass_best.pt"
CALC_BEST   = "/kaggle/input/weight-moe/calc_best.pt"
ROUTER_BEST = "/kaggle/input/weight-moe/router_best.pt"


# === CELL 1 — BUILD canonical 4-class test set (seed=42) ====================
import hashlib, shutil, pandas as pd, cv2, yaml
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

_cands = [Path('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'),
          Path('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset')]
CBIS = next((c for c in _cands if (c/'csv'/'dicom_info.csv').exists()), None)
assert CBIS is not None, f"CBIS not found in {_cands}"

ROOT = Path('/kaggle/working/canonical_4class'); ROOT.mkdir(parents=True, exist_ok=True)
SEED = 42

dicom = pd.read_csv(CBIS/'csv/dicom_info.csv')
full_img_map, mask_map = {}, {}
for _, r in dicom.iterrows():
    pid = str(r['PatientID']); desc = str(r['SeriesDescription']).lower()
    p = str(CBIS / str(r['image_path']).replace('CBIS-DDSM/', ''))
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

recs  = build(CBIS/'csv/mass_case_description_train_set.csv', 0) + \
        build(CBIS/'csv/mass_case_description_test_set.csv',  0)
recs += build(CBIS/'csv/calc_case_description_train_set.csv', 2) + \
        build(CBIS/'csv/calc_case_description_test_set.csv',  2)
rec = pd.DataFrame(recs).drop_duplicates(subset=['img_path','mask_path']).reset_index(drop=True)

tr_i, rest_i = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(rec, groups=rec['patient_id']))
rest = rec.iloc[rest_i].reset_index(drop=True)
va_i, te_i   = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(rest, groups=rest['patient_id']))
split_of = {}
for pid in rec.iloc[tr_i]['patient_id'].unique():  split_of[pid] = 'train'
for pid in rest.iloc[va_i]['patient_id'].unique(): split_of[pid] = 'val'
for pid in rest.iloc[te_i]['patient_id'].unique(): split_of[pid] = 'test'
rec['split'] = rec['patient_id'].map(split_of)

P = {s: set(rec[rec.split==s].patient_id) for s in ['train','val','test']}
assert P['train'].isdisjoint(P['val']) and P['train'].isdisjoint(P['test']) and P['val'].isdisjoint(P['test']), "LEAKAGE!"
print("patients train/val/test:", len(P['train']), len(P['val']), len(P['test']), "— zero overlap ✅")

def mask_boxes(mp, w, h):
    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if m is None: return []
    if m.shape != (h, w): m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    _, b = cv2.threshold(m, 10, 255, cv2.THRESH_BINARY)
    if b.max() == 0: return []
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < 8 or hh < 8 or (ww*hh)/(w*h) < 1e-4: continue
        out.append(((x+ww/2)/w, (y+hh/2)/h, ww/w, hh/h))
    return out

io = ROOT/'images'/'test'; lo = ROOT/'labels'/'test'
io.mkdir(parents=True, exist_ok=True); lo.mkdir(parents=True, exist_ok=True)
n = 0
for img_str, g in tqdm(rec[rec.split=='test'].groupby('img_path'), desc='4class/test'):
    img = cv2.imread(img_str)
    if img is None: continue
    h, w = img.shape[:2]; lines = []
    for _, row in g.iterrows():
        for bb in mask_boxes(Path(row['mask_path']), w, h):
            lines.append(f"{int(row['class_id'])} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
    if not lines: continue
    stem = Path(img_str).stem + '_' + hashlib.md5(img_str.encode()).hexdigest()[:8]
    shutil.copy2(img_str, io/(stem+'.jpg'))
    (lo/(stem+'.txt')).write_text('\n'.join(lines)+'\n')
    n += 1
print(f"test: {n} images ✅")


# === CELL 2 — GATED MoE INFERENCE + EVAL ====================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "torchmetrics"], check=False)
import torch, numpy as np, cv2
from pathlib import Path
from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO

SRC         = Path("/kaggle/working/canonical_4class")
CLASS_NAMES = ['benign_mass', 'malignant_mass', 'benign_calc', 'malignant_calc']
NC = 4; HIT_IOU = 0.20
ROUTER_CALC_IDX = 0   # alphabetical: calc < mass
ROUTER_MASS_IDX = 1

def yl(line, W, H):
    c, xc, yc, bw, bh = line.split()
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return int(c), [xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2]

def iou(a, b):
    ix0,iy0 = max(a[0],b[0]), max(a[1],b[1])
    ix1,iy1 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix1-ix0)*max(0,iy1-iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0

def froc(ap, ag, n_img, cls=None, fps=(0.5,1,2,4)):
    P = sorted([p for p in ap if cls is None or p[1]==cls], key=lambda x: -x[2])
    G = [g for g in ag if cls is None or g[1]==cls]; total = len(G)
    if total == 0: return None
    matched = set(); tp = fp = 0; sens = []; fppi = []
    for (im, c, sc, box) in P:
        best = -1; bj = None
        for j, (gim, gc, gbox) in enumerate(G):
            if gim != im or j in matched: continue
            v = iou(box, gbox)
            if v >= HIT_IOU and v > best: best = v; bj = j
        if bj is not None: matched.add(bj); tp += 1
        else: fp += 1
        sens.append(tp/total); fppi.append(fp/n_img)
    sens = np.array(sens); fppi = np.array(fppi); out = {}
    for f in fps:
        idx = np.where(fppi <= f)[0]
        out[f] = float(sens[idx[-1]]) if len(idx) else 0.0
    return out

def mk_pred(pb, ps, pl):
    return {
        "boxes":  torch.tensor(pb, dtype=torch.float).reshape(-1,4) if pb else torch.zeros((0,4)),
        "scores": torch.tensor(ps, dtype=torch.float),
        "labels": torch.tensor(pl, dtype=torch.long),
    }

def mk_gt(gb, gl):
    return {
        "boxes":  torch.tensor(gb, dtype=torch.float).reshape(-1,4) if gb else torch.zeros((0,4)),
        "labels": torch.tensor(gl, dtype=torch.long),
    }

print("Loading models...")
mass_model   = YOLO(MASS_BEST)
calc_model   = YOLO(CALC_BEST)
router_model = YOLO(ROUTER_BEST)
print("All 3 models loaded ✅")

test_imgs = sorted([p for p in (SRC/"images"/"test").glob("*")
                    if p.suffix.lower() in {".jpg",".jpeg",".png"}])
n_img = len(test_imgs)
print(f"Evaluating on {n_img} test images...")

metric_ds    = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
metric_gated = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
preds_ds_all, preds_gated_all, gts_all = [], [], []

for i, ip in enumerate(tqdm(test_imgs, desc="gated-moe")):
    H, W = cv2.imread(str(ip)).shape[:2]

    # Router → gate probabilities
    rr = router_model.predict(str(ip), imgsz=512, device=0, verbose=False)[0]
    probs  = rr.probs.data.cpu().tolist()
    p_calc = probs[ROUTER_CALC_IDX]
    p_mass = probs[ROUTER_MASS_IDX]

    # Mass expert (4-class ids 0, 1)
    mr = mass_model.predict(str(ip), imgsz=1280, conf=0.01, iou=0.6,
                            device=0, augment=True, verbose=False)[0]
    mass_dets = [(b.xyxy[0].tolist(), float(b.conf[0]), int(b.cls[0]))
                 for b in mr.boxes]

    # Calc expert (4-class ids 2, 3)
    cr = calc_model.predict(str(ip), imgsz=1280, conf=0.01, iou=0.6,
                            device=0, augment=True, verbose=False)[0]
    calc_dets = [(b.xyxy[0].tolist(), float(b.conf[0]), int(b.cls[0])+2)
                 for b in cr.boxes]

    all_dets = mass_dets + calc_dets

    # Domain-split: plain union, no gating
    pb_ds, ps_ds, pl_ds = [], [], []
    for box, conf, cls in all_dets:
        pb_ds.append(box); ps_ds.append(conf); pl_ds.append(cls)
        preds_ds_all.append((i, cls, conf, box))

    # Gated: multiply confidence by router probability
    pb_g, ps_g, pl_g = [], [], []
    for box, conf, cls in mass_dets:
        gc = conf * p_mass
        pb_g.append(box); ps_g.append(gc); pl_g.append(cls)
        preds_gated_all.append((i, cls, gc, box))
    for box, conf, cls in calc_dets:
        gc = conf * p_calc
        pb_g.append(box); ps_g.append(gc); pl_g.append(cls)
        preds_gated_all.append((i, cls, gc, box))

    # Ground truth
    gb, gl = [], []
    lp = SRC/"labels"/"test"/f"{ip.stem}.txt"
    if lp.exists():
        for ln in lp.read_text().splitlines():
            if ln.strip():
                c, box = yl(ln, W, H); gb.append(box); gl.append(c)
                gts_all.append((i, c, box))

    metric_ds.update([mk_pred(pb_ds, ps_ds, pl_ds)], [mk_gt(gb, gl)])
    metric_gated.update([mk_pred(pb_g, ps_g, pl_g)],  [mk_gt(gb, gl)])

# ── Results ─────────────────────────────────────────────────────────────────
m_ds = metric_ds.compute()
m_g  = metric_gated.compute()

print("\n" + "="*65)
print("  GATED MoE EVAL — CANONICAL 4-CLASS TEST")
print("="*65)
print(f"\n{'Method':<26} {'mAP50':>8}  {'mAP50-95':>10}")
print(f"{'Universal (baseline)':<26} {'0.2308':>8}  {'0.0958':>10}")
print(f"{'Domain-Split MoE':<26} {m_ds['map_50'].item():>8.4f}  {m_ds['map'].item():>10.4f}")
print(f"{'Gated MoE (ours)':<26} {m_g['map_50'].item():>8.4f}  {m_g['map'].item():>10.4f}")

if m_ds['map_per_class'].numel() == NC and m_g['map_per_class'].numel() == NC:
    print("\nPer-class mAP50-95:")
    for k, nm in enumerate(CLASS_NAMES):
        print(f"  {nm:<22}  DS={m_ds['map_per_class'][k].item():.4f}  "
              f"Gated={m_g['map_per_class'][k].item():.4f}")

print(f"\nFROC (IoU={HIT_IOU}):")
for cls in [None] + list(range(NC)):
    nm = "OVERALL" if cls is None else CLASS_NAMES[cls]
    r_ds    = froc(preds_ds_all,    gts_all, n_img, cls)
    r_gated = froc(preds_gated_all, gts_all, n_img, cls)
    if r_ds:
        print(f"  {nm:<22} DS:    "+"  ".join(f"S@{f}={r_ds[f]:.3f}"    for f in (0.5,1,2,4)))
    if r_gated:
        print(f"  {nm:<22} Gated: "+"  ".join(f"S@{f}={r_gated[f]:.3f}" for f in (0.5,1,2,4)))

print("\n" + "="*65)
gate_gated = m_g['map_50'].item()  > 0.2308
gate_ds    = m_ds['map_50'].item() > 0.2308
print(f">>> Domain-Split MoE {'✅ BEATS' if gate_ds else '❌ LOSES TO'} Universal  "
      f"({m_ds['map_50'].item():.4f} vs 0.2308)")
print(f">>> Gated MoE        {'✅ BEATS' if gate_gated else '❌ LOSES TO'} Universal  "
      f"({m_g['map_50'].item():.4f} vs 0.2308)")
print(f">>> Router delta:     {m_g['map_50'].item()-m_ds['map_50'].item():+.4f}  (Gated minus Domain-Split)")
print("="*65)
