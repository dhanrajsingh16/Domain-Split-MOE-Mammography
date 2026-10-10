# ════════════════════════════════════════════════════════════════════════════
# EXP 19 — MoE EVAL: mass expert (full-image) + calc expert (SAHI) -> 4-class
# ════════════════════════════════════════════════════════════════════════════
# Disjoint-class union: mass expert outputs benign_mass/malignant_mass,
# calc expert outputs benign_calc/malignant_calc (remapped to class 2,3).
# Evaluated on the SAME canonical seed=42 4-class test as Universal -> fair gate.
#
# ATTACH (Add Input):
#   - CBIS-DDSM
#   - mass notebook OUTPUT  (has weights/best.pt, run name 'mass')
#   - calc notebook OUTPUT  (has weights/best.pt, run name 'calc_p2_tiled')
# GPU T4. 2 cells: build 4-class test -> MoE eval (mAP + FROC).
# ════════════════════════════════════════════════════════════════════════════

# === CELL 1 — BUILD canonical 4-class test (seed=42, identical to universal) ==
DATASET = "moe"
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
        out.append({'img_path': ip, 'mask_path': mp, 'class_id': cls, 'patient_id': str(row['patient_id'])})
    return out

recs  = build(CBIS/'csv/mass_case_description_train_set.csv', 0) + build(CBIS/'csv/mass_case_description_test_set.csv', 0)
recs += build(CBIS/'csv/calc_case_description_train_set.csv', 2) + build(CBIS/'csv/calc_case_description_test_set.csv', 2)
rec = pd.DataFrame(recs).drop_duplicates(subset=['img_path','mask_path']).reset_index(drop=True)

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

CLASS_NAMES = ['benign_mass','malignant_mass','benign_calc','malignant_calc']
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

# only the TEST split is needed for the gate
io = ROOT/'images'/'test'; lo = ROOT/'labels'/'test'
io.mkdir(parents=True, exist_ok=True); lo.mkdir(parents=True, exist_ok=True)
sub = rec[rec['split']=='test']
n = 0
for img_str, g in tqdm(sub.groupby('img_path'), desc="moe/test"):
    img = cv2.imread(img_str)
    if img is None: continue
    h, w = img.shape[:2]; lines = []
    for _, row in g.iterrows():
        lesion = 'mass' if int(row['class_id']) < 2 else 'calc'
        for bb in boxes(Path(row['mask_path']), w, h, lesion):
            lines.append(f"{int(row['class_id'])} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
    if not lines: continue
    stem = Path(img_str).stem + '_' + hashlib.md5(img_str.encode()).hexdigest()[:8]
    shutil.copy2(img_str, io/(stem+'.jpg')); (lo/(stem+'.txt')).write_text('\n'.join(lines)+'\n'); n += 1
print(f"  test: {n} images (4-class)")


# === CELL 2 — MoE EVAL (mass full-image + calc SAHI) ========================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sahi", "torchmetrics", "ultralytics"], check=False)
import cv2, glob, torch, numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

SRC = Path("/kaggle/working/canonical_moe")
CLASS_NAMES = ['benign_mass','malignant_mass','benign_calc','malignant_calc']; NC = 4
HIT_IOU = 0.30; TILE, OVERLAP = 1024, 0.25

# --- auto-find the two best.pt among attached inputs ---
all_pt = glob.glob("/kaggle/input/**/best.pt", recursive=True) + glob.glob("/kaggle/input/**/weights/*.pt", recursive=True)
MASS_PT = next((p for p in all_pt if 'mass' in p.lower() and 'calc' not in p.lower()), None)
CALC_PT = next((p for p in all_pt if 'calc' in p.lower()), None)
assert MASS_PT and CALC_PT, f"need both weights. found: {all_pt}"
print("MASS_PT:", MASS_PT, "\nCALC_PT:", CALC_PT)

def yl(line, W, H):
    c, xc, yc, bw, bh = line.split()
    xc, yc, bw, bh = float(xc)*W, float(yc)*H, float(bw)*W, float(bh)*H
    return int(c), [xc-bw/2, yc-bh/2, xc+bw/2, yc+bh/2]
def iou(a, b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix1-ix0)*max(0,iy1-iy0)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
def froc(all_preds, all_gts, n_img, cls=None, fp_points=(0.5,1,2,4)):
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

mass_model = YOLO(MASS_PT)
calc_det = AutoDetectionModel.from_pretrained(model_type="ultralytics",
    model_path=CALC_PT, confidence_threshold=0.01, device="cuda:0")

test_imgs = sorted([p for p in (SRC/"images"/"test").glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}])
n_img = len(test_imgs)
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
all_preds, all_gts = [], []

for i, ip in enumerate(tqdm(test_imgs, desc="MoE test")):
    H, W = cv2.imread(str(ip)).shape[:2]
    pb, ps, pl = [], [], []
    # ---- mass expert: full-image, keep ONLY mass classes (0,1) ----
    rm = mass_model.predict(str(ip), imgsz=1024, conf=0.01, iou=0.6, device=0, augment=True, verbose=False)[0]
    for b in rm.boxes:
        c = int(b.cls[0])
        if c in (0, 1):
            box = b.xyxy[0].tolist(); pb.append(box); ps.append(float(b.conf[0])); pl.append(c)
            all_preds.append((i, c, float(b.conf[0]), box))
    # ---- calc expert: SAHI, classes 0,1 -> remap to 2,3 ----
    rc = get_sliced_prediction(str(ip), calc_det, slice_height=TILE, slice_width=TILE,
        overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
        postprocess_type="NMS", postprocess_match_threshold=0.5, verbose=0)
    for o in rc.object_prediction_list:
        c = o.category.id + 2
        box = list(o.bbox.to_xyxy()); pb.append(box); ps.append(o.score.value); pl.append(c)
        all_preds.append((i, c, o.score.value, box))
    # ---- GT ----
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
print("\n========  MoE (mass full + calc SAHI) — CANONICAL TEST  ========")
print(f"  mAP50    = {m['map_50'].item():.4f}")
print(f"  mAP50-95 = {m['map'].item():.4f}")
if "map_per_class" in m and m["map_per_class"].numel() == NC:
    for k, nm in enumerate(CLASS_NAMES):
        print(f"    {nm:18s} AP50-95 = {m['map_per_class'][k].item():.4f}")
print(f"  FROC (hit IoU={HIT_IOU}) — sensitivity @ FP/image:")
for cls in [None]+list(range(NC)):
    nm="OVERALL" if cls is None else CLASS_NAMES[cls]
    r=froc(all_preds, all_gts, n_img, cls)
    if r: print(f"    {nm:18s} " + "  ".join(f"S@{f}={r[f]:.3f}" for f in (0.5,1,2,4)))
print("================================================================")
print(f"\n>>> GATE: MoE mAP50 = {m['map_50'].item():.4f}  vs  Universal 0.2308")
print(">>> MoE WINS" if m['map_50'].item() > 0.2308 else ">>> MoE does NOT beat universal — see notes")
