# ════════════════════════════════════════════════════════════════════════════
# EXP 21 — CALC EXPERT, FULL-IMAGE (yolo11l, no tiling) — the version that wins
# ════════════════════════════════════════════════════════════════════════════
# Drops the tiled+SAHI pipeline (it collapsed: 0.070 tile -> 0.027 SAHI).
# Same architecture as mass & universal (yolo11l) for a clean comparison.
# EDGE for tiny calcs = higher resolution (imgsz=1280 vs 1024 elsewhere).
# Graded the SAME way in train & test (full image) -> the number won't evaporate.
# Attach ONLY CBIS-DDSM. GPU T4. 3 cells: build -> train -> eval (mAP + FROC).
# Run as Save & Run All (commit).
# ════════════════════════════════════════════════════════════════════════════

# === CELL 1 — BUILD canonical_calc (2-class, full-res, seed=42) ==============
DATASET = "calc"
import hashlib, shutil, pandas as pd, cv2, yaml
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
_cands = [Path('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'),
          Path('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset')]
CBIS = next((c for c in _cands if (c/'csv'/'dicom_info.csv').exists()), None)
assert CBIS is not None, f"CBIS not found: {_cands}"
ROOT = Path(f'/kaggle/working/canonical_{DATASET}'); ROOT.mkdir(parents=True, exist_ok=True); SEED=42
dicom = pd.read_csv(CBIS/'csv/dicom_info.csv')
full_img_map, mask_map = {}, {}
for _, r in dicom.iterrows():
    pid=str(r['PatientID']); desc=str(r['SeriesDescription']).lower(); p=str(CBIS/str(r['image_path']).replace('CBIS-DDSM/',''))
    if 'full mammogram' in desc: full_img_map.setdefault(pid,p)
    elif 'roi mask' in desc: mask_map.setdefault(pid,p)
def build(csv,off):
    df=pd.read_csv(csv); out=[]
    for _,row in df.iterrows():
        ip=full_img_map.get(str(row['image file path']).split('/')[0]); mp=mask_map.get(str(row['ROI mask file path']).split('/')[0])
        if not ip or not mp or not Path(ip).exists() or not Path(mp).exists(): continue
        cls=off+(1 if 'MALIGNANT' in str(row['pathology']).upper() else 0)
        out.append({'img_path':ip,'mask_path':mp,'class_id':cls,'patient_id':str(row['patient_id'])})
    return out
recs  = build(CBIS/'csv/mass_case_description_train_set.csv',0)+build(CBIS/'csv/mass_case_description_test_set.csv',0)
recs += build(CBIS/'csv/calc_case_description_train_set.csv',2)+build(CBIS/'csv/calc_case_description_test_set.csv',2)
rec = pd.DataFrame(recs).drop_duplicates(subset=['img_path','mask_path']).reset_index(drop=True)
tr_i,rest_i = next(GroupShuffleSplit(1,test_size=0.30,random_state=SEED).split(rec,groups=rec['patient_id']))
rest = rec.iloc[rest_i].reset_index(drop=True)
va_i,te_i = next(GroupShuffleSplit(1,test_size=0.50,random_state=SEED).split(rest,groups=rest['patient_id']))
split_of={}
for pid in rec.iloc[tr_i]['patient_id'].unique(): split_of[pid]='train'
for pid in rest.iloc[va_i]['patient_id'].unique(): split_of[pid]='val'
for pid in rest.iloc[te_i]['patient_id'].unique(): split_of[pid]='test'
rec['split']=rec['patient_id'].map(split_of)
P={s:set(rec[rec.split==s].patient_id) for s in ['train','val','test']}
assert P['train'].isdisjoint(P['val']) and P['train'].isdisjoint(P['test']) and P['val'].isdisjoint(P['test']), "LEAKAGE!"
print("patients train/val/test:", len(P['train']),len(P['val']),len(P['test']), "— zero overlap ✅")
CFG = dict(keep=[2,3], remap={2:0,3:1}, names=['benign_calc','malignant_calc'])
def boxes(mp,w,h):
    m=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
    if m is None: return []
    if m.shape!=(h,w): m=cv2.resize(m,(w,h),interpolation=cv2.INTER_NEAREST)
    _,b=cv2.threshold(m,10,255,cv2.THRESH_BINARY)
    if b.max()==0: return []
    cs,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); out=[]
    for c in cs:
        x,y,ww,hh=cv2.boundingRect(c)
        if ww<8 or hh<8 or (ww*hh)/(w*h)<1e-4: continue
        out.append(((x+ww/2)/w,(y+hh/2)/h,ww/w,hh/h))
    return out
sub = rec[rec['class_id'].isin(CFG['keep'])]
for s in ['train','val','test']:
    io=ROOT/'images'/s; lo=ROOT/'labels'/s; io.mkdir(parents=True,exist_ok=True); lo.mkdir(parents=True,exist_ok=True)
    n=0
    for img_str,g in tqdm(sub[sub.split==s].groupby('img_path'),desc=f"calc/{s}"):
        img=cv2.imread(img_str)
        if img is None: continue
        h,w=img.shape[:2]; lines=[]
        for _,row in g.iterrows():
            for bb in boxes(Path(row['mask_path']),w,h):
                lines.append(f"{CFG['remap'][int(row['class_id'])]} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")
        if not lines: continue
        stem=Path(img_str).stem+'_'+hashlib.md5(img_str.encode()).hexdigest()[:8]
        shutil.copy2(img_str,io/(stem+'.jpg')); (lo/(stem+'.txt')).write_text('\n'.join(lines)+'\n'); n+=1
    print(f"  {s}: {n} images")
yaml.dump({'path':str(ROOT),'train':'images/train','val':'images/val','test':'images/test',
           'nc':2,'names':CFG['names']}, open(ROOT/'data.yaml','w'), sort_keys=False)
print("canonical_calc ready")


# === CELL 2 — TRAIN CALC (yolo11l, full-image, imgsz=1280 = the edge) =========
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","ultralytics"], check=True)
from ultralytics import YOLO
model = YOLO("yolo11l.pt")
model.train(
    data="/kaggle/working/canonical_calc/data.yaml",
    imgsz=1280, epochs=120, batch=4, device=0,     # 1280 = resolution edge for tiny calcs
    optimizer="AdamW", lr0=0.0005, lrf=0.01,
    momentum=0.937, weight_decay=0.0005,
    warmup_epochs=10, cos_lr=True, patience=40,
    mosaic=0.0, close_mosaic=10, copy_paste=0.3, mixup=0.0,  # mosaic hurts calc; copy_paste adds calc instances
    fliplr=0.5, flipud=0.5, degrees=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,               # mammograms are grayscale
    scale=0.5, seed=42, deterministic=True, workers=4,
    project="/kaggle/working/canonical_calc/runs", name="calc_fullimg", exist_ok=True, plots=True,
)
print("BEST:", "/kaggle/working/canonical_calc/runs/calc_fullimg/weights/best.pt")
# >>> Save Version. Upload this best.pt (rename calc_best.pt) to weight-moe for the gated eval.


# === CELL 3 — CALC EVAL: full-image mAP + FROC (no SAHI) ======================
subprocess.run([sys.executable,"-m","pip","install","-q","torchmetrics"], check=False)
import cv2, torch, numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
SRC=Path("/kaggle/working/canonical_calc"); BEST="/kaggle/working/canonical_calc/runs/calc_fullimg/weights/best.pt"
CLASS_NAMES=['benign_calc','malignant_calc']; NC=2; HIT_IOU=0.20
def yl(line,W,H):
    c,xc,yc,bw,bh=line.split(); xc,yc,bw,bh=float(xc)*W,float(yc)*H,float(bw)*W,float(bh)*H
    return int(c),[xc-bw/2,yc-bh/2,xc+bw/2,yc+bh/2]
def iou(a,b):
    ix0,iy0=max(a[0],b[0]),max(a[1],b[1]); ix1,iy1=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix1-ix0)*max(0,iy1-iy0); ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
def froc(ap,ag,n_img,cls=None,fps=(0.5,1,2,4)):
    P=sorted([p for p in ap if cls is None or p[1]==cls],key=lambda x:-x[2]); G=[g for g in ag if cls is None or g[1]==cls]; total=len(G)
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
    for f in fps:
        idx=np.where(fppi<=f)[0]; out[f]=float(sens[idx[-1]]) if len(idx) else 0.0
    return out
from ultralytics import YOLO
model=YOLO(BEST); metric=MeanAveragePrecision(iou_type="bbox",class_metrics=True)
test_imgs=sorted([p for p in (SRC/"images"/"test").glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]); n_img=len(test_imgs)
all_preds,all_gts=[],[]
for i,ip in enumerate(tqdm(test_imgs,desc="calc test")):
    H,W=cv2.imread(str(ip)).shape[:2]
    r=model.predict(str(ip),imgsz=1280,conf=0.01,iou=0.6,device=0,augment=True,verbose=False)[0]
    pb,ps,pl=[],[],[]
    for b in r.boxes:
        box=b.xyxy[0].tolist(); pb.append(box); ps.append(float(b.conf[0])); pl.append(int(b.cls[0])); all_preds.append((i,int(b.cls[0]),float(b.conf[0]),box))
    gb,gl=[],[]; lp=SRC/"labels"/"test"/f"{ip.stem}.txt"
    if lp.exists():
        for ln in lp.read_text().splitlines():
            if ln.strip():
                c,box=yl(ln,W,H); gb.append(box); gl.append(c); all_gts.append((i,c,box))
    metric.update([{"boxes":torch.tensor(pb or [],dtype=torch.float).reshape(-1,4),"scores":torch.tensor(ps or [],dtype=torch.float),"labels":torch.tensor(pl or [],dtype=torch.long)}],
                  [{"boxes":torch.tensor(gb or [],dtype=torch.float).reshape(-1,4),"labels":torch.tensor(gl or [],dtype=torch.long)}])
m=metric.compute()
print(f"\n=== CALC (full-image) — CANONICAL TEST ===\n  mAP50={m['map_50'].item():.4f}  mAP50-95={m['map'].item():.4f}")
for k,nm in enumerate(CLASS_NAMES):
    if m['map_per_class'].numel()==NC: print(f"    {nm:18s} AP50-95={m['map_per_class'][k].item():.4f}")
print(f"  FROC (IoU={HIT_IOU}):")
for cls in [None]+list(range(NC)):
    nm="OVERALL" if cls is None else CLASS_NAMES[cls]; rr=froc(all_preds,all_gts,n_img,cls)
    if rr: print(f"    {nm:18s} "+"  ".join(f"S@{f}={rr[f]:.3f}" for f in (0.5,1,2,4)))
print(f"\n>>> calc full-image mAP50={m['map_50'].item():.4f}  (tiled+SAHI was 0.0267, Universal calc ~0.10-0.11)")
