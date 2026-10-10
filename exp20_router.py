# ════════════════════════════════════════════════════════════════════════════
# EXP 20 — ROUTER: per-image mass-vs-calc gate (fixes cross-domain false positives)
# ════════════════════════════════════════════════════════════════════════════
# The naive MoE failed because each disjoint expert fires false positives on the
# OTHER lesion type's images. This router is an image-level classifier that
# predicts P(mass-case) vs P(calc-case); at MoE inference each expert's scores are
# multiplied by its gate prob -> cross-domain FPs are suppressed.
# Trained on the SAME canonical seed=42 split. YOLO11l-cls, fast (~1h).
# Attach ONLY CBIS-DDSM. GPU T4. 2 cells: build cls dataset -> train router.
# ════════════════════════════════════════════════════════════════════════════

# === CELL 1 — BUILD classification dataset (mass vs calc, seed=42) ============
import shutil, pandas as pd, cv2
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm
import hashlib

_cands = [Path('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset'),
          Path('/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset')]
CBIS = next((c for c in _cands if (c/'csv'/'dicom_info.csv').exists()), None)
assert CBIS is not None, f"CBIS not found: {_cands}"
ROOT = Path('/kaggle/working/router_cls'); SEED = 42
dicom = pd.read_csv(CBIS/'csv/dicom_info.csv')
full_img_map = {}
for _, r in dicom.iterrows():
    pid=str(r['PatientID']); desc=str(r['SeriesDescription']).lower()
    if 'full mammogram' in desc:
        full_img_map.setdefault(pid, str(CBIS/str(r['image_path']).replace('CBIS-DDSM/','')))
def build(csv, off):
    df=pd.read_csv(csv); out=[]
    for _,row in df.iterrows():
        ip=full_img_map.get(str(row['image file path']).split('/')[0])
        if not ip or not Path(ip).exists(): continue
        out.append({'img_path':ip,'class_id':off,'patient_id':str(row['patient_id'])})
    return out
recs  = build(CBIS/'csv/mass_case_description_train_set.csv',0)+build(CBIS/'csv/mass_case_description_test_set.csv',0)
recs += build(CBIS/'csv/calc_case_description_train_set.csv',2)+build(CBIS/'csv/calc_case_description_test_set.csv',2)
rec = pd.DataFrame(recs).reset_index(drop=True)
# patient-level split identical to detection pipeline
uniq = rec.drop_duplicates('img_path')
tr_i, rest_i = next(GroupShuffleSplit(1,test_size=0.30,random_state=SEED).split(uniq,groups=uniq['patient_id']))
rest = uniq.iloc[rest_i]
va_i, te_i = next(GroupShuffleSplit(1,test_size=0.50,random_state=SEED).split(rest,groups=rest['patient_id']))
split_of={}
for pid in uniq.iloc[tr_i]['patient_id'].unique(): split_of[pid]='train'
for pid in rest.iloc[va_i]['patient_id'].unique(): split_of[pid]='val'
for pid in rest.iloc[te_i]['patient_id'].unique(): split_of[pid]='test'
rec['split']=rec['patient_id'].map(split_of)
# per-image label: mass if it has only mass lesions, calc if only calc; drop 'both'
img_types = rec.groupby('img_path')['class_id'].apply(lambda s: set(int(x)//2 for x in s))  # 0=mass,1=calc
def label_of(types): return 'mass' if types=={0} else ('calc' if types=={1} else None)
img_split = rec.drop_duplicates('img_path').set_index('img_path')['split']
n_per = {}
for img_str, types in tqdm(img_types.items(), desc="router-cls"):
    lab = label_of(types)
    if lab is None: continue
    sp = img_split[img_str]
    if sp not in ('train','val'): continue   # router trained on train/val
    img = cv2.imread(img_str)
    if img is None: continue
    d = ROOT/sp/lab; d.mkdir(parents=True, exist_ok=True)
    stem = Path(img_str).stem + '_' + hashlib.md5(img_str.encode()).hexdigest()[:8]
    cv2.imwrite(str(d/(stem+'.jpg')), img)
    n_per[(sp,lab)] = n_per.get((sp,lab),0)+1
print("counts:", n_per)
print("router_cls ready at", ROOT)


# === CELL 2 — TRAIN ROUTER (YOLO11l-cls, mass vs calc) =======================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO
model = YOLO("yolo11l-cls.pt")
model.train(
    data="/kaggle/working/router_cls",
    epochs=40, imgsz=512, batch=32, device=0,
    optimizer="AdamW", lr0=0.0005, cos_lr=True, warmup_epochs=3,
    fliplr=0.5, flipud=0.5, seed=42, patience=15,
    project="/kaggle/working/router_cls/runs", name="router", exist_ok=True, plots=True,
)
print("ROUTER BEST:", "/kaggle/working/router_cls/runs/router/weights/best.pt")
# >>> Save Version. Then upload this best.pt (rename router_best.pt) to weight-moe for the gated eval.
