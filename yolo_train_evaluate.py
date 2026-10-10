"""
CBIS-DDSM Mammogram Lesion Detection — YOLO Training & Evaluation
Stage 2: Train strongest YOLO baseline + publication-quality metrics
Paste each CELL block into a separate Kaggle notebook cell.
"""

# ══════════════════════════════════════════════════════════════════════
# CELL 1 │ Install & GPU Verification
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
import subprocess
subprocess.run(['pip', 'install', 'ultralytics', '-q', '--upgrade'], check=True)

import torch
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}   : {p.name}  |  {p.total_memory / 1e9:.1f} GB VRAM")

import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")
ultralytics.checks()
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 2 │ Imports + Publication Style + Directory Setup
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
import os, json, shutil, warnings, time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ── Publication-quality matplotlib settings ───────────────────────────
plt.rcParams.update({
    'font.family'        : 'DejaVu Sans',
    'font.size'          : 12,
    'axes.labelsize'     : 13,
    'axes.titlesize'     : 14,
    'axes.titleweight'   : 'bold',
    'axes.titlepad'      : 10,
    'xtick.labelsize'    : 11,
    'ytick.labelsize'    : 11,
    'legend.fontsize'    : 11,
    'legend.framealpha'  : 0.9,
    'legend.edgecolor'   : '#cccccc',
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.1,
    'axes.spines.top'    : False,
    'axes.spines.right'  : False,
    'axes.grid'          : True,
    'grid.alpha'         : 0.3,
    'grid.linestyle'     : '--',
    'grid.linewidth'     : 0.7,
    'lines.linewidth'    : 2.2,
    'patch.linewidth'    : 0.8,
})

# ── Paths ─────────────────────────────────────────────────────────────
DATA_YAML   = Path('/kaggle/working/yolo_cbis/data.yaml')
BASE_DIR    = Path('/kaggle/working/experiment')
WEIGHTS_DIR = BASE_DIR / 'weights'
METRICS_DIR = BASE_DIR / 'metrics'
PLOTS_DIR   = BASE_DIR / 'plots'
PREDS_DIR   = BASE_DIR / 'val_predictions'
RUNS_DIR    = BASE_DIR / 'runs'

for d in [WEIGHTS_DIR, METRICS_DIR, PLOTS_DIR, PREDS_DIR, RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['benign', 'malignant']
NC          = len(CLASS_NAMES)
PALETTE     = ['#2196F3', '#F44336']    # blue=benign, red=malignant
PALETTE_DARK= ['#1565C0', '#B71C1C']

print("Setup complete.")
print(f"data.yaml exists: {DATA_YAML.exists()}")
for d in [WEIGHTS_DIR, METRICS_DIR, PLOTS_DIR, PREDS_DIR]:
    print(f"  {d}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 3 │ Train YOLO11x  (Transfer Learning from COCO)
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
model = YOLO('yolo11x.pt')   # pretrained COCO weights → transfer learning
print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")

t_start = time.time()

train_results = model.train(
    data          = str(DATA_YAML),
    epochs        = 100,
    patience      = 20,        # early stopping — no improvement for 20 epochs
    imgsz         = 640,
    batch         = -1,        # auto-batch: fills ~60 % GPU VRAM
    device        = 0,
    optimizer     = 'AdamW',
    lr0           = 0.001,     # initial LR
    lrf           = 0.01,      # final LR = lr0 * lrf (cosine decay target)
    momentum      = 0.937,
    weight_decay  = 0.0005,
    warmup_epochs = 3,
    cos_lr        = True,      # cosine LR schedule
    amp           = True,      # mixed precision FP16
    # ── Augmentation ──────────────────────────────
    degrees       = 10,        # rotation ±10°
    flipud        = 0.3,       # vertical flip (anatomically valid for MLO/CC)
    fliplr        = 0.5,       # horizontal flip
    hsv_h         = 0.015,
    hsv_s         = 0.1,       # low sat augment (mammograms are near-grayscale)
    hsv_v         = 0.4,
    mosaic        = 1.0,
    mixup         = 0.1,
    copy_paste    = 0.0,
    close_mosaic  = 10,        # disable mosaic last 10 epochs
    # ── Output ────────────────────────────────────
    project       = str(RUNS_DIR),
    name          = 'yolo11x_cbis',
    exist_ok      = True,
    save          = True,
    save_period   = 10,        # checkpoint every 10 epochs
    plots         = True,
    verbose       = True,
    seed          = 42,
    workers       = 2,
    conf          = 0.001,     # low conf threshold for mAP sweep
    iou           = 0.6,
    single_cls    = False,
)

elapsed = time.time() - t_start
print(f"\nTotal training time : {elapsed/3600:.2f} h")

SAVE_DIR = Path(train_results.save_dir)
BEST_PT  = SAVE_DIR / 'weights' / 'best.pt'

shutil.copy2(BEST_PT, WEIGHTS_DIR / 'best.pt')
shutil.copy2(SAVE_DIR / 'weights' / 'last.pt', WEIGHTS_DIR / 'last.pt')
print(f"Best weights saved : {WEIGHTS_DIR / 'best.pt'}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 4 │ Evaluate Best Model on Val + Test Splits
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
# If running after kernel restart, set SAVE_DIR manually:
# SAVE_DIR = Path('/kaggle/working/experiment/runs/yolo11x_cbis')

model = YOLO(str(WEIGHTS_DIR / 'best.pt'))

def run_eval(split: str):
    print(f"\n{'─'*50}")
    print(f"Evaluating: {split}")
    m = model.val(
        data      = str(DATA_YAML),
        split     = split,
        imgsz     = 640,
        batch     = 32,
        conf      = 0.001,
        iou       = 0.6,
        device    = 0,
        save_json = True,
        plots     = True,
        project   = str(RUNS_DIR),
        name      = f'eval_{split}',
        exist_ok  = True,
        verbose   = False,
    )
    return m

val_metrics  = run_eval('val')
test_metrics = run_eval('test')

# ── Extract per-class + aggregate metrics ─────────────────────────────
def extract(m, split_name: str) -> dict:
    b   = m.box
    p   = np.array(b.p).flatten()
    r   = np.array(b.r).flatten()
    f1  = np.array(b.f1).flatten()
    ap50   = np.array(b.ap50).flatten()
    ap5095 = np.array(b.ap).flatten()

    row = {
        'split'     : split_name,
        'mP'        : float(b.mp),
        'mR'        : float(b.mr),
        'mF1'       : float(np.mean(f1)) if len(f1) else 0.0,
        'mAP50'     : float(b.map50),
        'mAP50-95'  : float(b.map),
    }
    for i, cls in enumerate(CLASS_NAMES):
        safe = lambda arr, idx: float(arr[idx]) if idx < len(arr) else None
        row[f'{cls}_P']    = safe(p,    i)
        row[f'{cls}_R']    = safe(r,    i)
        row[f'{cls}_F1']   = safe(f1,   i)
        row[f'{cls}_AP50'] = safe(ap50,   i)
        row[f'{cls}_AP']   = safe(ap5095, i)
    return row

records = [extract(val_metrics, 'val'), extract(test_metrics, 'test')]
metrics_df = pd.DataFrame(records)
metrics_df.to_csv(METRICS_DIR / 'metrics.csv', index=False)

with open(METRICS_DIR / 'metrics.json', 'w') as f:
    json.dump({'val': records[0], 'test': records[1]}, f, indent=2)

print('\n' + '═'*60)
print('EVALUATION RESULTS')
print('═'*60)
print(metrics_df.to_string(index=False))
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 5 │ Figure 1 — Training Curves
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
results_csv = SAVE_DIR / 'results.csv'
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()

# Map expected column name fragments
def _col(df, *keywords):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keywords):
            return c
    return None

ep       = df.iloc[:, 0].values   # epoch column (first col)

c_tbox   = _col(df, 'train', 'box')
c_tcls   = _col(df, 'train', 'cls')
c_tdfl   = _col(df, 'train', 'dfl')
c_vbox   = _col(df, 'val', 'box')
c_vcls   = _col(df, 'val', 'cls')
c_map50  = _col(df, 'map50') or _col(df, 'mAP50')
c_map    = _col(df, 'map50-95') or _col(df, 'mAP50-95')
c_prec   = _col(df, 'precision')
c_rec    = _col(df, 'recall')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('YOLO11x Training Curves — CBIS-DDSM Mammogram Dataset', fontsize=15, y=1.01)

def _plot(ax, col, label, color, title, ylabel):
    if col and col in df.columns:
        ax.plot(ep, df[col].values, color=color, label=label)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend()

# Row 0: losses
for ax, col, lbl, color, title, ylab in [
    (axes[0,0], c_tbox, 'Train Box Loss', '#1E88E5', 'Box Loss',            'Loss'),
    (axes[0,1], c_tcls, 'Train Cls Loss', '#E53935', 'Classification Loss', 'Loss'),
    (axes[0,2], c_tdfl, 'Train DFL Loss', '#43A047', 'DFL Loss',            'Loss'),
]:
    if col:
        axes_obj = ax
        axes_obj.plot(ep, df[col].values, color=color,    label=lbl,        linewidth=2)
        if col == c_tbox and c_vbox:
            axes_obj.plot(ep, df[c_vbox].values, color=color, linestyle='--', label='Val Box Loss', alpha=0.8)
        if col == c_tcls and c_vcls:
            axes_obj.plot(ep, df[c_vcls].values, color=color, linestyle='--', label='Val Cls Loss', alpha=0.8)
        axes_obj.set_title(title, fontweight='bold')
        axes_obj.set_xlabel('Epoch'); axes_obj.set_ylabel(ylab)
        axes_obj.legend(fontsize=9)

# Row 1: detection metrics
if c_map50 and c_map:
    axes[1,0].plot(ep, df[c_map50].values,  color='#7B1FA2', label='mAP@50')
    axes[1,0].plot(ep, df[c_map].values,    color='#E65100', label='mAP@50-95', linestyle='--')
    axes[1,0].set_title('Mean Average Precision', fontweight='bold')
    axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('mAP')
    axes[1,0].set_ylim(0, 1); axes[1,0].legend()

if c_prec and c_rec:
    axes[1,1].plot(ep, df[c_prec].values, color='#00897B', label='Precision')
    axes[1,1].plot(ep, df[c_rec].values,  color='#FB8C00', label='Recall', linestyle='--')
    axes[1,1].set_title('Precision & Recall', fontweight='bold')
    axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Score')
    axes[1,1].set_ylim(0, 1); axes[1,1].legend()

# F1 = 2PR/(P+R)
if c_prec and c_rec:
    p_arr = df[c_prec].values; r_arr = df[c_rec].values
    f1_arr = np.where((p_arr + r_arr) > 0, 2*p_arr*r_arr/(p_arr+r_arr), 0)
    axes[1,2].plot(ep, f1_arr, color='#D81B60', label='F1-Score')
    axes[1,2].set_title('F1 Score', fontweight='bold')
    axes[1,2].set_xlabel('Epoch'); axes[1,2].set_ylabel('F1')
    axes[1,2].set_ylim(0, 1); axes[1,2].legend()

plt.tight_layout()
out = PLOTS_DIR / 'fig1_training_curves.png'
plt.savefig(out, dpi=300)
plt.savefig(PLOTS_DIR / 'fig1_training_curves.pdf')
plt.show()
print(f"Saved: {out}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 6 │ Figure 2 — Per-Class & Aggregate Metrics Bar Chart
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
row = records[1]   # test split metrics

metrics_show = ['P', 'R', 'F1', 'AP50', 'AP']
metric_labels = ['Precision', 'Recall', 'F1', 'AP@50', 'AP@50-95']
x = np.arange(len(metrics_show))
width = 0.28

fig, ax = plt.subplots(figsize=(11, 5))

for i, (cls, color, dcolor) in enumerate(zip(CLASS_NAMES, PALETTE, PALETTE_DARK)):
    vals = [row.get(f'{cls}_{m}') or 0.0 for m in metrics_show]
    bars = ax.bar(x + (i - 0.5) * width, vals, width,
                  label=cls.capitalize(), color=color,
                  edgecolor=dcolor, linewidth=0.8, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Mean line
mean_vals = []
for m in metrics_show:
    vs = [row.get(f'{cls}_{m}') or 0.0 for cls in CLASS_NAMES]
    mean_vals.append(np.mean(vs))
ax.plot(x, mean_vals, 'D--', color='#212121', markersize=7,
        linewidth=1.5, label='Mean', zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=12)
ax.set_ylabel('Score', fontsize=13)
ax.set_ylim(0, 1.12)
ax.set_title('Per-Class Detection Performance — Test Set', fontweight='bold', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
ax.grid(axis='y', alpha=0.35, linestyle='--')
ax.set_axisbelow(True)

# Aggregate text box
agg_text = (f"mAP@50: {row['mAP50']:.4f}   "
            f"mAP@50-95: {row['mAP50-95']:.4f}   "
            f"mF1: {row['mF1']:.4f}")
fig.text(0.5, -0.02, agg_text, ha='center', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#90CAF9'))

plt.tight_layout()
out = PLOTS_DIR / 'fig2_per_class_metrics.png'
plt.savefig(out, dpi=300)
plt.savefig(PLOTS_DIR / 'fig2_per_class_metrics.pdf')
plt.show()
print(f"Saved: {out}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 7 │ Figure 3 — Confusion Matrix (Normalised + Raw)
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
def get_confusion_matrix(metrics_obj):
    """Extract nc×nc confusion matrix from Ultralytics metrics."""
    cm_obj = metrics_obj.confusion_matrix
    matrix = np.array(cm_obj.matrix)      # shape (nc+1, nc+1) — last = background
    return matrix[:NC, :NC].astype(int)   # drop background row/col

raw_cm = get_confusion_matrix(test_metrics)

# Row = actual, Col = predicted  (transpose from YOLO convention)
# YOLO stores (predicted, actual) → transpose to (actual, predicted)
raw_cm_display = raw_cm.T

norm_cm = raw_cm_display.astype(float)
row_sums = norm_cm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
norm_cm = norm_cm / row_sums

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Confusion Matrix — Test Set (CBIS-DDSM)', fontsize=14, fontweight='bold')

tick_labels = [c.capitalize() for c in CLASS_NAMES]

for ax, data, title, fmt, vmax in [
    (axes[0], raw_cm_display, 'Raw Counts',            'd',    None),
    (axes[1], norm_cm,        'Row-Normalised (Recall)','0.2f', 1.0),
]:
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        data, ax=ax, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=tick_labels, yticklabels=tick_labels,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 14, 'weight': 'bold'},
        vmin=0, vmax=vmax,
        cbar_kws={'shrink': 0.82},
    )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label',      fontsize=12)
    ax.tick_params(labelsize=11)

# Print per-class recall from normalised CM diagonal
print("Per-class Recall (from CM):")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:12s}: {norm_cm[i,i]:.4f}")

plt.tight_layout()
out = PLOTS_DIR / 'fig3_confusion_matrix.png'
plt.savefig(out, dpi=300)
plt.savefig(PLOTS_DIR / 'fig3_confusion_matrix.pdf')
plt.show()
print(f"Saved: {out}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 8 │ Figure 4 — PR Curves per Class
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
def get_pr_curves(metrics_obj):
    """
    Returns (px, py_per_class, ap50_per_class).
    Falls back gracefully if internal arrays are unavailable.
    """
    b = metrics_obj.box
    try:
        px = np.array(b.px).flatten()           # recall axis (0→1)
        py = np.array(b.prec_values)            # (nc, len(px))
        if py.ndim == 1:
            py = py.reshape(1, -1)
        ap50 = np.array(b.ap50).flatten()
        return px, py, ap50
    except AttributeError:
        pass
    try:
        # Alternative attribute name in some Ultralytics versions
        curves = b.curves_results                # list of (x, y, names)
        if curves:
            px, py, _ = curves[0]
            px = np.array(px); py = np.array(py)
            ap50 = np.array(b.ap50).flatten()
            return px, py, ap50
    except (AttributeError, TypeError, IndexError):
        pass
    return None, None, None

px, py, ap50 = get_pr_curves(test_metrics)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Precision-Recall Curves — Test Set (CBIS-DDSM)', fontsize=14, fontweight='bold')

if px is not None and py is not None:
    # Individual class curves
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, PALETTE)):
        if i < py.shape[0]:
            prec = py[i]
            label = f'{cls.capitalize()}  (AP@50 = {ap50[i]:.3f})' if ap50 is not None and i < len(ap50) else cls.capitalize()
            axes[0].plot(px, prec, color=color, linewidth=2.5, label=label)

    axes[0].set_xlabel('Recall',    fontsize=13)
    axes[0].set_ylabel('Precision', fontsize=13)
    axes[0].set_title('Per-Class PR Curves',  fontweight='bold')
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc='lower left')

    # Mean PR curve
    mean_py = py.mean(axis=0)
    mean_ap = float(ap50.mean()) if ap50 is not None else 0.0
    axes[1].fill_between(px, mean_py, alpha=0.18, color='#7B1FA2')
    axes[1].plot(px, mean_py, color='#7B1FA2', linewidth=2.5,
                 label=f'Mean (mAP@50 = {mean_ap:.3f})')
    axes[1].set_xlabel('Recall',    fontsize=13)
    axes[1].set_ylabel('Precision', fontsize=13)
    axes[1].set_title('Mean PR Curve', fontweight='bold')
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc='lower left')

else:
    # Fallback: bar chart of AP50 per class if curve data unavailable
    for ax in axes:
        ax.bar(CLASS_NAMES, [records[1].get(f'{c}_AP50', 0) for c in CLASS_NAMES],
               color=PALETTE, edgecolor='#333', linewidth=0.8)
        ax.set_ylabel('AP@50'); ax.set_ylim(0, 1)
        ax.set_title('AP@50 per Class (PR curve data unavailable)')
    print("Note: PR curve raw data not available — showing AP@50 bar chart instead.")

plt.tight_layout()
out = PLOTS_DIR / 'fig4_pr_curves.png'
plt.savefig(out, dpi=300)
plt.savefig(PLOTS_DIR / 'fig4_pr_curves.pdf')
plt.show()
print(f"Saved: {out}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 9 │ Figure 5 — Validation Prediction Samples
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
VAL_IMG_DIR = Path('/kaggle/working/yolo_cbis/images/val')
sample_imgs = sorted(VAL_IMG_DIR.glob('*.jpg'))[:16]   # 4×4 grid

model = YOLO(str(WEIGHTS_DIR / 'best.pt'))

BBOX_COLORS = {'benign': (33, 150, 243), 'malignant': (244, 67, 54)}

def draw_boxes(img_path: Path, results_list) -> Image.Image:
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for r in results_list:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            cls_id  = int(box.cls[0])
            conf    = float(box.conf[0])
            if conf < 0.25:
                continue
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            color = BBOX_COLORS.get(cls_name, (255, 255, 0))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            lw = max(2, int(min(W, H) * 0.003))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            label = f'{cls_name} {conf:.2f}'
            tw, th = draw.textlength(label), 12
            draw.rectangle([x1, y1 - th - 4, x1 + int(tw) + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 2), label, fill=(255,255,255))
    return img

# Run inference
preds = model.predict(
    source=[str(p) for p in sample_imgs],
    conf=0.25, iou=0.6, device=0, verbose=False
)

# Build 4×4 grid
GRID_ROWS, GRID_COLS = 4, 4
THUMB_SIZE = 320

fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                          figsize=(GRID_COLS * 3.2, GRID_ROWS * 3.2))
fig.suptitle('Sample Validation Predictions — YOLO11x on CBIS-DDSM\n'
             '(Blue = Benign · Red = Malignant)',
             fontsize=13, fontweight='bold')

for idx in range(GRID_ROWS * GRID_COLS):
    ax = axes[idx // GRID_COLS][idx % GRID_COLS]
    ax.axis('off')
    if idx >= len(sample_imgs):
        continue
    img_with_boxes = draw_boxes(sample_imgs[idx], [preds[idx]])
    img_with_boxes = img_with_boxes.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    ax.imshow(np.array(img_with_boxes))

plt.tight_layout()
out = PLOTS_DIR / 'fig5_val_predictions.png'
plt.savefig(out, dpi=200)
plt.show()
print(f"Saved: {out}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 10 │ Figure 6 — Summary Dashboard
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --
fig = plt.figure(figsize=(14, 8))
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

fig.suptitle('YOLO11x Performance Summary — CBIS-DDSM Lesion Detection',
             fontsize=15, fontweight='bold', y=1.01)

# ── Panel A: mAP comparison val vs test ──────────────────────────────
ax_map = fig.add_subplot(gs[0, 0])
splits_data = [records[0], records[1]]   # val, test
splits_lab  = ['Validation', 'Test']
met_keys    = ['mAP50', 'mAP50-95', 'mP', 'mR', 'mF1']
met_labels  = ['mAP@50', 'mAP@50-95', 'mPrec', 'mRec', 'mF1']

x = np.arange(len(met_keys)); w = 0.35
for i, (split_rec, slabel, color) in enumerate(
    zip(splits_data, splits_lab, ['#1E88E5', '#E53935'])
):
    vals = [split_rec.get(k, 0) for k in met_keys]
    ax_map.bar(x + (i - 0.5) * w, vals, w,
               label=slabel, color=color, alpha=0.85,
               edgecolor='#333', linewidth=0.7)

ax_map.set_xticks(x); ax_map.set_xticklabels(met_labels, rotation=30, ha='right', fontsize=9)
ax_map.set_ylim(0, 1.1); ax_map.set_ylabel('Score')
ax_map.set_title('Val vs Test Metrics', fontweight='bold')
ax_map.legend(fontsize=9)

# ── Panel B: per-class AP@50 radar-style bar ─────────────────────────
ax_cls = fig.add_subplot(gs[0, 1])
cls_met = ['AP50', 'AP', 'F1']
cls_met_labels = ['AP@50', 'AP@50-95', 'F1']
x2 = np.arange(len(cls_met))
for i, (cls, color, dcolor) in enumerate(zip(CLASS_NAMES, PALETTE, PALETTE_DARK)):
    vals = [records[1].get(f'{cls}_{m}', 0) for m in cls_met]
    ax_cls.bar(x2 + (i - 0.5)*0.35, vals, 0.35,
               label=cls.capitalize(), color=color,
               edgecolor=dcolor, linewidth=0.7, zorder=3)
    for j, (xp, v) in enumerate(zip(x2 + (i-0.5)*0.35, vals)):
        ax_cls.text(xp, v + 0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

ax_cls.set_xticks(x2); ax_cls.set_xticklabels(cls_met_labels)
ax_cls.set_ylim(0, 1.1); ax_cls.set_ylabel('Score')
ax_cls.set_title('Per-Class (Test)', fontweight='bold')
ax_cls.legend(fontsize=9)

# ── Panel C: training mAP curve ──────────────────────────────────────
ax_trn = fig.add_subplot(gs[0, 2])
if c_map50 and c_map:
    ax_trn.plot(ep, df[c_map50].values,  color='#7B1FA2', label='mAP@50',    linewidth=2)
    ax_trn.plot(ep, df[c_map].values,    color='#E65100', label='mAP@50-95', linewidth=2, linestyle='--')
    ax_trn.set_xlabel('Epoch'); ax_trn.set_ylabel('mAP')
    ax_trn.set_ylim(0, 1); ax_trn.legend(fontsize=9)
ax_trn.set_title('mAP During Training', fontweight='bold')

# ── Panel D: confusion matrix (normalised) ───────────────────────────
ax_cm = fig.add_subplot(gs[1, :2])
sns.heatmap(norm_cm, ax=ax_cm, annot=True, fmt='0.2f',
            cmap='Blues', linewidths=0.5, linecolor='white',
            xticklabels=[c.capitalize() for c in CLASS_NAMES],
            yticklabels=[c.capitalize() for c in CLASS_NAMES],
            annot_kws={'size': 12, 'weight': 'bold'},
            vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
ax_cm.set_title('Normalised Confusion Matrix (Test)', fontweight='bold')
ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')

# ── Panel E: key metrics text box ────────────────────────────────────
ax_txt = fig.add_subplot(gs[1, 2])
ax_txt.axis('off')
r = records[1]
lines = [
    'TEST SET SUMMARY',
    '─' * 28,
    f"mAP@50     : {r['mAP50']:.4f}",
    f"mAP@50-95  : {r['mAP50-95']:.4f}",
    f"Precision  : {r['mP']:.4f}",
    f"Recall     : {r['mR']:.4f}",
    f"F1 Score   : {r['mF1']:.4f}",
    '',
    'PER CLASS',
    '─' * 28,
]
for cls in CLASS_NAMES:
    lines.append(f"{cls.capitalize()}")
    lines.append(f"  P={r.get(f'{cls}_P',0):.3f}  R={r.get(f'{cls}_R',0):.3f}")
    lines.append(f"  F1={r.get(f'{cls}_F1',0):.3f}  AP50={r.get(f'{cls}_AP50',0):.3f}")
    lines.append('')

text = '\n'.join(lines)
ax_txt.text(0.05, 0.97, text, transform=ax_txt.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F3E5F5', edgecolor='#CE93D8', alpha=0.9))

out = PLOTS_DIR / 'fig6_dashboard.png'
plt.savefig(out, dpi=300)
plt.savefig(PLOTS_DIR / 'fig6_dashboard.pdf')
plt.show()
print(f"Saved: {out}")
# -- end cell --


# ══════════════════════════════════════════════════════════════════════
# CELL 11 │ Save Everything + Print Final Summary
# ══════════════════════════════════════════════════════════════════════
# -- paste from here --

# Copy Ultralytics auto-generated plots (already exist in SAVE_DIR)
for auto_plot in (SAVE_DIR / 'eval_test').glob('*.png'):
    shutil.copy2(auto_plot, PLOTS_DIR / f'ultralytics_{auto_plot.name}')

# Save training CSV
shutil.copy2(results_csv, METRICS_DIR / 'training_history.csv')

# Write experiment config
cfg_log = {
    'model'         : 'yolo11x.pt (pretrained COCO)',
    'dataset'       : 'CBIS-DDSM',
    'classes'       : CLASS_NAMES,
    'imgsz'         : 640,
    'optimizer'     : 'AdamW',
    'lr0'           : 0.001,
    'amp'           : True,
    'early_stopping': 'patience=20',
    'seed'          : 42,
    'train_images'  : 2190,
    'val_images'    : 455,
    'test_images'   : 458,
    'total_bboxes'  : 4875,
}
with open(METRICS_DIR / 'experiment_config.json', 'w') as f:
    json.dump(cfg_log, f, indent=2)

# ── Final summary ─────────────────────────────────────────────────────
print('═' * 60)
print('EXPERIMENT COMPLETE')
print('═' * 60)
print(f"\n  Best weights : {WEIGHTS_DIR}/best.pt")
print(f"  Metrics      : {METRICS_DIR}/metrics.csv")
print(f"  Plots        : {PLOTS_DIR}/")
print()
print('  Test Performance:')
r = records[1]
print(f"    mAP@50      : {r['mAP50']:.4f}")
print(f"    mAP@50-95   : {r['mAP50-95']:.4f}")
print(f"    Precision   : {r['mP']:.4f}")
print(f"    Recall      : {r['mR']:.4f}")
print(f"    F1 Score    : {r['mF1']:.4f}")
print()
print('  Plots generated:')
for f in sorted(PLOTS_DIR.glob('fig*.png')):
    print(f"    {f.name}")
print()
print('  Awaiting next command.')
# -- end cell --
