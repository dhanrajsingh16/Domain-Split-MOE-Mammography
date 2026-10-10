import matplotlib.pyplot as plt
import matplotlib.patches as patches

def rbox(ax, cx, cy, w, h, txt, fc='#2B6CB0', tc='white', fs=8, bold=False):
    ax.add_patch(patches.FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.1",
        facecolor=fc, edgecolor='white', linewidth=1.5, zorder=3))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold' if bold else 'normal',
            zorder=4, multialignment='center', linespacing=1.4)

def arr(ax, x1, y1, x2, y2, c='#555', lw=2.0):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=15), zorder=5)

def hline(ax, x1, x2, y, c='#777', lw=1.8):
    ax.plot([x1,x2],[y,y], color=c, lw=lw, zorder=5)

def shade(ax, y_top, y_bot, fc):
    ax.add_patch(patches.FancyBboxPatch(
        (0.2, y_bot), 23.6, y_top-y_bot,
        boxstyle="round,pad=0.1", facecolor=fc,
        edgecolor='none', alpha=0.13, zorder=1))

def slbl(ax, y, txt, c):
    ax.text(0.4, y, txt, ha='left', va='center', fontsize=8.5,
            color=c, fontweight='bold', zorder=6)

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 70))
ax.set_xlim(0, 24); ax.set_ylim(0, 70); ax.axis('off')
fig.patch.set_facecolor('#F4F6F8'); ax.set_facecolor('#F4F6F8')

# TITLE
ax.text(12, 69.3, 'Mammography MoE Detection — Full Experiment Pipeline',
        ha='center', fontsize=20, fontweight='bold', color='#1A202C')
ax.text(12, 68.6, 'AAAI 2027  ·  CBIS-DDSM  ·  YOLO11l  ·  Leakage-Free  ·  Domain-Split MoE vs Universal',
        ha='center', fontsize=10.5, color='#4A5568', style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — RAW DATA    shade: 67.0 – 68.2
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 68.2, 67.0, '#2B6CB0')
slbl(ax, 67.95, '  STAGE 1 · RAW DATA SOURCE', '#2B6CB0')
rbox(ax, 12, 67.35, 10, 0.85,
     'CBIS-DDSM Dataset  ·  ~3,500 Mammograms  ·  US Screen-Film X-rays  ·  Expert Radiologist Annotations',
     fc='#2B6CB0', fs=9.5, bold=True)

arr(ax, 12, 66.92, 12, 66.22)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — PATIENT SPLIT    shade: 63.8 – 66.2
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 66.2, 63.8, '#276749')
slbl(ax, 65.95, '  STAGE 2 · LEAKAGE-FREE PATIENT SPLIT', '#276749')
rbox(ax, 12, 65.15, 13, 0.9,
     'Patient-Level GroupShuffleSplit\nSeeds: 42  |  1  |  123     ←     3 independent splits, each fully leakage-free',
     fc='#276749', fs=9.5, bold=True)
ax.text(12, 64.3,
        'All images from ONE patient go to EXACTLY ONE split  ·  No patient ever appears in both Train and Test',
        ha='center', fontsize=8.5, color='#276749', style='italic')

# fan arrows to 4 datasets
DS_X = [3.2, 8.5, 15.5, 20.8]
for x in DS_X:
    arr(ax, 12, 63.85, x, 63.22, c='#999', lw=1.4)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — DATASETS    shade: 57.0 – 63.2
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 63.2, 57.0, '#744210')
slbl(ax, 62.95, '  STAGE 3 · CANONICAL DATASETS  (all built from the same patient split)', '#744210')

DS_LBL = [
    'canonical_mass_clean\n─────────────────\n604  train\n  26  val\n  26  test\n─────────────────\nnc = 2\nbenign_mass\nmalign_mass',
    'canonical_calc_clean\n─────────────────\n483  train\n  17  val\n  18  test\n─────────────────\nnc = 2\nbenign_calc\nmalign_calc',
    'canonical_4class\n─────────────────\n2190  train\n  455  val\n  458  test\n─────────────────\nnc = 4\nAll 4 classes',
    'canonical_router_clean\n─────────────────\n760  train\n162  val\n165  test\n─────────────────\nnc = 2 (CLS)\ncalc=0 / mass=1',
]
DS_COL = ['#2C5282','#22543D','#8B4513','#553C9A']

for x, lbl, col in zip(DS_X, DS_LBL, DS_COL):
    rbox(ax, x, 59.8, 4.5, 4.7, lbl, fc=col, fs=8.5)

# arrows down to training
for x in DS_X:
    arr(ax, x, 57.45, x, 56.72, c='#999', lw=1.4)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — TRAINING    shade: 49.5 – 56.7
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 56.7, 49.5, '#1A365D')
slbl(ax, 56.45, '  STAGE 4 · MODEL TRAINING  (RTX 5090 · 32 GB VRAM · conda: torch_gap)', '#1A365D')

M_LBL = [
    'YOLO11l\n─────────────────\nMASS EXPERT\nnc = 2\nimgsz = 1280\n~100 epochs\n─────────────────\nOutputs:\nmass_clean_seed42\nmass_clean_seed1\nmass_clean_seed123',
    'YOLO11l\n─────────────────\nCALC EXPERT\nnc = 2\nimgsz = 1280\n~120 epochs\n─────────────────\nOutputs:\ncalc_clean_seed42\ncalc_clean_seed1\ncalc_clean_seed123',
    'YOLO11l\n─────────────────\nUNIVERSAL\nnc = 4\nimgsz = 1280\n~100 epochs\n─────────────────\nOutputs:\nuniversal_seed42\nuniversal_seed1\nuniversal_seed123',
    'YOLO11l-cls\n─────────────────\nROUTER\nnc = 2 (CLS)\nimgsz = 640\n50 epochs\n─────────────────\nOutputs:\nrouter_seed42\nrouter_seed1\nrouter_seed123',
]
VAL = ['Val mAP50\n= 0.499 mean','Val mAP50\n= 0.167 mean','Val mAP50\n= 0.062 mean','Val top1\n= 0.712 mean']

for x, lbl, col, v in zip(DS_X, M_LBL, DS_COL, VAL):
    rbox(ax, x, 53.6, 4.5, 5.0, lbl, fc=col, fs=8.5)
    rbox(ax, x, 50.5, 4.0, 0.75, v, fc='#2D3748', tc='#E2E8F0', fs=8.5)

for x in DS_X:
    arr(ax, x, 50.12, x, 49.55, c='#999', lw=1.4)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — INFERENCE    shade: 30.0 – 49.5
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 49.5, 30.0, '#702459')
slbl(ax, 49.25, '  STAGE 5 · TEST-TIME INFERENCE  (458 test images · 518 ground-truth boxes)', '#702459')

# converge 4 datasets to test set
hline(ax, DS_X[0], DS_X[-1], 49.55)
arr(ax, 12, 49.55, 12, 48.95)
rbox(ax, 12, 48.55, 7, 0.75,
     'TEST SET  ·  458 images  ·  518 GT boxes', fc='#1A202C', fs=10.5, bold=True)

# fan to 3 options
B_X = [4.5, 12.0, 19.5]
for bx in B_X:
    arr(ax, 12, 48.17, bx, 47.72, c='#aaa', lw=1.5)

# ── Option A: Universal ──────────────────────────────────────────────────────
rbox(ax, 4.5, 47.35, 6.5, 0.65, 'Option A — UNIVERSAL MODEL', fc='#8B4513', fs=9, bold=True)
arr(ax, 4.5, 47.02, 4.5, 46.42)
rbox(ax, 4.5, 46.05, 6.5, 0.65,
     'Predicts cls 0, 1, 2, 3 directly\n(one model, four classes at once)', fc='#975A16', fs=9)
# long arrow to eval
arr(ax, 4.5, 45.72, 4.5, 30.25, c='#8B4513', lw=1.8)

# ── Option B: Domain-Split MoE ───────────────────────────────────────────────
rbox(ax, 12.0, 47.35, 7.0, 0.65, 'Option B — DOMAIN-SPLIT MOE  (Main Method)', fc='#276749', fs=9, bold=True)
arr(ax, 12.0, 47.02, 12.0, 46.52)
arr(ax, 12.0, 46.52, 9.5, 46.12, c='#aaa', lw=1.4)
arr(ax, 12.0, 46.52, 14.5, 46.12, c='#aaa', lw=1.4)

rbox(ax, 9.5, 45.65, 4.8, 0.75,
     'Mass Expert\ncls 0 → benign_mass\ncls 1 → malign_mass', fc='#2C5282', fs=9)
rbox(ax, 14.5, 45.65, 4.8, 0.75,
     'Calc Expert  (+2 offset)\ncls 0→2 benign_calc\ncls 1→3 malign_calc', fc='#22543D', fs=9)
ax.text(14.5, 46.17, '⬆ class remapping: +2', ha='center', fontsize=8, color='#F6E05E', fontweight='bold')

arr(ax, 9.5, 45.27, 10.8, 44.82, c='#aaa', lw=1.4)
arr(ax, 14.5, 45.27, 13.2, 44.82, c='#aaa', lw=1.4)
rbox(ax, 12.0, 44.52, 6.5, 0.55, 'MERGE — union of both experts', fc='#276749', fs=9.5, bold=True)
# long arrow to eval
arr(ax, 12.0, 44.24, 12.0, 30.25, c='#276749', lw=1.8)

# ── Option C: Gated MoE ───────────────────────────────────────────────────────
rbox(ax, 19.5, 47.35, 6.5, 0.65, 'Option C — GATED MOE  (Ablation)', fc='#553C9A', fs=9, bold=True)
arr(ax, 19.5, 47.02, 19.5, 46.42)
rbox(ax, 19.5, 46.0, 5.5, 0.75, 'Router Classifier\ntop1_acc = 0.712', fc='#6B46C1', fs=9)
arr(ax, 19.5, 45.62, 19.5, 45.12)
arr(ax, 19.5, 45.12, 17.5, 44.72, c='#aaa', lw=1.4)
arr(ax, 19.5, 45.12, 21.5, 44.72, c='#aaa', lw=1.4)
rbox(ax, 17.5, 44.3, 3.5, 0.65, 'Mass Expert\n× mass_conf', fc='#2C5282', fs=9)
rbox(ax, 21.5, 44.3, 3.5, 0.65, 'Calc Expert\n× calc_conf', fc='#22543D', fs=9)
arr(ax, 17.5, 43.97, 18.5, 43.52, c='#aaa', lw=1.4)
arr(ax, 21.5, 43.97, 20.5, 43.52, c='#aaa', lw=1.4)
rbox(ax, 19.5, 43.22, 6.0, 0.55, 'Confidence-Scaled Predictions', fc='#553C9A', fs=9)
ax.text(19.5, 42.7,
        'RESULT: WORSE than Option B\nRouter too inaccurate at 71%',
        ha='center', fontsize=8.5, color='#FC8181', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#1A202C', edgecolor='#FC8181', lw=1))
# long arrow to eval
arr(ax, 19.5, 42.94, 19.5, 30.25, c='#553C9A', lw=1.8)

# Option labels
for bx, lbl in zip(B_X, ['A','B','C']):
    ax.text(bx, 47.85, lbl, ha='center', va='center', fontsize=9.5,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='circle,pad=0.28', facecolor='#2D3748', edgecolor='white', lw=1.2))

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — EVALUATION    shade: 20.8 – 30.0
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 30.0, 20.8, '#9B2C2C')
slbl(ax, 29.72, '  STAGE 6 · EVALUATION METRICS', '#9B2C2C')

hline(ax, 4.5, 19.5, 30.25)
arr(ax, 12, 30.25, 12, 29.72)

rbox(ax, 6.0, 25.5, 7.5, 7.5,
     'mAP Evaluation\n─────────────────────\nLibrary : torchmetrics\nIoU threshold : 0.50\n─────────────────────\nOutputs:\n  mAP50\n  mAP50-95\n  Per-class mAP50\n  (all 4 classes)',
     fc='#9B2C2C', fs=9.5)
rbox(ax, 18.0, 25.5, 7.5, 7.5,
     'FROC Evaluation\n─────────────────────\nLibrary : custom code\nHIT_IOU : 0.30\n─────────────────────\nOutputs:\n  Full FROC curve\n  S@2 = Sensitivity\n  at 2 FP/image',
     fc='#C53030', fs=9.5)

ax.text(12, 25.5, '↔\nSame\npredictions\nfed to both',
        ha='center', va='center', fontsize=9, color='#718096')

arr(ax, 6.0, 21.75, 6.0, 21.0, c='#9B2C2C', lw=1.8)
arr(ax, 18.0, 21.75, 18.0, 21.0, c='#C53030', lw=1.8)
hline(ax, 6.0, 18.0, 21.0)
arr(ax, 12, 21.0, 12, 20.52)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — CBIS-DDSM RESULTS    shade: 9.5 – 20.8
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 20.8, 9.5, '#1A365D')
slbl(ax, 20.52, '  STAGE 7 · CBIS-DDSM RESULTS  (mean ± std, seeds 42 / 1 / 123)', '#1A365D')

# ── Overall table ──
C1, C2, C3 = 5.0, 12.0, 19.0
for x, h in zip([C1,C2,C3], ['Model','mAP50  (mean ± std)','FROC S@2  (mean ± std)']):
    ax.text(x, 20.1, h, ha='center', fontsize=9.5, fontweight='bold', color='#1A202C')

ROWS = [
    ('Universal',          '0.0570 ± 0.0014','0.2021 ± 0.0071','#744210'),
    ('Domain-Split MoE ✓', '0.0649 ± 0.0114','0.2027 ± 0.0114','#276749'),
    ('Gated MoE',          '0.0589 ± 0.0075','0.1950 ± 0.0150','#553C9A'),
]
for (m, v1, v2, col), ry in zip(ROWS, [19.5, 18.7, 17.9]):
    best = '✓' in m
    rbox(ax, C1, ry, 6.5, 0.65, m,  fc=col, fs=9.5, bold=best)
    fc2 = '#276749' if best else '#EDF2F7'; tc2 = 'white' if best else '#2D3748'
    rbox(ax, C2, ry, 6.5, 0.65, v1, fc=fc2, tc=tc2, fs=9.5, bold=best)
    rbox(ax, C3, ry, 6.5, 0.65, v2, fc=fc2, tc=tc2, fs=9.5, bold=best)

ax.text(12, 17.22,
        '+13.9% mAP50  ·  FROC S@2 tied  ·  Mass mAP: +27% benign / +15% malignant  ·  Calc expert is BOTTLENECK',
        ha='center', fontsize=9, color='#276749', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FFF4', edgecolor='#276749', lw=1.5))

# ── Per-class mAP50 table ──
ax.text(12, 16.6, 'Per-Class mAP50  (mean, 3 seeds):', ha='center', fontsize=9.5,
        fontweight='bold', color='#2D3748')
PC_X = [3.5, 7.8, 11.5, 15.2, 20.5]
for x, h in zip(PC_X, ['Model','benign_mass','malign_mass','benign_calc','malign_calc']):
    ax.text(x, 16.15, h, ha='center', fontsize=9, fontweight='bold', color='#1A202C')

PC_ROWS = [
    ('Universal',  '0.0664','0.1236','0.0144','0.0237','#744210'),
    ('DS-MoE',     '0.0843\n+27%','0.1421\n+15%','0.0099\n-31%','0.0234\n-1%','#276749'),
    ('Gated MoE',  '0.0802','0.1133','0.0082','0.0338\n+43%','#553C9A'),
]
for (m, bm, mm, bc, mc, col), ry in zip(PC_ROWS, [15.55, 14.75, 13.95]):
    rbox(ax, PC_X[0], ry, 4.5, 0.65, m, fc=col, fs=9)
    for x, val in zip(PC_X[1:], [bm, mm, bc, mc]):
        has_p = '+' in val and col == '#276749'
        has_m = '-' in val and col == '#276749'
        fc_v = '#276749' if has_p else ('#9B2C2C' if has_m else '#EDF2F7')
        tc_v = 'white' if (has_p or has_m) else '#2D3748'
        rbox(ax, x, ry, 4.5, 0.65, val, fc=fc_v, tc=tc_v, fs=9)

ax.text(12, 13.22,
        'Gated MoE wins malignant_calc by +42.6%  (most dangerous missed finding — safety argument for paper)',
        ha='center', fontsize=9, color='#553C9A', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#FAF5FF', edgecolor='#553C9A', lw=1.2))

arr(ax, 12, 12.88, 12, 12.22)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 8 — INBREAST EXTERNAL VALIDATION    shade: 3.5 – 12.2
# ─────────────────────────────────────────────────────────────────────────────
shade(ax, 12.2, 3.5, '#234E52')
slbl(ax, 11.92, '  STAGE 8 · EXTERNAL VALIDATION — INbreast  (Zero-Shot · No Retraining)', '#234E52')

# Flow row
rbox(ax, 3.8, 10.8, 5.8, 1.75,
     'INbreast Dataset\n─────────────────\n412 DICOMs → PNG\n343 plist XML files\nPortugal  ·  Digital\nMammograms\n(never seen in training)',
     fc='#234E52', fs=9, bold=True)
arr(ax, 6.7, 10.8, 8.3, 10.8, c='#777', lw=2)
rbox(ax, 10.2, 10.8, 3.8, 1.75,
     'Custom XML Parser\n─────────────────\nFix: strip "(x,y)"\nparentheses\nMass GT = 116\nCalc  = 7,170 pts',
     fc='#285E61', fs=9)
arr(ax, 12.1, 10.8, 13.6, 10.8, c='#E53E3E', lw=2)
rbox(ax, 17.0, 11.15, 5.5, 0.75, 'Calc  EXCLUDED', fc='#9B2C2C', fs=9.5, bold=True)
rbox(ax, 17.0, 10.35, 5.5, 0.9,
     'Annotation mismatch:\nPoint specks ≠ cluster boxes\nIoU cannot reach 0.30',
     fc='#742A2A', fs=8.5)

arr(ax, 10.2, 9.92, 10.2, 9.32)
rbox(ax, 7.5, 8.95, 5.0, 0.65, 'Mass Expert (no retrain)', fc='#2C5282', fs=9)
rbox(ax, 13.5, 8.95, 5.0, 0.65, 'Universal Model (no retrain)', fc='#8B4513', fs=9)
arr(ax, 10.2, 9.32, 7.5, 9.28, c='#777', lw=1.5)
arr(ax, 10.2, 9.32, 13.5, 9.28, c='#777', lw=1.5)
arr(ax, 7.5, 8.62, 9.2, 8.2, c='#777', lw=1.5)
arr(ax, 13.5, 8.62, 11.8, 8.2, c='#777', lw=1.5)
rbox(ax, 10.5, 7.92, 7.0, 0.65,
     'FROC S@2  —  Mass Only  (HIT_IOU = 0.30)', fc='#C53030', fs=9.5, bold=True)
ax.text(10.5, 7.52,
        'Why mass only: INbreast annotates individual calc specks, not clusters — IoU matching not valid',
        ha='center', fontsize=8, color='#C53030', style='italic')

# ── INbreast results table ──
IB_X = [3.5, 8.5, 14.0, 20.0]
for x, h in zip(IB_X, ['Seed','Universal S@2','DS-MoE S@2','Improvement']):
    ax.text(x, 7.05, h, ha='center', fontsize=9.5, fontweight='bold', color='#1A202C')

IB_ROWS = [
    ('Seed  42', '0.7759', '0.7759', '+0.0000  (FROC tie)'),
    ('Seed   1', '0.7328', '0.8448', '+0.1121'),
    ('Seed 123', '0.7241', '0.8707', '+0.1466'),
    ('MEAN',     '0.7443 ± 0.0226', '0.8305 ± 0.0400', '+11.6%  (+0.0862)'),
]
for (seed, uni, moe, delta), ry in zip(IB_ROWS, [6.52, 6.0, 5.48, 4.82]):
    is_mean = seed == 'MEAN'
    is_tie  = 'tie' in delta
    fb = '#234E52' if is_mean else '#EDF2F7'
    tb = 'white'   if is_mean else '#2D3748'
    fw = '#276749' if is_mean else '#C6F6D5'
    tw = 'white'   if is_mean else '#22543D'
    ft = '#744210'
    rbox(ax, IB_X[0], ry, 4.5, 0.42, seed,  fc=fb, tc=tb, fs=9.5, bold=is_mean)
    rbox(ax, IB_X[1], ry, 4.5, 0.42, uni,   fc=fb, tc=tb, fs=9.5, bold=is_mean)
    rbox(ax, IB_X[2], ry, 4.5, 0.42, moe,   fc=fw, tc=tw, fs=9.5, bold=is_mean)
    rbox(ax, IB_X[3], ry, 4.5, 0.42, delta, fc=ft if is_tie else fw, tc='white', fs=9, bold=is_mean)

ax.text(12, 4.35,
        'INbreast confirms MoE mass advantage on an independent digital mammography dataset  (+11.6% mean)  ·  2 of 3 seeds show clear gain',
        ha='center', fontsize=9, color='white', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#276749', edgecolor='none'))

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND
# ─────────────────────────────────────────────────────────────────────────────
legend_patches = [
    patches.Patch(facecolor='#2C5282', ec='white', label='Mass Expert'),
    patches.Patch(facecolor='#22543D', ec='white', label='Calc Expert'),
    patches.Patch(facecolor='#8B4513', ec='white', label='Universal Model  (Baseline)'),
    patches.Patch(facecolor='#553C9A', ec='white', label='Router / Gated MoE  (Ablation)'),
    patches.Patch(facecolor='#276749', ec='white', label='Domain-Split MoE  ← Main Method'),
    patches.Patch(facecolor='#9B2C2C', ec='white', label='Evaluation Metrics'),
    patches.Patch(facecolor='#234E52', ec='white', label='INbreast External Validation'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9.5,
          framealpha=0.95, edgecolor='#CBD5E0',
          title='Colour Guide', title_fontsize=10)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig('pipeline_diagram.png', dpi=150, bbox_inches='tight', facecolor='#F4F6F8')
print('Saved: pipeline_diagram.png')
plt.show()
