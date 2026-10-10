"""
CVIP 2026 — DS-MoE Inference Pipeline (professional, clean arrows)
Run: python draw_pipeline_cvip.py
Saves: pipeline_diagram.png
"""

import matplotlib
matplotlib.use('Agg')          # non-blocking: save only, no window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 10.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 10.5)
ax.axis('off')
BG = '#F7F9FC'
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Colours ───────────────────────────────────────────────────────────────────
C_INPUT  = '#2B6CB0'
C_MASS   = '#2C5282'
C_CALC   = '#22543D'
C_OFF    = '#B7791F'
C_MERGE  = '#276749'
C_NMS    = '#1A365D'
C_OUT    = '#744210'
C_ROUTER = '#553C9A'
C_SCORE  = '#6B46C1'
C_RESULT = '#9B2C2C'
C_DASH   = '#805AD5'
C_WIRE   = '#4A5568'

PAD = 0.06

# ── Primitives ────────────────────────────────────────────────────────────────
def box(cx, cy, W, H, txt, fc, tc='white', fs=9.5, bold=False):
    w, h = W - 2*PAD, H - 2*PAD
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={PAD}',
        facecolor=fc, edgecolor='white', linewidth=2.0, zorder=3))
    ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs, color=tc,
            fontweight='bold' if bold else 'normal',
            zorder=4, multialignment='center', linespacing=1.5)

def arrow(x1, y1, x2, y2, c=C_WIRE, lw=2.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                mutation_scale=18,
                                joinstyle='miter', capstyle='butt'), zorder=5)

def wire(x1, y1, x2, y2, c=C_WIRE, lw=2.0):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, zorder=4,
            solid_capstyle='round')

def darrow(x1, y1, x2, y2, c=C_DASH, lw=1.8):
    """Dashed shaft (~94%) + crisp SOLID arrowhead on the final segment."""
    hx, hy = x1 + 0.94*(x2 - x1), y1 + 0.94*(y2 - y1)   # 94% along the line
    ax.plot([x1, hx], [y1, hy], color=c, lw=lw,
            linestyle=(0, (5, 3)), zorder=4, solid_capstyle='round')
    ax.annotate('', xy=(x2, y2), xytext=(hx, hy),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                                mutation_scale=16), zorder=5)

def dwire(x1, y1, x2, y2, c=C_DASH, lw=1.8):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw,
            linestyle=(0, (5, 3)), zorder=4)

# =============================================================================
#  LAYOUT
# =============================================================================
XC, XM, XR = 6.0, 2.8, 9.2
XG  = 12.9
SEP = 11.35

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(XC, 10.05, 'DS-MoE Inference Pipeline',
        ha='center', fontsize=15, fontweight='bold', color='#1A202C')
ax.text(XC, 9.66,
        'Both expert networks process every mammogram independently   ·   No routing decision required',
        ha='center', fontsize=9, color='#4A5568', style='italic')

# ── (0) Input  cy=8.60 (bot 8.19) ────────────────────────────────────────────
box(XC, 8.60, 3.4, 0.82,
    'Input Mammogram\nresized to 1280 px',
    C_INPUT, fs=10.5, bold=True)

# distribution: input -> rail -> two experts
wire(XC, 8.19, XC, 7.85)
wire(XM, 7.85, XR, 7.85)
arrow(XM, 7.85, XM, 7.57, c=C_MASS)
arrow(XR, 7.85, XR, 7.57, c=C_CALC)

# ── (1) Experts  cy=6.85  H=1.4 (top 7.55, bot 6.15) ─────────────────────────
box(XM, 6.85, 3.4, 1.40,
    'Mass Expert\nYOLO11l   ·   nc = 2\n————————————\ncls 0 → benign_mass\ncls 1 → malign_mass',
    C_MASS, fs=9.5)
box(XR, 6.85, 3.4, 1.40,
    'Calc Expert\nYOLO11l   ·   nc = 2\n————————————\ncls 0 → benign_calc\ncls 1 → malign_calc',
    C_CALC, fs=9.5)

# calc -> +2 offset  (arrow, enters box)
arrow(XR, 6.15, XR, 5.77, c=C_CALC)

# ── (2) +2 offset  cy=5.50  H=0.5 (top 5.75, bot 5.25) ──────────────────────
box(XR, 5.50, 3.4, 0.50, '+2 class offset   (0 → BC,  1 → MC)',
    '#FEFCBF', tc='#744210', fs=9.5, bold=True)
ax.add_patch(FancyBboxPatch((XR-(3.4-2*PAD)/2, 5.50-(0.5-2*PAD)/2),
    3.4-2*PAD, 0.5-2*PAD, boxstyle=f'round,pad={PAD}',
    facecolor='none', edgecolor=C_OFF, linewidth=2.0, zorder=4))

# ── merge junction: two colour-coded feeder arrows into the union box ────────
RAIL = 5.05
LX, RX = 4.35, 7.65                    # entry points, symmetric about XC
# mass stream (blue): down, across, into union
wire (XM, 6.15, XM, RAIL, c=C_MASS)
wire (XM, RAIL, LX, RAIL, c=C_MASS)
arrow(LX, RAIL, LX, 4.66, c=C_MASS)
# calc stream (green): offset-bottom down, across, into union
wire (XR, 5.25, XR, RAIL, c=C_CALC)
wire (XR, RAIL, RX, RAIL, c=C_CALC)
arrow(RX, RAIL, RX, 4.66, c=C_CALC)

# ── (3) Class-Offset Union  cy=4.30  H=0.66 (top 4.63, bot 3.97) ────────────
box(XC, 4.30, 6.4, 0.66,
    'Class-Offset Union\n{ mass preds: cls 0, 1 }   ∪   { calc preds: cls 2, 3  (after +2) }',
    C_MERGE, fs=9.5, bold=True)

arrow(XC, 3.97, XC, 3.68, c=C_NMS)

# ── (4) NMS  cy=3.35  H=0.6 (top 3.65, bot 3.05) ─────────────────────────────
box(XC, 3.35, 6.0, 0.60,
    'Non-Maximum Suppression    ·    conf ≥ 0.25    ·    IoU 0.45',
    C_NMS, fs=9, bold=True)

arrow(XC, 3.05, XC, 2.73, c=C_OUT)

# ── (5) Four-Class Output  cy=2.25  H=0.9 (top 2.70, bot 1.80) ──────────────
box(XC, 2.25, 5.6, 0.90,
    'Four-Class Output\ncls 0: benign_mass      cls 1: malign_mass\ncls 2: benign_calc       cls 3: malign_calc',
    C_OUT, fs=9.5, bold=True)

# =============================================================================
#  GATED MOE — right panel
# =============================================================================
wire(SEP, 1.85, SEP, 8.98, c='#CBD5E0', lw=1.3)
ax.text(XG, 9.42, 'Gated MoE Variant', ha='center',
        fontsize=11, fontweight='bold', color=C_ROUTER)
ax.text(XG, 9.12, 'dashed path — ablation', ha='center',
        fontsize=8.5, color=C_DASH, style='italic')

# Router  cy=8.55  H=0.75 (bot 8.175)
box(XG, 8.55, 2.9, 0.75, 'Router\nYOLO11-cls   ·   top-1 = 71.2%', C_ROUTER, fs=9)
darrow(XC + 1.6 + 0.06, 8.55, XG - 1.47, 8.55)      # input -> router

# Score modulation  cy=7.05  H=1.0 (top 7.55, bot 6.55)
box(XG, 7.05, 2.9, 1.00,
    'Score Modulation\n————————————\n'
    r'$\tilde{s}_m = p_m \cdot s_m$' + '\n'
    r'$\tilde{s}_c = (1 - p_m)\,\cdot\, s_c$',
    C_SCORE, fs=9)
darrow(XG, 8.175, XG, 7.57)                          # router -> score

# Result  cy=5.55  H=1.15 (top 6.125, bot 4.975)
box(XG, 5.55, 2.9, 1.15,
    'Result\n————————————\nWorse than DS-MoE\n(71% router accuracy\nnot sufficient)',
    C_RESULT, fs=9)
darrow(XG, 6.55, XG, 6.14)                           # score -> result

# note: scores merge before NMS (kept out of the flow to avoid clutter)
ax.text(XG, 4.72,
        'Gated scores are merged\nand passed to NMS\n(same as DS-MoE)',
        ha='center', va='center', fontsize=8, color=C_DASH, style='italic')

# =============================================================================
#  STEP RAIL (left)
# =============================================================================
wire(0.70, 1.85, 0.70, 8.85, c='#CBD5E0', lw=1.3)
for y, lbl, c in [
    (8.60, 'INPUT',   C_INPUT),
    (6.85, 'EXPERTS', C_WIRE),
    (5.50, 'OFFSET',  C_OFF),
    (4.30, 'MERGE',   C_MERGE),
    (3.35, 'NMS',     C_NMS),
    (2.25, 'OUTPUT',  C_OUT),
]:
    ax.text(0.36, y, lbl, ha='center', va='center', fontsize=7.5,
            color=c, fontweight='bold', rotation=90, zorder=6)

# =============================================================================
#  LEGEND
# =============================================================================
handles = [
    mpatches.Patch(fc=C_INPUT,   ec='white', label='Input'),
    mpatches.Patch(fc=C_MASS,    ec='white', label='Mass Expert  (nc = 2)'),
    mpatches.Patch(fc=C_CALC,    ec='white', label='Calc Expert  (nc = 2)'),
    mpatches.Patch(fc='#FEFCBF', ec=C_OFF,   label='+2 class offset'),
    mpatches.Patch(fc=C_MERGE,   ec='white', label='Class-Offset Union  (DS-MoE)'),
    mpatches.Patch(fc=C_NMS,     ec='white', label='NMS'),
    mpatches.Patch(fc=C_OUT,     ec='white', label='Four-Class Output'),
    mpatches.Patch(fc=C_ROUTER,  ec='white', label='Gated MoE Router  (ablation)'),
]
leg = ax.legend(handles=handles, loc='lower left',
                fontsize=8.5, framealpha=0.96, edgecolor='#CBD5E0',
                bbox_to_anchor=(0.01, 0.005),
                title='Colour Legend', title_fontsize=9, ncol=2)
leg.get_frame().set_linewidth(1.0)

# =============================================================================
plt.savefig('pipeline_diagram.png', dpi=200, bbox_inches='tight',
            facecolor=BG, pad_inches=0.25)
print('Saved: pipeline_diagram.png')
