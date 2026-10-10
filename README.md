# A Domain-Split Mixture-of-Experts Framework for Four-Class Mammography Lesion Detection on CBIS-DDSM

**Accepted at CVIP 2026** : Computer Vision and Image Processing Conference, Published in *Springer Lecture Notes in Computer Science*
Research Paper Link : https://www.researchgate.net/publication/415539642_A_Domain-Split_Mixture-of-Experts_Framework_for_Four-Class_Mammography_Lesion_Detection_on_CBIS-DDSM?_sg%5B0%5D=-enIg8T6HyFoq-ewOC8c3I0fBrVxbO8dTckadiGuXC6fDhfD33Jz9E7rqVtMa9JAwoxdw-qtd_3KG8x9zFzAgnqRDpHE6dl437W4G_Bb.x_OA-uZGwMOwfG67TaggEGyo1WOGUtSnVi2siqFnh3aQVNRp_kuO--pMop2bmgaTynh-mlVrE4SOUVj6DBeX0g&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InByb2ZpbGUiLCJwYWdlIjoicHJvZmlsZSIsInBvc2l0aW9uIjoicGFnZUNvbnRlbnQiLCJzdWJQYWdlIjoicmVzZWFyY2gifX0

## Overview

This repository contains the full implementation of our Domain-Split Mixture-of-Experts (MoE) framework for four-class mammography lesion detection on the CBIS-DDSM dataset.

**Core idea:** Instead of one universal detector for all 4 lesion classes, we train two domain-specialized experts:
- **Mass Expert** — detects `benign_mass` and `malignant_mass`
- **Calc Expert** — detects `benign_calc` and `malignant_calc`

A gated router combines their predictions at inference time, achieving better per-class mAP than a single universal model.

**4 Target Classes:**

| Class ID | Name | Description |
|---|---|---|
| 0 | benign_mass | Harmless solid lump |
| 1 | malignant_mass | Cancerous solid lump |
| 2 | benign_calc | Harmless calcium deposit cluster |
| 3 | malignant_calc | Cancerous calcium deposit cluster |

## Results

All results on the 458-image CBIS-DDSM test set, mean ± std over 3 seeds (42/1/123), patient-level leakage-free evaluation.

### Main Comparison (Table 3)

| Model | mAP@0.5 | mAP@0.5:0.95 | FROC S@2 |
|---|---|---|---|
| Universal (baseline) | 0.0570 ± 0.0014 | 0.0243 ± 0.0003 | 0.2021 ± 0.0071 |
| **DS-MoE (ours)** | **0.0649 ± 0.0114** | **0.0300 ± 0.0049** | **0.2027 ± 0.0114** |
| Gated MoE | 0.0589 ± 0.0075 | 0.0263 ± 0.0034 | 0.1950 ± 0.0150 |

**DS-MoE vs Universal: mAP@0.5 +13.9% · FROC S@2 +0.3%**

### Per-Class mAP@0.5 (Table 4)

| Model | Benign Mass | Malignant Mass | Benign Calc | Malignant Calc |
|---|---|---|---|---|
| Universal | 0.0664 | 0.1236 | **0.0144** | 0.0237 |
| **DS-MoE (ours)** | **0.0843** | **0.1421** | 0.0099 | 0.0234 |
| Gated MoE | 0.0802 | 0.1133 | 0.0082 | **0.0338** |
| Δ DS-MoE | +27.0% | +15.0% | −31.3% | −1.3% |

### Expert Validation mAP@0.5 (Table 2)

| Expert | mAP@0.5 | Ben. Mass | Mal. Mass | Ben. Calc | Mal. Calc |
|---|---|---|---|---|---|
| Mass Expert (YOLO11l) | 0.499 | 0.264 | 0.734 | 0.000 | 0.000 |
| Calc Expert (YOLO11l) | 0.167 | 0.000 | 0.000 | 0.189 | 0.145 |
| Universal (4-class) | 0.062 | 0.066 | 0.124 | 0.014 | 0.024 |

### Zero-Shot Cross-Dataset Transfer

| Dataset | Universal | DS-MoE | Δ |
|---|---|---|---|
| INbreast (mass FROC S@2, 116 GT boxes) | 0.7443 ± 0.0226 | **0.8305 ± 0.0400** | +11.6% |
| MIAS (mAP@0.5, 119 GT boxes) | 0.0880 ± 0.0330 | **0.0917 ± 0.0063** | +4.2% |
| CMMD (image-level sensitivity, proxy) | 0.0020 ± 0.0014 | **0.1548 ± 0.0287** | +77× |

> **Note on absolute scores:** Values are intentionally conservative — strict patient-level evaluation removes the 30–50% inflation common in image-level-split benchmarks. Read as controlled relative comparisons, not absolute capability.

*(See `ALL_RESULTS.txt` and `moe_eval_3seeds.ipynb` for full per-seed breakdown and bootstrap confidence intervals)*

## Repository Structure

```
.
├── cbis_ddsm_to_yolo.py          # Dataset conversion: CBIS-DDSM → YOLO format
├── yolo_train_evaluate.py        # General training + evaluation utilities
│
├── exp11_yolov8l_mass_expert.py  # Mass Expert — YOLOv8l training
├── exp12_yolov8l_calc_expert.py  # Calc Expert — YOLOv8l training
├── exp13_yolo11l_mass_final.py   # Mass Expert — YOLO11l (final)
├── exp14_yolov8l_calc_final.py   # Calc Expert — final ablation
├── exp15_moe_ensemble.py         # MoE ensemble (naive averaging)
├── exp16_calc_FROM_SCRATCH.py    # Calc expert trained from scratch
├── exp18_universal_FROM_SCRATCH.py # Universal model baseline
├── exp19_moe_eval.py             # MoE evaluation
├── exp20_router.py               # Domain router / gating mechanism
├── exp21_calc_fullimage.py       # Calc expert on full-resolution images
├── exp22_gated_moe_eval.py       # Gated MoE final evaluation
│
├── mass_eval.py                  # Mass expert standalone evaluation
├── cmmd_eval.py                  # Cross-dataset evaluation on CMMD
├── mias_eval.py                  # Cross-dataset evaluation on MIAS
├── bootstrap_ci.py               # Bootstrap confidence intervals (3 seeds)
│
├── moe_eval_3seeds.ipynb         # Notebook: 3-seed evaluation + results table
├── plot_froc.py                  # FROC curve plotting
├── draw_pipeline_cvip.py         # Pipeline diagram generation
├── draw_barchart_cvip.py         # Per-class mAP bar chart generation
│
├── pipeline_diagram.png          # Figure 1: System architecture
├── froc_curves.png               # Figure 2: FROC curves
├── perclass_map.png              # Figure 3: Per-class mAP comparison
│
├── cvip_paper.tex                # LaTeX source of the accepted paper
├── ALL_RESULTS.txt               # All experimental results
└── USER_GUIDE.md                 # Detailed project documentation
```

## Setup

```bash
pip install ultralytics opencv-python pandas numpy matplotlib scikit-learn
```

**Dataset:** [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM) — download and run `cbis_ddsm_to_yolo.py` to convert to YOLO format.

## Usage

**1. Convert dataset:**
```bash
python cbis_ddsm_to_yolo.py
```

**2. Train Mass Expert:**
```bash
python exp13_yolo11l_mass_final.py
```

**3. Train Calc Expert:**
```bash
python exp14_yolov8l_calc_final.py
```

**4. Run MoE Evaluation:**
```bash
python exp22_gated_moe_eval.py
```

**5. Generate FROC curves:**
```bash
python plot_froc.py
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{singh2026domainsplit,
  title     = {A Domain-Split Mixture-of-Experts Framework for Four-Class Mammography Lesion Detection on CBIS-DDSM},
  author    = {Singh, Dhanraj},
  booktitle = {Computer Vision and Image Processing (CVIP 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

## License

This code is released for academic research purposes. The CBIS-DDSM dataset has its own license — see [TCIA](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).
