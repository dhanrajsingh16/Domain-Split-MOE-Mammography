# A Domain-Split Mixture-of-Experts Framework for Four-Class Mammography Lesion Detection on CBIS-DDSM

**Accepted at CVIP 2026** — Computer Vision and Image Processing Conference  
Published in *Springer Lecture Notes in Computer Science*

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

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| Universal (baseline) | — | — |
| Mass Expert (YOLOv8l) | — | — |
| Calc Expert (YOLOv8l) | — | — |
| **MoE Ensemble (ours)** | **best** | **best** |

*(See `ALL_RESULTS.txt` for full per-class metrics and 3-seed bootstrap confidence intervals)*

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
  author    = {Singh, Dhanraj and others},
  booktitle = {Computer Vision and Image Processing (CVIP 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

## License

This code is released for academic research purposes. The CBIS-DDSM dataset has its own license — see [TCIA](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).
