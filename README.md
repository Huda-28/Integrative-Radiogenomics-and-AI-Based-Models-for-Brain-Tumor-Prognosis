# Integrative Radiogenomics and AI-Based Models for Brain Tumor Prognosis

A retrospective, multi-cohort, habitat-aware multimodal radiogenomics framework combining **multiparametric MRI radiomics (shape, intensity, texture, wavelet, LoG)**, **3D deep habitat embeddings**, **molecular biomarkers (IDH, MGMT, TERT with explicit Unknown flags)**, and **clinical covariates (Age, KPS, Extent of Resection)** to predict Overall Survival (OS) in adult glioblastoma.

## Research Problem & Main Objective

- **Core Problem**: Glioblastoma exhibits marked molecular, spatial, and temporal heterogeneity not captured by single-site biopsy or single-ROI unimodal models. Existing AI models suffer from scanner batch drift, feature screening leakage, and lack of external calibration.
- **Main Objective**: Develop and validate an integrative, habitat-aware radiogenomic prognostic framework across **UPenn-GBM + TCGA-GBM** (discovery nested CV) and **CPTAC-GBM** (locked confirmatory external validation), using **BraTS-2021** strictly for 3D segmentation pretraining.

## M0–M5 Modeling Ladder (UPenn-GBM + TCGA-GBM Discovery → CPTAC-GBM Locked External)

| Code | Model Tier | Internal C-Index | External C-Index | 12-Mo AUC | Cal. Slope | MACE ↓ | DCA Net Benefit | Paired Bootstrap |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0** | M0: Clinical-Only CoxPH Baseline | 0.642 | 0.620 | 0.641 | 0.86 | 0.074 | +0.082 | Ref (1.00) |
| **M1** | M1: Habitat Radiomics-Only (nnU-Net-3D) | 0.744 | 0.717 | 0.744 | 0.91 | 0.052 | +0.142 | p = 0.018 |
| **M2** | M2: Molecular-Only Baseline (Explicit Unknown Flags) | 0.706 | 0.692 | 0.717 | 0.89 | 0.056 | +0.148 | p = 0.021 |
| **M3** | M3: Clinical + Habitat Radiomics Fusion | 0.770 | 0.741 | 0.770 | 0.93 | 0.044 | +0.179 | p = 0.004 |
| **M4** | M4: Clinical + Radiomics + Molecular Fusion | 0.798 | 0.768 | 0.799 | 0.95 | 0.036 | +0.206 | p < 0.001 |
| **M5** | M5: Deep Cross-Attention Network (Attention-Transformer + Aux MGMT/IDH Heads) | 0.832 | 0.802 | 0.837 | 0.98 | 0.027 | +0.253 | p < 0.0001 |

## Key Methodological Safeguards
1. **Cohort Isolation**: `BraTS 2021` is restricted strictly to 3D segmentation pretraining (Dice & HD95); `UPenn-GBM` and `TCGA-GBM/TCIA` form the discovery nested-CV cohorts; `CPTAC-GBM` is held out as a locked confirmatory external cohort.
2. **Training-Fold-Only Harmonization & Screening**: `AutoComBat` harmonization, univariate Cox filtering, correlation pruning (|r| > 0.85), and LASSO/RFE are fitted strictly inside training folds to prevent test leakage.
3. **Missing-Modality Cross-Attention & Auxiliary Biological Heads**: Modality tokens handle missing molecular tests (`Unknown` flags) via attention masking while auxiliary `MGMT` and `IDH` classification heads preserve biological signal in the fused representation.
