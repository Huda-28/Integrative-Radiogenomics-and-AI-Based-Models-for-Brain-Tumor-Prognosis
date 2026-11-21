# Research Problem, Hypotheses, and Specific Objectives

## 1. Research Problem
Develop a clinically credible, multimodal, and explainable radiogenomic AI framework that predicts adult glioblastoma overall survival (OS) more accurately than unimodal approaches while remaining generalizable and calibrated across independent multi-institutional cohorts [16][104][134][136][159].

## 2. Study Hypotheses
- **Primary Hypothesis**: H1 (Primary): A habitat-aware multimodal fused model (M5) will significantly outperform clinical-only (M0), radiomics-only (M1), and genomics-only (M2) baselines for Overall Survival prediction on paired bootstrap resamples [97][146][147].
- **H2a**: Habitat-aware subregional modeling (enhancing rim, necrotic core, peritumoral edema) improves prognostic discrimination over whole-tumor-only ROI modeling. [137] [148] [149]
- **H2b**: Explainability methods (SHAP, cross-modal attention weights, Grad-CAM) recover biologically plausible predictors aligned with IDH, MGMT, and tumor habitat biology. [13] [14] [150]
- **H2c**: Locked external validation (CPTAC-GBM) will show expected attenuation relative to internal nested CV, and training-fold-only ComBat harmonization trades off slight discrimination changes for superior calibration stability. [16] [97] [98] [134] [140]

## 3. Main Objective
To develop and validate an integrative radiogenomics and AI-based prognostic model for brain tumors that combines habitat-aware MRI radiomics, deep 3D representations, genomic biomarkers, and clinical variables to improve overall survival prediction and risk stratification [102][134][136][250].

## 4. Seven Specific Objectives
1. **Multi-Cohort Assembly & Fold-Safe Harmonization** (Phases 1 & 2): Assemble and harmonize multimodal MRI, genomic, and clinical records using UPenn-GBM and TCGA-GBM/TCIA for discovery, BraTS 2021 strictly for segmentation pretraining, and locked CPTAC-GBM for confirmatory external testing. [74] [104] [134] [137] [151]
2. **Subregional & Habitat-Aware Feature Extraction** (Phases 3 & 4): Segment whole tumor, enhancing tumor, necrotic core, edema, and peritumoral habitats (reporting Dice & HD95) and extract shape, intensity, texture, wavelet, LoG, and deep encoder embeddings. [104] [149] [152] [247] [248]
3. **Non-Invasive Molecular Biomarker Integration (With Unknown Flags)** (Phases 4, 6 & 7): Integrate IDH, MGMT promoter methylation, and TERT status coded with explicit unknown flags alongside auxiliary MGMT/IDH multi-task heads. [142] [146] [159] [249] [251]
4. **Systematic M0–M5 Modeling Ladder Comparison** (Phases 5 & 6): Benchmark M0 (clinical-only Cox), M1 (radiomics-only), M2 (molecular-only), M3 (clinical + radiomics), M4 (clinical + radiomics + molecular), and M5 (deep cross-attention token fusion). [97] [102] [146] [158] [162]
5. **Rigorous Nested CV, External Validation, Calibration & DCA** (Phase 9): Evaluate Harrell C-index, time-dependent AUC, calibration slope, mean absolute calibration error (MACE), Kaplan-Meier risk stratification, Decision-Curve Analysis, and paired bootstrap resamples. [14] [138] [245] [254] [256]
6. **Explainable AI (SHAP, Attention Weights & 3D Grad-CAM)** (Phase 8): Deploy XAI as a clinical validation and trust layer tracing predictions to specific tumor habitats, age, KPS, IDH, and MGMT status. [13] [14] [50] [141] [257] [258]
7. **Biological Phenotype Discovery & Clinical Nomogram Translation** (Phases 7 & 10): Identify prognostic imaging-molecular correlates (HIF-1α hypoxia, MGMT repair, CD8+ infiltration, S100B/ITGB5) and translate outputs into an interactive clinical decision-support nomogram. [122] [149] [180] [259]

## 5. Three Core Manuscript Contributions
1. **Multimodal Prognostic Gain (M0 → M5 Ladder)**: Demonstrates statistically significant C-index and net clinical benefit improvement across a 6-step modeling ladder with missing-modality cross-attention and auxiliary MGMT/IDH heads. [97] [146] [158]
2. **Spatially Resolved Radiogenomic Interpretation**: Links subregional MRI habitat radiomics (shape, texture, wavelet, LoG) and deep 3D tokens directly to molecular and immune pathways. [139] [148] [149] [180]
3. **Clinical Readiness via Leakage-Free Validation & Calibration**: Enforces training-fold-only harmonization and feature screening, paired bootstrap testing, calibration slope/MACE reporting, and DCA on locked external cohorts. [16] [32] [98] [138]
