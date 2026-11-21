# Integrative Radiogenomics and AI-Based Models for Brain Tumor Prognosis

A harmonized, multicenter, MRI-centered multimodal radiogenomic framework combining **3D habitat segmentation**, **IBSI PyRadiomics & 3D Transformer features**, **molecular/genomic biomarkers (IDH, MGMT, TERT, 1p/19q)**, **cross-modal attention fusion**, **DeepSurv time-to-event survival modeling**, and **SHAP/Grad-CAM biological interpretation**.

## 10-Phase End-to-End Pipeline

1. **Data Acquisition**: Multi-institutional MRI (T1, T1ce, T2, FLAIR), genomic/methylation, and clinical covariates from **BraTS, TCGA-GBM** (discovery) and **NCI-CPTAC** (external validation).
2. **Image Harmonization & Preprocessing**: 1 mm³ isotropic registration, HD-BET skull stripping, Z-score intensity normalization, and **AutoComBat** multicenter scanner batch correction.
3. **Habitat-Aware Tumor Segmentation**: **SFF-Former** 3D segmentation of Whole Tumor (WT), Tumor Core (TC), Enhancing Tumor (ET), and Peritumoral Edema habitats.
4. **Multimodal Feature Extraction**: PyRadiomics handcrafted habitat features + 3D Swin/ViT encoder embeddings + IDH1/2, MGMT, TERT, 1p/19q, and transcriptomic pathway scores.
5. **Feature Selection & Fusion**: LASSO-Cox collinearity pruning followed by **Attention-Transformer** cross-modal representation learning.
6. **Prognostic Survival Modeling**: **DeepSurv-Attention** time-to-event survival head predicting Overall Survival (OS), Progression-Free Survival (PFS), and 12-month recurrence risk.
7. **Radiogenomic Biological Interpretation**: Correlating imaging habitat phenotypes with MGMT promoter methylation, HIF-1α angiogenesis, and CD8+ immune infiltration.
8. **Explainable AI (XAI)**: Patient-level SHAP feature attributions and 3D spatial Grad-CAM attention overlays.
9. **Validation Strategy**: 5-fold internal cross-validation + locked multi-center external validation reporting Harrell's C-index, 12-month AUC, Brier calibration, and Decision Curve Analysis (DCA).
10. **Clinical Translation**: Interactive radiogenomic nomogram for neuro-oncology risk stratification.

## Benchmark Comparison (Discovery vs. External Validation: NCI-CPTAC)

| Model Baseline | Modalities | Internal C-Index | External C-Index | 12-Mo AUC | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Clinical-Only CoxPH (Age, KPS, EOR) | Clinical | 0.642 | 0.621 | 0.643 | 0.214 |
| MRI Habitat Radiomics (SFF-Former + AutoComBat) | Imaging | 0.758 | 0.730 | 0.761 | 0.165 |
| Molecular-Only Signature (IDH, MGMT, TERT, 1p/19q) | Genomics | 0.712 | 0.697 | 0.725 | 0.172 |
| **Fused Radiogenomic Model (Attention-Transformer + DeepSurv-Attention)** | **Multimodal** | **0.855** | **0.829** | **0.863** | **0.120** |
