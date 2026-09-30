# Existing Project Analysis: ZamirDev/active-asteroids-ml vs. Our Research

## 1. Overview of the Reference Baseline

The public repository [`ZamirDev/active-asteroids-ml`](https://github.com/ZamirDev/active-asteroids-ml) was established as an open, reproducible machine learning baseline for detecting comet-like activity in cutout images from the NASA Zooniverse Active Asteroids project.

### Technical Profile of Existing Work
- **Author/Repo**: ZamirDev (`ZamirDev/active-asteroids-ml`)
- **Published**: 2024
- **Scope**: A rapid 2-day prototyping sprint comparing an 8-feature hand-crafted logistic regression against an ImageNet-pretrained ResNet18 fine-tune.
- **Hardware**: Run on free Kaggle T4/P100 GPUs.
- **Dataset Composition (Documented: 2,054 cutouts)**:
  - **1,315 training positives**: From Zooniverse Subject Set `96361` ("CSB00 Training Set 1", the teaching cutouts containing volunteer-labeled textbook active comets and centaurs).
  - **66 eval positives**: From general-stream subject sets (`36656`, `96360`, `97743`, `99007`, `100220`, `107536`) matched by packed MPC designation or `#Name` to a 25-entry known-active catalog (derived from arXiv:2403.09768). Split ~50/50 between validation and test.
  - **673 negatives**: General-stream cutouts without catalog matches, split 60% train / 20% val / 20% test.
- **Models & Preprocessing**:
  - *Baseline*: Logistic regression (`C=1.0`, `class_weight='balanced'`) on 8 hand-crafted spatial and flux features computed on 96×96 greyscale cutouts (central concentration, global contrast, hot-tail spread, centre structure, horizontal gradient roughness, Laplacian edge energy, peak-to-background ratio, and elongation ratio).
  - *CNN*: ResNet18 fine-tuning (ImageNet pretrained), 224×224 RGB input, Adam optimizer (`lr=3e-4`), `CosineAnnealingLR` scheduler, `WeightedRandomSampler` for 50/50 batch balancing, and `BCEWithLogitsLoss`.
- **Reported Baseline Results**:
  - Logistic Regression: Val ROC-AUC = 0.526 (PR-AUC = 0.258); Test ROC-AUC = 0.532 (PR-AUC = 0.219).
  - ResNet18: Val ROC-AUC = 0.603 (PR-AUC = 0.358); Test ROC-AUC = **0.650** (PR-AUC = **0.279**).
  - Operating point at threshold 0.5: Test Recall = 0.152, Precision = 0.294, F1 = 0.200.

---

## 2. Dataset Reproducibility Assessment

### Can the Documented 2,054 Image Dataset Be Exactly Reproduced?
- **Public Accessibility of Source**: The Zooniverse Panoptes API provides public endpoints (`/api/subjects?subject_set_id=...`) for the Active Asteroids project (ID `6801`). However, large-scale multi-page downloading directly from Panoptes requires persistent network connections and is subject to rate limiting and remote connection resets.
- **Catalog Reproducibility**: The 25-entry starter catalog of known active asteroids is fully reproducible from published scientific literature (primarily Chandler et al. 2024, arXiv:2403.09768).
- **Subject Matching Logic**: The exact heuristic for extracting packed MPC designations from filenames (e.g., `0062412A` for asteroid 62412, `0002060U` for Centaur 2060 Chiron) is reproducible and verified against the actual Panoptes metadata.
- **Our Dataset Scale**: Because of remote server connection limits during batch export, our primary experimental pipeline operates on a verified subset of 200 subjects directly acquired from Panoptes (Subject Set `96361` and Set `36656`). This subset contains:
  - 100 high-confidence teaching cutouts (Set 96361)
  - 16 catalog-matched positive cutouts (Set 36656: 1 Chiron, 15 of Main-belt comet 62412)
  - 84 unmatched negative cutouts (Set 36656: inactive asteroids such as 65 Cybele, 704 Interamnia, 3552 Don Quixote)
- **Scientific Equivalence**: This split mirrors the exact distribution design of ZamirDev (teaching set for training, catalog matches for eval, and unmatched cutouts for negative pool) while operating within an honest, auditable local compute budget.

---

## 3. Key Differences: Existing Work vs. Our Research

| Component | Existing Work (`ZamirDev/active-asteroids-ml`) | Our Research (This Project) |
|---|---|---|
| **Core Research Question** | "Can a standard CNN beat a simple hand-crafted logistic baseline on citizen science cutouts?" | **"How does image-level label uncertainty affect computer vision classification and error patterns in active asteroid images?"** |
| **Scientific Focus** | Feasibility benchmark & model comparison | **Label uncertainty quantification, stratified error analysis, and evaluation of object-level vs. image-level ground truth** |
| **Treatment of Ground Truth** | Assumes catalog-matched asteroids are reliably positive training/evaluation instances. | **Recognizes that cataloged active asteroids do not guarantee visible activity in every observation**, treating catalog matches as uncertain labels. |
| **Confidence Modeling** | Unmodeled; standard sigmoid outputs evaluated at arbitrary thresholds (0.5, 0.7, 0.9). | **Explicit stratification into confidence groups**: High-Confidence Teaching Positives, Medium-Confidence Catalog Matches, and Low-Confidence Unmatched Negatives. |
| **Error Diagnosis** | High-confidence negatives noted as candidate false positives without systematic taxonomy. | **Formal taxonomy of error types** (TP, TN, FP, FN, HC-FP, HC-FN) linked to specific astronomical and imaging mechanisms (galaxies, CCD streaks, SNR limits, epoch quiescence). |
| **Methodological Rigor** | 2-day sprint; single random seed; limited formal documentation. | **End-to-end reproducible research pipeline**: multi-seed robustness analysis, verified data quality auditing, complete research log, and formal scientific reporting. |

---

## 4. Specific Limitations Identified in Existing Work

1. **Unacknowledged Label Ambiguity**: The reference project notes that *"trained on textbook-activity teaching cutouts, evaluated on the general stream... validation PR-AUC never improved past epoch 1."* It attributed this solely to "distribution shift." However, our analysis reveals a deeper cause: **label uncertainty**. When an active asteroid like 62412 is observed during a quiescent orbital epoch, it physically lacks a coma or tail. Calling that image a "positive ground truth" injects label noise that directly penalizes a model capable of recognizing true comet tails.
2. **Evaluation Metric Fragility**: Under extreme rare-class imbalance, metrics like accuracy are meaningless, and PR-AUC is highly sensitive to the exact positive sample count. The reference project used an arbitrary 0.5 threshold which gave recall of only 15%.
3. **No Stratified Uncertainty Analysis**: The baseline did not analyze whether model confidence correlates with the source or reliability of the label.
4. **Lack of Automated Data Quality Verification**: No automated audit of duplicates, metadata integrity, or split leakage was provided.

Our research directly addresses each of these limitations.
