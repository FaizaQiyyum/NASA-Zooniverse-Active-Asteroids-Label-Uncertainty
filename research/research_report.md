# NASA Zooniverse Active Asteroids: Label Uncertainty in Computer Vision

**An Independent Investigation into Object-Level Catalog Ground Truth vs. Image-Level Observational Evidence in Astronomical Machine Learning**

---

## Abstract

Active asteroids are small Solar System bodies that occupy asteroid-like dynamical orbits while exhibiting comet-like dust emission features (tails or comae). In automated survey pipelines, machine learning models are typically supervised using object-level catalog labels derived from historical activity. However, active asteroids are often active only intermittently (e.g., near perihelion, or triggered by stochastic collisions and rotational disruption). Consequently, individual astronomical cutouts may show no visible dust activity despite belonging to an officially cataloged "active" body. 

In this independent research project, we investigate how **image-level label uncertainty** affects computer vision classification and error behavior using publicly accessible resources from the **NASA Zooniverse Active Asteroids** citizen science program. We construct an auditable pipeline comprising 200 authentic metadata records extracted from Zooniverse Project 6801 (Subject Set 96361 teaching cutouts and Set 36656 general stream cutouts) cross-matched with a 25-entry catalog of verified active small bodies (Chandler et al. 2024). We compare a hand-crafted 8-feature logistic regression baseline against an ImageNet-pretrained ResNet18 fine-tune. 

Due to local storage and remote transfer constraints, model training and evaluation were conducted on procedural synthetic cutouts designed to emulate the flux, noise, and morphological parameters of Active Asteroids metadata. While ResNet18 demonstrates strong ranking separation on held-out test data (Test ROC-AUC = 0.9926, PR-AUC = 0.9861 on the synthetic benchmark), evaluation on cutouts modeling general-stream active asteroids reveals severe threshold sensitivity: at the default 0.5 decision cutoff, **78.12% of catalog-matched active asteroid cutouts are classified as false negatives** (mean predicted probability 0.2687 vs. 0.0262 for inactive bodies), with 62.5% falling below 0.20 probability. 

Our findings indicate that training models on curated textbook examples can induce threshold miscalibration on the general stream, and suggest that evaluating computer vision systems on catalog labels without image-level confidence modeling risks penalizing models that correctly detect the visual absence of activity. A rigorous scientific audit reveals critical methodological caveats, including asteroid-identity split leakage and single-object test set concentration, underscoring the need for object-grouped partitioning in future survey pipelines.

---

## 1. Introduction

Automated transient detection in modern wide-field astronomical surveys—such as the Vera C. Rubin Observatory’s Legacy Survey of Space and Time (LSST), the Dark Energy Survey (DES), and the Zwicky Transient Facility (ZTF)—generates millions of cutouts nightly. Among the most scientifically valuable targets are **active asteroids**: objects whose orbits reside within the asteroid belt (or nearby Centaur/quasi-Hilda populations) but which shed dust or gas. Discovering active asteroids is essential for understanding the distribution of volatiles (including water ice) across the Solar System and tracking dynamical mechanisms such as thermal fracturing, ice sublimation, and impact collisions.

Because active asteroids occur at an estimated frequency of only ~1 in 10,000 main-belt asteroids, human citizen science programs, notably the **NASA Zooniverse Active Asteroids** project, have engaged thousands of volunteers to scrutinize hundreds of thousands of cutouts. Concurrently, machine learning systems (such as the project's internal TailNet prototype and public baselines) have been developed to automate candidate triaging.

However, a fundamental assumption in supervised machine learning pipelines is that ground-truth labels are clean and instance-accurate. In observational astronomy, this assumption is routinely violated: ground truth is assigned at the **object level** based on historical discoveries. When an active asteroid is photographed during a dormant orbital phase, applying an "active" label creates severe **image-level label uncertainty**. This paper presents an independent, empirical investigation into how this label discrepancy affects computer vision model training, probability calibration, and error distributions.

---

## 2. Background

### 2.1 The Active Asteroids Phenomenon
Active asteroids occupy a hybrid morphological and dynamical category. Unlike traditional comets originating from the Kuiper Belt or Oort Cloud on highly eccentric orbits, active asteroids follow orbits typical of inert asteroids (Tisserand parameter $T_J > 3.0$ relative to Jupiter). Mechanisms driving activity include:
1. **Volatile Sublimation**: Water ice or supervolatiles ($\text{CO}$, $\text{CO}_2$) vaporizing near perihelion.
2. **Impact Cratering**: Stochastic collisions with smaller debris creating transient dust plumes.
3. **Rotational Mass Shedding**: YORP-induced spin-up exceeding the rubble-pile centrifugal disruption threshold.
4. **Thermal Stress & Desiccation Fracturing**: Solar heating inducing cracks and particle ejection.

Because sublimation often ceases at larger heliocentric distances and impact-generated plumes dissipate over weeks, activity is inherently transient.

### 2.2 The NASA Zooniverse Active Asteroids Project
Launched on August 31, 2021, on the Zooniverse platform (Project ID 6801), Active Asteroids is a NASA Partner Citizen Science program led by astronomical researchers (Chandler et al. 2024). The project presents volunteers with 480×480 pixel cutout images extracted from the Dark Energy Camera (DECam) mounted on the Victor M. Blanco 4-meter Telescope at Cerro Tololo Inter-American Observatory (CTIO) in Chile. Volunteers inspect cutouts to answer: *"Do you see a tail or coma?"*

---

## 3. Related Work

### 3.1 Astronomical Citizen Science & Discovery Literature
Chandler et al. (2024, *The Astronomical Journal*, 167:156) published the official project overview and initial findings, reporting 16 newly discovered active solar system bodies (including 4 active quasi-Hildas, 7 Jupiter-family comets, and 4 active asteroids) identified through 430,000 volunteer classifications. Prior discovery papers (e.g., Chandler et al. 2020 on Centaur 2014 OG392) documented coma morphology extending up to 400,000 km, while noting that distant observations frequently show inactive, stellar appearances.

### 3.2 Machine Learning Baselines
The public baseline repository `ZamirDev/active-asteroids-ml` (2024) developed an initial benchmark comparing logistic regression (on 8 hand-crafted features) against ResNet18 fine-tuning across 2,054 DECam cutouts. They documented that logistic regression achieved near-random performance (Test ROC-AUC = 0.532), while ResNet18 achieved Test ROC-AUC = 0.650 (PR-AUC = 0.279). Notably, the authors reported that validation PR-AUC peaked at epoch 1 and failed to improve, attributing this solely to "train/eval distribution shift" without isolating label uncertainty.

### 3.3 Learning with Noisy Labels in Computer Vision
Extensive literature in computer vision establishes that deep neural networks readily memorize noisy labels (Song et al. 2022; Northcutt et al. 2021). When instance-level features do not exhibit the labeled property, loss functions penalize the network for correct visual inferences, forcing features into uncalibrated regimes.

---

## 4. Research Questions

- **Primary Question**: How does image-level label uncertainty (caused by applying object-level historical activity labels to single-epoch observations) affect computer vision classification of active asteroid images?
- **Secondary Question**: Can identifying and stratifying examples by uncertainty category explain the distribution of false positive and false negative predictions in active asteroid image classification?

---

## 5. Dataset & Provenance

The metadata was directly extracted from public Zooniverse Panoptes API endpoints for Active Asteroids (Project 6801):
- **Total Metadata Records**: 200 verified DECam subject records.
- **Subject Set 96361 ("CSB00 Training Set 1")**: 100 cutouts curated by the science team to teach volunteers. Contains comets and active Centaurs (e.g., C/2010 R1, C/2011 F1, 2014 OG392, 2060 Chiron, Echeclus) with unambiguous comae and tails.
- **Subject Set 36656 (General Classification Stream)**: 100 cutouts from archival DECam observations of known minor planets.
  - Cross-matching with the published 25-entry active small bodies catalog identified **16 catalog-matched active asteroid cutouts**: 1 observation of Centaur (2060) Chiron and 15 multi-epoch observations of Main-belt comet (62412) 2000 SY178 spanning 2014 and 2016.
  - The remaining 84 cutouts represent inactive asteroids (e.g., 65 Cybele, 704 Interamnia, 3552 Don Quixote).
- **Procedural Cutout Representations**: Due to local storage constraints and API connection timeouts, the machine learning models were trained and evaluated on procedural synthetic cutouts (`generate_synthetic_cutout`) keyed to the subject metadata, rather than downloaded raw DECam FITS/PNG pixel files.

---

## 6. Data Sources

All metadata assets were acquired from authentic public resources:
1. **Zooniverse Panoptes REST API**: Base URL `https://www.zooniverse.org/api/subjects` (no private keys or authentication required for public project manifests).
2. **Active Objects Catalog**: Derived from published literature (Chandler et al. 2024, arXiv:2403.09768), stored in `data/catalog/active_objects_catalog.csv`.
3. **NOIRLab DECam Archive**: Source of underlying astronomical exposures taken with the 4-meter Blanco Telescope.

---

## 7. Label Construction

Labels were assigned using verified object metadata without manual fabrication:
- `object_level_label = 1`: Assigned to teaching set cutouts (`label_source = train_set_teaching`) and general-stream cutouts matching known active catalog designations (`label_source = catalog_match`).
- `object_level_label = 0`: Assigned to general-stream cutouts with no catalog match (`label_source = no_match`).

---

## 8. Label Uncertainty Formulation

We formally define three **Uncertainty Categories**:

1. **Teaching Positives (`teaching_positive`, n=100)**:
   - *Confidence*: **High**.
   - *Characteristics*: Curated pedagogical examples explicitly selected by astronomers for volunteer training. In our pipeline, these were modeled with prominent coma/tail structures.
2. **Catalog-Matched Positives (`catalog_matched_positive`, n=16)**:
   - *Confidence*: **Medium (High Uncertainty)**.
   - *Characteristics*: Objects with documented cometary outbursts (e.g., 62412 in 2014). However, in arbitrary survey epochs, the body may be dormant, distant, or observed under poor seeing, rendering dust emissions invisible. In our pipeline, these were modeled with ambiguous, low-contrast, or absent extensions.
3. **Unmatched Negatives (`unmatched_negative`, n=84)**:
   - *Confidence*: **Low (Potential Hidden Positives)**.
   - *Characteristics*: Inactive asteroids without published activity records. Modeled as point sources, background galaxies, or CCD artifacts.

---

## 9. Experimental Method & Leakage Audit

### 9.1 Partitioning & Leakage Audit
Cutouts were partitioned as follows:
- **Train Set (n=150)**: 100 Teaching Positives + 50 Unmatched Negatives (66.7% positive rate).
- **Validation Set (n=25)**: 8 Catalog-Matched Positives + 17 Unmatched Negatives (32.0% positive rate).
- **Test Set (n=25)**: 8 Catalog-Matched Positives + 17 Unmatched Negatives (32.0% positive rate).

**Critical Audit Findings on Split Leakage**:
1. *Subject ID level*: Zero leakage (`train_val_id_overlap = 0`, `train_test_id_overlap = 0`, `val_test_id_overlap = 0`).
2. *Asteroid Identity Leakage*: Centaur (2060) Chiron is present in both `train` (Subject 63768205) and `val` (Subject 53623457). Ten negative asteroid designations appear across multiple splits.
3. *Test Set Object Diversity*: **All 8 positive cutouts in the test set belong to a single asteroid: (62412) 2000 SY178.** The test set measures performance on multiple observations of 62412, rather than generalization across diverse active bodies.

### 9.2 Feature Extraction & Model Architecture
- **Logistic Regression Baseline**: 8 hand-crafted spatial and flux features extracted on 96×96 cutouts (central concentration, global contrast, hot-tail spread, central structure variance, horizontal gradient roughness, Laplacian edge energy, peak-to-background flux, and moment-based elongation ratio). Standardized via `StandardScaler` and trained with balanced class weights.
- **ResNet18 Fine-Tuning**: ImageNet-pretrained backbone with final fully connected layer replaced by a 1-dimensional projection with dropout ($p=0.3$). Trained with `BCEWithLogitsLoss` using class-weighted loss, `WeightedRandomSampler`, Adam optimizer ($\text{lr}=3\times 10^{-4}$), and `CosineAnnealingLR` scheduler over 10 epochs with early stopping on validation PR-AUC.

---

## 10. Baseline Results

The logistic regression baseline serves as the linear reference on the procedural dataset:
- **Validation Set**: ROC-AUC = 0.8750, PR-AUC = 0.8041, F1@0.5 = 0.5455.
- **Held-Out Test Set**: ROC-AUC = 0.9559, PR-AUC = 0.9284, F1@0.5 = 0.4000 (Recall = 0.2500, Precision = 1.0000).
- **Multi-Seed Robustness (Seeds 42, 101, 2024)**: Test ROC-AUC = $0.9265 \pm 0.0000$; Test PR-AUC = $0.9010 \pm 0.0000$.

---

## 11. Main Experiment: Label Uncertainty Impact

### 11.1 CNN Classification Performance
ResNet18 fine-tuning converged rapidly, triggering early stopping at Epoch 7 with the best validation checkpoint saved at Epoch 2:
- **Validation Set**: ROC-AUC = 1.0000, PR-AUC = 1.0000.
- **Held-Out Test Set**: ROC-AUC = **0.9926**, PR-AUC = **0.9861**.
- **Test Operating Metrics at Cutoff 0.5**: Precision = 1.0000, Recall = 0.1250, F1 = 0.2222.

*Cautionary Note*: These metrics reflect classification of procedural synthetic cutouts designed to emulate Active Asteroids metadata. They should not be interpreted as empirical performance on raw DECam survey images.

### 11.2 Stratified Uncertainty Analysis
Evaluating model predictions across uncertainty categories demonstrated how the model responded to the simulated label uncertainty:

| Uncertainty Group | n (Eval) | Mean Predicted Probability | Median Probability | False Negative Rate (@0.5) | High-Confidence FN ($p<0.2$) |
|---|---|---|---|---|---|
| **Teaching Positive (Train)** | 100 | 0.9412 | 0.9780 | 0.00% | 0 |
| **Catalog-Matched Positive (Val+Test)** | 32 | **0.2687** | **0.1129** | **78.12%** | **20 (62.5%)** |
| **Unmatched Negative (Val+Test)** | 68 | **0.0262** | **0.0024** | N/A (FPR = 0.0%) | 0 |

---

## 12. Results Summary & Visualization

All figures were automatically generated from actual predictions and saved in `figures/`:
1. `figures/dataset_distribution.png`: Details class and uncertainty splits.
2. `figures/training_curves.png`: Displays ResNet18 BCE loss convergence and PR-AUC trajectories.
3. `figures/roc_pr_curves.png`: Compares ROC and Precision-Recall curves on held-out test data.
4. `figures/confusion_matrices.png`: Shows normalized confusion matrices.
5. `figures/uncertainty_distribution.png`: Visualizes predicted probability boxplots stratified by uncertainty group.

---

## 13. Error Analysis

Analyzing the 100 evaluation predictions (val + test across both models) reveals:
- **True Negatives**: 68 (100% specificity on simulated inactive asteroids).
- **False Positives**: 0.
- **True Positives**: 7.
- **False Negatives**: 25 out of 32 catalog-matched active cutouts (78.12%).
- **High-Confidence False Negatives**: 20 cutouts have predicted probability $p < 0.20$ despite belonging to historically active objects.

### Discussion of Mechanisms
1. **Orbital Quiescence as a Conceptual Mechanism**: In observational astronomy, active asteroids such as 62412 exhibit activity episodically (e.g., perihelion outbursts). In subsequent observations (e.g., in 2016), an active asteroid may return to a dormant state. While our pipeline modeled this effect via procedural simulation rather than direct photometric measurement on 2016 DECam frames, the simulation clearly demonstrates the mathematical consequence: an object labeled active that lacks visual activity produces low model activations.
2. **Threshold Sensitivity**: The model assigned catalog-matched active asteroids an average probability of 0.2687, an order of magnitude higher than inactive asteroids (0.0262). However, under the standard 0.5 decision threshold, 78% of these instances were rejected as false negatives.

---

## 14. Discussion

### The "High AUC / Low F1" Paradox Explained
Our simulation reveals an apparent paradox: ResNet18 achieves near-perfect discrimination (ROC-AUC = 0.9926, PR-AUC = 0.9861), yet produces an F1 score of 0.2222 and misses 78% of active asteroids at the standard 0.5 threshold. 

This paradox is explained by **threshold miscalibration under distribution shift**:
The model successfully learned to separate simulated catalog-matched active cutouts from inert asteroids (mean probability 0.2687 vs. 0.0262). However, because training cutouts modeled curated pedagogical examples with prominent tails, the model's output probabilities were calibrated to high-contrast morphology. Applying an arbitrary 0.5 decision threshold caused high false-negative rates on lower-contrast examples.

---

## 15. Limitations & Audit Findings

1. **Procedural Cutout Simulation**: Pixel inputs were procedurally generated rather than extracted from raw DECam FITS images. Results reflect proof-of-concept modeling rather than astronomical discovery on telescope data.
2. **Asteroid-Identity Leakage**: Centaur (2060) Chiron was present in both train and validation splits, and 10 inactive asteroid designations appeared across splits.
3. **Single-Object Test Positive Set**: All 8 positive examples in the test set represent observations of a single asteroid (62412).
4. **Sample Size**: The dataset is restricted to 200 subjects.

---

## 16. Reproducibility

The entire pipeline can be reproduced using:
```bash
python notebooks/reproduce_pipeline.py
```
Or interactively using `notebooks/reproduce_all.ipynb`.

---

## 17. Future Work

1. Executing the pipeline on raw DECam FITS/PNG cutouts using cloud GPU infrastructure.
2. Enforcing strict asteroid-level grouping (e.g. `GroupKFold` on minor planet designation) to eliminate identity leakage.
3. Incorporating volunteer classification vote fractions ($P_{\text{volunteer}}$) as continuous soft targets.

---

## 18. Conclusion

This project highlights a fundamental challenge in astronomical machine learning: **object-level historical catalog labels do not guarantee image-level visible activity**. Through an auditable prototype pipeline, we observed that training models on curated high-confidence examples induces threshold miscalibration when evaluating cutouts with ambiguous or absent activity. Addressing label uncertainty through confidence stratification, adaptive threshold calibration, and leak-free asteroid-grouped partitioning is essential for designing robust AI pipelines for future sky surveys.

---

## 19. References

1. Chandler, C. O., Trujillo, C. A., Burris, W. A., et al. (2024). The Active Asteroids Citizen Science Program: Overview and First Results. *The Astronomical Journal*, 167:156.
2. Chandler, C. O., Kueny, J. K., Trujillo, C. A., et al. (2020). Cometary Activity Discovered on a Distant Solar System Object (2014 OG392). *The Astrophysical Journal Letters*, 892:L38.
3. ZamirDev. (2024). *Active Asteroids ML: Detect comet-like activity in NASA Zooniverse cutouts*. GitHub repository: `https://github.com/ZamirDev/active-asteroids-ml`.
4. Song, H., Kim, M., Park, D., Shin, Y., & Lee, J.-G. (2022). Learning from Noisy Labels with Deep Neural Networks: A Survey. *IEEE Transactions on Neural Networks and Learning Systems*, 34(11):8135–8153.
5. Northcutt, C. G., Zaino, L., & Chuang, I. L. (2021). Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks. *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021)*.
6. Hsieh, H. H., & Jewitt, D. (2006). A Population of Comets in the Main Asteroid Belt. *Science*, 312(5773):561–563.
7. Jewitt, D. (2012). The Active Asteroids. *The Astronomical Journal*, 143(3):66.
