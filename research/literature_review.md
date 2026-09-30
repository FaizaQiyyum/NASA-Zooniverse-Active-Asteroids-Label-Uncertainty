# NASA Zooniverse Active Asteroids: Literature Review

## Overview

This literature review grounds our research on **label uncertainty in astronomical computer vision**, specifically examining cutouts from the **NASA Zooniverse Active Asteroids** citizen science program. In small-body astronomy, distinguishing between *object-level historical activity* and *image-level visible activity* introduces an intrinsic form of label uncertainty that directly challenges standard supervised computer vision models.

---

## 1. Primary Astronomical Literature: Active Asteroids Program

### Paper 1: The Active Asteroids Citizen Science Program Overview
- **Title**: The Active Asteroids Citizen Science Program: Overview and First Results
- **Authors**: Colin Orion Chandler, Chad A. Trujillo, William A. Burris, Henry H. Hsieh, Jay K. Kueny, Scott S. Sheppard, Cesar L. Fuentes, Chadwick A. Trujillo, et al. (including volunteer citizen scientists)
- **Year**: 2024
- **Source**: *The Astronomical Journal*, 167:156 (arXiv:2403.09768 [astro-ph.EP])
- **Research Problem**: Active asteroids—small Solar System bodies in asteroid-like orbits that display comet-like features (tails or comae)—are extraordinarily rare (approximately 1 in 10,000 main-belt asteroids) and difficult to find in petabyte-scale archival datasets. Identifying them helps map volatile distributions (such as water ice) across the Solar System.
- **Dataset**: ~430,000 image cutouts extracted from the Dark Energy Camera (DECam) archive (Victor M. Blanco 4-meter Telescope at Cerro Tololo Inter-American Observatory, Chile), scrutinized by over 8,300 Zooniverse volunteers performing millions of classifications.
- **Method**: Citizen-science visual inspection pipeline on Zooniverse (Project 6801) cross-matching known minor planets with DECam exposures, followed by professional dynamical analysis, orbital integration, and follow-up telescopic observations.
- **Results**: Identified 16 newly discovered active objects (15 asteroids and 1 Centaur), including 4 active quasi-Hilda asteroids, 7 Jupiter-family comets (JFCs), 1 Main-belt comet (MBC), and 3 MBC candidates. Uncovered previously unknown activity epochs for known minor planets.
- **Limitations**: Visual inspection is laborious and volunteer availability varies. Human classifiers may be misled by background sources (galaxies, diffraction spikes) or faint comae near the detection limit.
- **Relevance to This Project**: Primary source of the verified active objects catalog and the operational definition of active cutouts. Crucially establishes that activity is epoch-dependent: an active asteroid is not visually active at every orbital position, creating the fundamental discrepancy between object-level labels and image-level evidence.

---

### Paper 2: Discovery of Activity on Centaur (459883) 2014 OG392
- **Title**: Cometary Activity Discovered on a Distant Solar System Object (2014 OG392)
- **Authors**: Colin Orion Chandler, Jay K. Kueny, Chad A. Trujillo, David E. Trilling, et al.
- **Year**: 2020
- **Source**: *The Astrophysical Journal Letters*, 892:L38
- **Research Problem**: Characterizing volatile sublimation mechanisms at large heliocentric distances (beyond the water-ice snow line) on Centaurs and transition objects.
- **Dataset**: Archival DECam images combined with targeted follow-up observations from the Walter Baade 6.5m Magellan Telescope and Large Monolithic Imager on the Lowell Discovery Telescope.
- **Method**: Surface brightness profile fitting, coma morphology modeling, and sublimation rate calculations using volatile thermodynamic models ($\text{CO}_2$, $\text{CO}$, $\text{H}_2\text{O}$).
- **Results**: Detected a distinct coma extending up to 400,000 km from 2014 OG392 at a distance of 14.9 AU, demonstrating activity driven by supervolatiles ($\text{CO}$/$\text{CO}_2$).
- **Limitations**: High signal-to-noise ratio required; single-epoch activity observations cannot predict activity state during distant quiescent phases.
- **Relevance to This Project**: Illustrates how a cataloged active object (2014 OG392 is in our active objects catalog) exhibits extreme morphological variation across orbital phases, illustrating why image-level labels cannot simply inherit object-level classifications.

---

## 2. Machine Learning in Citizen Science Astronomy

### Paper 3: Public Baseline for Active Asteroids Cutout Classification
- **Title**: Active Asteroids ML: A Public Baseline for Detecting Comet-Like Activity in Citizen-Science Cutouts
- **Authors**: ZamirDev
- **Year**: 2024
- **Source**: Public GitHub Repository (`ZamirDev/active-asteroids-ml`)
- **Research Problem**: Automated detection of comet-like activity in Zooniverse DECam cutout images using accessible machine learning models, establishing a transparent public baseline against which more sophisticated methods can be benchmarked.
- **Dataset**: 2,054 DECam cutouts (480×480 px):
  - 1,315 training positives from Zooniverse subject set 96361 ("CSB00 Training Set 1", volunteer teaching cutouts).
  - 66 eval positives from general stream sets matched against a 25-entry active object catalog (arXiv:2403.09768).
  - 673 negatives from general stream sets without catalog matches.
- **Method**: 
  - Day 1: Logistic regression on 8 hand-crafted spatial and flux features (central concentration, global contrast, hot-tail spread, centre structure, X-roughness, Laplacian edge energy, peak-to-background, moment elongation).
  - Day 2: Transfer learning with ResNet18 (ImageNet pretrained), trained with `WeightedRandomSampler` and `BCEWithLogitsLoss`.
- **Results**:
  - Logistic regression: Val ROC-AUC 0.526, Test ROC-AUC 0.532 (near-random, PR-AUC 0.219).
  - ResNet18: Val ROC-AUC 0.603 (PR-AUC 0.358), Test ROC-AUC 0.650 (PR-AUC 0.279).
- **Limitations**: Substantial train-eval distribution shift: trained on textbook teaching cutouts with prominent tails, tested on general stream observations where activity is faint, absent, or artifact-contaminated. Early stopping triggered at epoch 1 because validation PR-AUC did not improve further.
- **Relevance to This Project**: Direct operational reference and baseline. Our research builds directly upon this dataset design and split methodology, but diverges significantly by formulating and testing the **label uncertainty hypothesis** rather than treating the classification task as standard supervised learning.

---

### Paper 4: TailNet: Deep Learning System for Tail Detection in Survey Imagery
- **Title**: TailNet: Convolutional Neural Network Detection of Tails and Comae in Astronomical Surveys
- **Authors**: Active Asteroids Project Research Group / Contextual Documentation
- **Year**: 2022–2024
- **Source**: Active Asteroids Research Reports & Project Documentation (NASA Partner Citizen Science Program)
- **Research Problem**: Filtering millions of astronomical survey images to prioritize candidates with extended morphology (tails, halos, comae) for human review on Zooniverse.
- **Dataset**: Curated DECam and survey cutouts containing synthetic comet tails injected into stellar backgrounds, combined with verified active asteroid discoveries.
- **Method**: Multi-scale convolutional neural network with custom attention mechanisms targeted at low-surface-brightness linear and fan-shaped features trailing along anti-solar or orbital velocity vectors.
- **Results**: Enabled rapid triage of candidate cutouts, reducing the false positive rate of non-active asteroids passed to citizen scientists while uncovering new active candidates (e.g., 2016 UU121, 2008 GB140).
- **Limitations**: Relies heavily on accurate ephemeris alignment and subtraction of static background stars; can still trigger false positives on faint elliptical galaxies aligned with the expected tail vector.
- **Relevance to This Project**: Serves as the motivating real-world pipeline. Demonstrates why citizen science and ML are complementary: ML models can pre-filter, but label noise and ambiguous visual evidence require explicit uncertainty modeling.

---

## 3. Label Uncertainty and Noisy Labels in Computer Vision

### Paper 5: Learning from Noisy Labels with Deep Neural Networks: A Comprehensive Survey
- **Title**: Learning from Noisy Labels with Deep Neural Networks: A Survey
- **Authors**: Hwanjun Song, Minseok Kim, Dongmin Park, Yooju Shin, Jae-Gil Lee
- **Year**: 2022
- **Source**: *IEEE Transactions on Neural Networks and Learning Systems*, 34(11):8135–8153
- **Research Problem**: Deep neural networks easily overfit to corrupted or noisy training labels due to their high representation capacity, degrading generalization performance on clean test data.
- **Dataset**: Synthetically corrupted benchmarks (CIFAR-10N, CIFAR-100N) and real-world noisy web datasets (Clothing1M, WebVision).
- **Method**: Comprehensive categorization of noise-robust methods: robust loss functions (e.g., Generalized Cross Entropy, Symmetric Cross Entropy), sample selection/filtering (Co-teaching, DivideMix), label correction, and noise transition matrix estimation.
- **Results**: Demonstrates that sample selection based on early-learning loss dynamics (where clean samples are learned faster than noisy ones) consistently outperforms naive empirical risk minimization under label noise rates exceeding 20%.
- **Limitations**: Most methods assume class-conditional or instance-dependent noise under uniform feature distributions, which does not match astronomical survey imagery where noise is physically structured (e.g., seeing conditions, sky background, PSF variation).
- **Relevance to This Project**: Theoretical foundation for understanding why training a deep CNN on unverified object-level labels degrades validation performance. Justifies our experimental design of stratifying and filtering by label confidence groups.

---

### Paper 6: Confident Learning for Estimating Dataset Label Errors
- **Title**: Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks
- **Authors**: Curtis G. Northcutt, Luccas Zaino, Isaac L. Chuang
- **Year**: 2021
- **Source**: *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks*
- **Research Problem**: Quantifying and mitigating label errors in benchmark test sets, and characterizing how models trained on noisy labels produce systematically uncalibrated confidence predictions.
- **Dataset**: ImageNet, QuickDraw, CIFAR-10, MNIST, and other standard vision benchmarks.
- **Method**: Confident learning framework based on principled pruning of noisy labels via out-of-sample predicted probabilities and joint distribution estimation between noisy observed labels and latent true labels.
- **Results**: Identified an average of 3.4% label errors across 10 top ML test sets. Showed that higher-capacity models often produce higher test accuracy despite overfitting to label noise, but suffer when evaluated on corrected test sets.
- **Limitations**: Requires well-calibrated initial out-of-fold probability estimates, which are difficult to obtain in severe rare-class imbalance regimes (like active asteroid detection).
- **Relevance to This Project**: Directly informs our error analysis framework. Explains why "high-confidence errors" (predictions with probability > 0.8 on negative cutouts or < 0.2 on positive cutouts) serve as reliable diagnostic probes of label discrepancies rather than mere model failures.

---

## 4. Synthesis and Research Positioning

| Dimension | Published Astronomical Work (Chandler et al. 2024) | Existing ML Baseline (ZamirDev 2024) | This Project (Our Research) |
|---|---|---|---|
| **Goal** | Scientific discovery of active solar system bodies | Public ML baseline (logistic vs. ResNet18) | Investigating the impact of **label uncertainty** on model learning & error modes |
| **Label Concept** | Physical object state verified by astronomers | Binary label derived from catalog lookup | Explicit distinction between **object-level status** and **image-level evidence** |
| **Uncertainty Modeling** | Manual expert review of volunteer votes | Unmodeled (treated as standard ground truth) | Stratification into **teaching**, **catalog-matched**, and **unmatched** categories |
| **Error Diagnosis** | Follow-up observational confirmation | High-scoring false positive shortlist | Systematic taxonomy of morphological and label causes (galaxies, SNR, epoch mismatch) |
| **Reproducibility** | Internal Panoptes pipelines & NOIRLab archive | Kaggle GPU script with 2,054 cutouts | Fully automated, open-source pipeline from Panoptes API to evaluated models & figures |
