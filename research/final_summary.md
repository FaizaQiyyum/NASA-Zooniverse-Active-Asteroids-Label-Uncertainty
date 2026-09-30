# NASA Zooniverse Active Asteroids: Final Research Summary (Post-Audit)

## 1. Research Questions
- **Primary**: How does image-level label uncertainty (arising from applying historical object-level catalog labels to single-epoch observations) affect computer vision classification of active asteroid cutouts?
- **Secondary**: Can stratifying cutouts by label confidence explain the mechanism behind false positive and false negative error rates?

---

## 2. Dataset & Scale
- **Data Source**: Authentic public metadata and cutout identifiers fetched from the NASA Zooniverse Active Asteroids project (Panoptes API, Project ID `6801`).
- **Total Metadata Records**: **200 cutouts**.
  - **100 Teaching Positives** (Subject Set `96361`): Curated active comets and Centaurs (e.g., Chiron, 2014 OG392, Echeclus, C/2010 R1).
  - **16 Catalog-Matched Positives** (Subject Set `36656`): 1 observation of (2060) Chiron and 15 observations of Main-belt comet (62412) 2000 SY178.
  - **84 Unmatched Negatives** (Subject Set `36656`): Inactive asteroids (e.g., 65 Cybele, 704 Interamnia, 3552 Don Quixote).
- **Procedural Simulation**: Cutouts were evaluated using procedural synthetic cutouts modeled after Active Asteroids parameters due to local storage limits.
- **Split Distribution**:
  - Train: 100 teaching positives + 50 unmatched negatives (n=150)
  - Validation: 8 catalog-matched positives + 17 unmatched negatives (n=25)
  - Test: 8 catalog-matched positives + 17 unmatched negatives (n=25)

---

## 3. Label Methodology
- **Binary Ground Truth**: `1` = Historically active object (catalog match or teaching set); `0` = No catalog match.
- **Label Confidence Stratification**:
  - `High Confidence` (`teaching_positive`): Curated training examples with visible activity.
  - `Medium Confidence` (`catalog_matched_positive`): Historically active objects, but visible activity in any given single exposure is uncertain due to orbital dormancy or seeing conditions.
  - `Low Confidence` (`unmatched_negative`): Assumed inactive, but with potential for undiscovered faint activity.

---

## 4. Models Evaluated
1. **Baseline**: Logistic Regression on 8 hand-crafted spatial/flux features (central concentration, global contrast, hot-tail spread, centre variance, horizontal roughness, Laplacian edge energy, peak-to-background, moment elongation).
2. **Deep Learning Model**: ResNet18 fine-tuning with ImageNet pre-trained weights, class-weighted binary cross-entropy loss (`BCEWithLogitsLoss`), and `WeightedRandomSampler`.

---

## 5. Empirical Results (Procedural Benchmark)

### Baseline (Logistic Regression)
- **Validation**: ROC-AUC = 0.8750, PR-AUC = 0.8041, F1@0.5 = 0.5455
- **Held-Out Test**: ROC-AUC = 0.9559, PR-AUC = 0.9284, F1@0.5 = 0.4000 (Precision = 1.000, Recall = 0.250)
- **Multi-Seed Robustness (Seeds 42, 101, 2024)**: Test ROC-AUC = 0.9265 ± 0.0000; Test PR-AUC = 0.9010 ± 0.0000

### Main Experiment (ResNet18 Fine-Tuning)
- **Validation**: ROC-AUC = 1.0000, PR-AUC = 1.0000
- **Held-Out Test**: ROC-AUC = **0.9926**, PR-AUC = **0.9861**
- **Operating Metrics at Cutoff 0.5**: Precision = 1.0000, Recall = 0.1250, F1 = **0.2222**

---

## 6. Error Analysis Results
Across all 100 evaluation predictions (val + test across both models on the benchmark):
- **True Negatives**: 68 (0% false discovery rate on inactive asteroids).
- **False Positives**: 0.
- **True Positives**: 7.
- **False Negatives**: 25 out of 32 catalog-matched active cutouts (**78.12% false negative rate**).
- **High-Confidence False Negatives ($p < 0.20$)**: 20 out of 32 (62.5%).

---

## 7. Most Interesting Finding
**The Threshold Miscalibration Effect**:
The fine-tuned CNN achieved high discriminatory ranking (Test ROC-AUC = 0.9926 on the benchmark), yet at the standard 0.5 decision threshold, it missed over 78% of simulated active asteroids. 

The model's predicted probability for catalog-matched active asteroids averaged **0.2687**—substantially higher than inactive asteroids (**0.0262**), but below the default 0.5 threshold. Because the model was trained on curated high-contrast examples, its decision boundaries became miscalibrated for lower-contrast or ambiguous observations.

---

## 8. Audit Limitations
1. **Procedural Cutout Simulation**: Results were evaluated on procedural synthetic cutouts rather than downloaded raw DECam FITS images.
2. **Asteroid-Identity Leakage**: Chiron appears in both train and validation splits, and 10 inactive asteroid designations appear across multiple splits.
3. **Single-Object Test Positives**: All 8 positive examples in the test set represent observations of a single asteroid (62412).
4. **Epistemological Scope**: Demonstrates the mathematical and pipeline consequences of label uncertainty in simulation, but does not claim astronomical discovery on telescope data.

---

## 9. Future Research
1. Executing on raw DECam FITS/PNG cutouts using cloud GPU infrastructure.
2. Applying `GroupKFold` on asteroid designation to guarantee zero identity leakage.
3. Incorporating volunteer classification vote fractions ($P_{\text{volunteer}}$) as continuous soft targets.
