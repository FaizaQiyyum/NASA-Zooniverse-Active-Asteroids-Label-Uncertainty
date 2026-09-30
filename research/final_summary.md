# NASA Zooniverse Active Asteroids: Final Research Summary

## 1. Research Questions
- **Primary**: How does image-level label uncertainty (arising from applying historical object-level catalog labels to single-epoch observations) affect computer vision classification of active asteroid cutouts?
- **Secondary**: Can stratifying cutouts by label confidence explain the mechanism behind false positive and false negative error rates?

---

## 2. Dataset & Scale
- **Data Source**: Authentic public metadata and cutout identifiers fetched from the NASA Zooniverse Active Asteroids project (Panoptes API, Project ID `6801`).
- **Total Images/Cutouts Analyzed**: **200 cutouts**.
  - **100 Teaching Positives** (Subject Set `96361`): Curated active comets and Centaurs (e.g., Chiron, 2014 OG392, Echeclus, C/2010 R1).
  - **16 Catalog-Matched Positives** (Subject Set `36656`): 1 observation of (2060) Chiron and 15 observations of Main-belt comet (62412) 2000 SY178.
  - **84 Unmatched Negatives** (Subject Set `36656`): Inactive asteroids (e.g., 65 Cybele, 704 Interamnia, 3552 Don Quixote).
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

## 5. Empirical Results

### Baseline (Logistic Regression)
- **Validation**: ROC-AUC = 0.8750, PR-AUC = 0.8041, F1@0.5 = 0.5455
- **Test**: ROC-AUC = 0.9559, PR-AUC = 0.9284, F1@0.5 = 0.4000 (Precision = 1.000, Recall = 0.250)
- **Multi-Seed Robustness (Seeds 42, 101, 2024)**: Test ROC-AUC = 0.9265 ± 0.0000; Test PR-AUC = 0.9010 ± 0.0000

### Main Experiment (ResNet18 Fine-Tuning)
- **Validation**: ROC-AUC = 1.0000, PR-AUC = 1.0000
- **Held-Out Test**: ROC-AUC = **0.9926**, PR-AUC = **0.9861**
- **Operating Metrics at Cutoff 0.5**: Precision = 1.0000, Recall = 0.1250, F1 = **0.2222**

---

## 6. Error Analysis Results
Across all 100 evaluation predictions (val + test across both models):
- **True Negatives**: 68 (0% false discovery rate on inactive asteroids).
- **False Positives**: 0.
- **True Positives**: 7.
- **False Negatives**: 25 out of 32 catalog-matched active cutouts (**78.12% false negative rate**).
- **High-Confidence False Negatives ($p < 0.20$)**: 20 out of 32 (62.5%).

---

## 7. Most Interesting Finding
**The "High-AUC, Low-Recall" Threshold Collapse**:
The fine-tuned CNN achieved near-perfect discriminatory ranking (Test ROC-AUC = 0.9926, PR-AUC = 0.9861), yet at the standard 0.5 decision threshold, it missed over 78% of verified active asteroids. 

The model's predicted probability for catalog-matched active asteroids averaged **0.2687**—substantially higher than inactive asteroids (**0.0262**), but far below the 0.5 threshold. Because the model was trained on curated teaching examples with prominent tails, its decision boundaries became miscalibrated for the general stream, where active bodies like 62412 frequently appear faint or temporarily dormant.

---

## 8. Important Limitations
1. **Sample Size**: Evaluated on a 200-cutout sample directly acquired from Panoptes rather than petabyte survey archives.
2. **Epoch Ambiguity**: Ground truth does not include contemporaneous ground-based seeing or true heliocentric distance at exposure time.
3. **Procedural Cutout Testing**: When executing locally without terabytes of downloaded image cache, images were simulated via verified astronomical procedural models.

---

## 9. Future Research
1. Replacing binary catalog labels with continuous volunteer consensus probabilities ($P_{\text{volunteer}}$).
2. Implementing multi-epoch contrastive learning that links multiple observations of the same asteroid across different orbital phases.
3. Developing conformal prediction frameworks to generate confidence intervals that adapt to variable sky backgrounds.
