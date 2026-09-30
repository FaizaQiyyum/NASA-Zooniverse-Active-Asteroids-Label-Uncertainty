# Independent Scientific Audit Report
## Project: NASA Zooniverse Active Asteroids — Label Uncertainty in Computer Vision

**Audit Date**: 2026-09-30  
**Auditor Role**: Independent Scientific Integrity & Verification Audit  
**Audit Objective**: Strict, non-negotiable verification of every factual, technical, astronomical, and statistical claim. No results are modified to appear more favorable.

---

## 1. Executive Summary

This independent audit evaluated 23 discrete claims across the repository `NASA-Zooniverse-Active-Asteroids-Label-Uncertainty`. 

### Key Findings:
1. **Metadata & Catalogs are Authentic**: The project successfully retrieved 200 real metadata records from the official Zooniverse Panoptes REST API (Project ID 6801, Subject Sets 96361 and 36656). The 25-entry active asteroid catalog is derived from genuine published literature (Chandler et al. 2024, arXiv:2403.09768).
2. **Critical Image Pixel Finding (SIMULATION DISCLOSURE)**: Due to local storage constraints and remote API download interruptions, model training and evaluation were executed on **procedurally simulated cutout images** (`generate_synthetic_cutout`) rather than downloaded raw DECam astronomical pixel files. While the code correctly implements the full deep learning and feature extraction pipelines, **claims implying that the convolutional network observed actual astronomical comae or "detected orbital quiescence in 2016 DECam exposures" are UNSUPPORTED and INCORRECT.**
3. **Asteroid-Identity Leakage**: While splits are strictly leak-free at the `subject_id` and image URL level, **leakage exists at the asteroid-identity level**:
   - Centaur (2060) Chiron is present in both `train` (Subject 63768205) and `val` (Subject 53623457).
   - 10 inactive asteroid designations cross split boundaries (e.g., Asteroid_457175 appears in train, val, and test).
   - **Test Set Diversity Failure**: All 8 positive examples in the held-out test set are multi-epoch observations of a single asteroid: Main-Belt Comet (62412) 2000 SY178. The test set therefore tests repeatability on 62412, not generalization across distinct active asteroids.
4. **Metric Validity**: Numerical metrics (Test ROC-AUC = 0.9926, PR-AUC = 0.9861, 78.12% false negative rate) are mathematically verified against the execution logs, but reflect the behavior of models on **procedural synthetic benchmarks**, not raw astronomical survey data.
5. **Epistemological Status**: The project successfully formulates a valid research question and demonstrates proof-of-concept pipeline engineering, but answers the question within a **simulation and methodology framework**, not as an empirical discovery on DECam images.

---

## 2. Individual Claim Audits (23 Items)

| # | Claim Audited | Audit Status | Evidence / File / Code / Reason |
|---|---|---|---|
| **1** | Dataset provenance | **PARTIALLY VERIFIED** | Metadata in `data/metadata/subjects.csv` is authentic from Panoptes API. However, cutout pixels were procedurally generated in `src/preprocessing/preprocess.py` (`use_synthetic=True`), not downloaded DECam pixels. |
| **2** | Zooniverse API source | **VERIFIED** | Verified query to `https://www.zooniverse.org/api/subjects` in `src/data/fetch_metadata.py`. Raw JSON payloads preserved in `data/metadata/subjects_full.csv`. |
| **3** | Project ID 6801 | **VERIFIED** | Official public ID for NASA Partner Active Asteroids on Zooniverse (`src/data/fetch_metadata.py:27`). |
| **4** | Subject sets 96361 & 36656 | **VERIFIED** | Set 96361 is the verified CSB00 teaching set; Set 36656 is the verified general classification stream set. Authenticated via raw Panoptes metadata. |
| **5** | The 200 image dataset | **PARTIALLY VERIFIED** | 200 authentic metadata records exist. However, the pixel data evaluated were 200 synthetic cutouts, not 200 downloaded astronomical images. |
| **6** | 25-object active asteroid catalog | **VERIFIED** | `data/catalog/active_objects_catalog.csv` contains 25 genuine active objects from Chandler et al. (2024, AJ 167:156). |
| **7** | Every positive and negative label | **PARTIALLY VERIFIED** | Teaching positives are verified curated active objects. The 16 catalog matches are verified historically active bodies. However, individual exposure activity was unverified; unmatched negatives are assumed inactive without exhaustive confirmation. |
| **8** | Train validation test split | **PARTIALLY VERIFIED** | Partitioned 150/25/25 in `src/data/build_clean_splits.py`. Leak-free for `subject_id`, but contaminated at the asteroid identity level. |
| **9** | Asteroid-level leakage | **INCORRECT** | Asteroid-level leakage IS present. Centaur 2060 Chiron appears in both `train` and `val`. 10 negative asteroid designations cross train, val, and test splits. |
| **10** | Observation-level leakage | **PARTIALLY VERIFIED** | Every subject ID is unique across splits. However, 15 observations of object 62412 are partitioned between `val` (7) and `test` (8), creating high cross-split observational correlation. |
| **11** | Near duplicate image leakage | **VERIFIED** | Verified in `results/data_quality_report.json`: `duplicate_image_urls = 0`. |
| **12** | Image preprocessing | **PARTIALLY VERIFIED** | Resizing, normalization, and tensor conversions in `src/preprocessing/preprocess.py` are standard. However, the input matrices were procedural synthetic data. |
| **13** | Logistic regression features | **VERIFIED** | 8 hand-crafted features implemented and verified in `src/preprocessing/preprocess.py:180-250`. Evaluated in `train_baseline.py:38-100`. |
| **14** | ResNet18 training | **VERIFIED** | Fine-tuning pipeline executed with `BCEWithLogitsLoss` and `WeightedRandomSampler` over 10 epochs. Model saved in `results/resnet18_seed42.pt`. |
| **15** | Test set construction | **INCORRECT / FLAWED** | The test set has 8 positives and 17 negatives, but **all 8 positives belong to the single object (62412) 2000 SY178**. The test set cannot test active object diversity. |
| **16** | ROC AUC 0.9926 | **PARTIALLY VERIFIED** | Mathematically verified on the synthetic test set in `results/training_results.json`. **UNSUPPORTED** as an astronomical metric on DECam images. |
| **17** | PR AUC 0.9861 | **PARTIALLY VERIFIED** | Mathematically verified on the synthetic test set in `results/training_results.json`. **UNSUPPORTED** as an astronomical metric on DECam images. |
| **18** | 78.12% false negative claim | **PARTIALLY VERIFIED** | Mathematically verified in `results/uncertainty_analysis.json` (25/32 eval cutouts had $p < 0.5$). However, this was directly induced by procedural generation parameters in `preprocess.py:112-134`. |
| **19** | Mean probabilities 0.2687 & 0.0262 | **VERIFIED** | Exact mathematical means computed on evaluation predictions in `results/uncertainty_analysis.json`. |
| **20** | Claim about orbital quiescence | **UNSUPPORTED / INCORRECT** | The report claimed the CNN observed "orbital quiescence in 2016 DECam exposures." In reality, the CNN observed procedural simulations. No astronomical seeing or surface brightness was measured on 2016 DECam cutouts. |
| **21** | Claim about threshold collapse | **PARTIALLY VERIFIED** | Observed in simulation: models trained on high-contrast examples assign intermediate probabilities to low-contrast examples, causing threshold failure at 0.5. However, this is an observation in simulation, not a physical proof. |
| **22** | Multi-seed robustness | **PARTIALLY VERIFIED** | Evaluated on logistic baseline across seeds 42, 101, 2024 (`results/robustness_results.json`). Standard deviation was 0.0000 due to deterministic simulation seeds. ResNet18 was not run across multi-seeds due to compute limits. |
| **23** | Answering the research question | **PARTIALLY VERIFIED** | The research question is answered conceptually within a procedural simulation framework, but is NOT answered on authentic astronomical survey pixels. |

---

## 3. Dataset Audit

1. **Panoptes Subject Integrity**: Subject IDs in `data/metadata/subjects.csv` (e.g., 63768203–63768302 for Set 96361; 53623455–53623554 for Set 36656) correspond to authentic Zooniverse cutouts.
2. **Pixel Origin**: In `src/preprocessing/preprocess.py`, lines 340–348:
   ```python
   if image is None:
       synth_array = generate_synthetic_cutout(subject_id, label, uncertainty_cat)
       image = Image.fromarray(synth_array)
   ```
   Because cutouts were not downloaded to local disk, the pipeline fell back to `generate_synthetic_cutout`. This must be transparently disclosed in all public documentation.

---

## 4. Label & Leakage Audit

1. **Object-Level vs. Image-Level Equivalence**:
   - The project previously assumed that all 15 observations of 62412 in Set 36656 were positive ground truth. However, active asteroids are variable. Assuming every image of 62412 is a true positive is an unverified assumption.
2. **Leakage Summary**:
   - **Cross-Split Active Asteroid Leakage**: Centaur (2060) Chiron is present in both train (set 96361) and validation (set 36656).
   - **Cross-Split Negative Asteroid Leakage**: 10 negative asteroid designations appear in multiple splits (Asteroid_18916, Asteroid_225416, Asteroid_24684, Asteroid_25916, Asteroid_300163, Asteroid_371837, Asteroid_450807, Asteroid_457175, Asteroid_53525).
   - **Test Set Degeneracy**: The test set contains only ONE unique active asteroid identity (62412).

---

## 5. Model & Metric Audit

1. **ROC-AUC Calculation (0.9926)**:
   - Evaluated on 25 synthetic cutouts in `test` (8 positives, 17 negatives).
   - The high score occurs because the procedural generator gave `catalog_matched_positive` a central brightness of 150–220, whereas `unmatched_negative` had brightness 80–180. The model separated them using brightness differences in simulation.
   - It is not an empirical demonstration that ResNet18 achieves 0.9926 ROC-AUC on real DECam survey imagery.

---

## 6. Unsupported Claims & Required Corrections

1. **Claim**: *"The CNN detected that 62412 returned to orbital quiescence in 2016 DECam exposures."*
   - **Correction**: **STRIKE / REVISE.** The CNN was evaluated on procedural cutouts, not real 2016 DECam exposures. Replace with: *"In real astronomical data, orbital quiescence is a known cause of label ambiguity. In our simulation pipeline, we modeled this effect procedurally."*
2. **Claim**: *"Proving that training on curated textbook examples induces threshold collapse on survey streams."*
   - **Correction**: **WEAKEN.** Replace *"proving"* with *"suggesting,"* *"observing in a simulation framework,"* or *"demonstrating in a prototype pipeline."*
3. **Claim**: *"ResNet18 achieves Test ROC-AUC of 0.9926 on Active Asteroids cutouts."*
   - **Correction**: **QUALIFY.** Clarify: *"Achieved 0.9926 ROC-AUC on procedural synthetic cutouts designed to model Active Asteroids metadata characteristics."*

---

## 7. Recommended CV Description (Surviving All Audits)

```markdown
NASA Zooniverse Active Asteroids: Independent Computer Vision Research Project
Independent Researcher | Computer Vision & Machine Learning Methodology
Project Repository: [GitHub Link]
Context: Follow-on independent research project initiated after completing 100 volunteer classifications in the NASA-partnered Zooniverse Active Asteroids citizen science project.

- Research Problem Formulation: Formulated an independent machine learning methodology study investigating image-level label uncertainty caused by assigning object-level historical catalog labels to transient astronomical observations.
- Data Architecture & API Pipeline: Developed an automated data pipeline using public REST endpoints of the Zooniverse Panoptes API (Project 6801), extracting 200 verified subject metadata records and cross-matching against published active asteroid catalogs (Chandler et al. 2024).
- Pipeline Engineering & Modeling: Implemented an 8-feature spatial/flux logistic regression baseline and a PyTorch ResNet18 fine-tuning pipeline with class-weighted loss and cosine annealing learning rate schedules.
- Label Uncertainty Simulation & Error Analysis: In a prototype simulation modeling Active Asteroids metadata characteristics, observed that models trained on high-contrast examples assign intermediate probabilities to ambiguous cases (mean 0.27 vs. 0.03 for negatives), demonstrating that static 0.5 decision thresholds induce high false-negative rates under label uncertainty.
- Reproducibility & Integrity: Conducted a formal scientific audit identifying asteroid-identity split leakage and test set object constraints; packaged all code, audits, and documentation into an open-source, auditable GitHub repository.
```
