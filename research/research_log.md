# Active Asteroids: Research Log

## Scientific Audit Trail & Experiment History

This log documents every experimental step, hypothesis, unexpected finding, failure mode, and technical decision across the lifecycle of the project.

---

### Entry 1: Project Initialization & Scope Definition
- **Date**: 2026-09-30
- **Task**: Define research boundaries, ethical guidelines, and experimental objectives.
- **Reason**: Convert personal experience (100 volunteer classifications in NASA Zooniverse Active Asteroids) into an independent, rigorous, reproducible computer vision research project.
- **Action**: Formulated primary and secondary research questions focused on **image-level label uncertainty** resulting from applying object-level catalog labels to time-series astronomical cutouts. Established strict integrity constraints: no NASA affiliation claims, no fabricated labels or citations, and complete public data provenance.
- **Result**: Core implementation plan formalized across 18 distinct research phases.
- **Problem**: Need to ensure absolute scientific honesty regarding personal role (independent citizen science volunteer, not NASA employee/intern).
- **Decision**: Clearly document dataset origin as the public NASA Zooniverse Active Asteroids project while explicitly describing author role as independent researcher.
- **Next Step**: Conduct systematic literature review and reference code audit.

---

### Entry 2: Literature Review & TailNet Investigation
- **Date**: 2026-09-30
- **Task**: Survey astronomical publications, public baseline code, and computer vision literature on noisy labels and label uncertainty.
- **Reason**: Contextualize our contribution against official Active Asteroids publications (Chandler et al. 2024, arXiv:2403.09768) and existing machine learning baselines (`ZamirDev/active-asteroids-ml`).
- **Action**: Performed literature searches and text retrieval on primary discovery papers, TailNet descriptions, and deep learning noisy-label surveys (Song et al. 2022, Northcutt et al. 2021).
- **Result**: Created `research/literature_review.md` and compiled `data/catalog/active_objects_catalog.csv` containing 25 verified active asteroids and Centaurs (e.g., 2060 Chiron, 62412, 6478 Gault, 10199 Chariklo).
- **Problem**: "TailNet" is an internal convolutional network tool referenced in project communications rather than a widely published standalone Python library.
- **Decision**: Benchmark against the published public baseline `ZamirDev/active-asteroids-ml` (logistic regression + ResNet18) and ground scientific context in Chandler et al. (2024).
- **Next Step**: Analyze baseline dataset construction and verify API reproducibility.

---

### Entry 3: Panoptes API Data Acquisition & Network Interruption
- **Date**: 2026-09-30
- **Task**: Query Zooniverse Panoptes public API endpoint (`https://www.zooniverse.org/api/subjects`) for Active Asteroids project ID `6801`.
- **Reason**: Obtain authentic cutouts from teaching sets (`96361`) and general stream subject sets (`36656`, `96360`, etc.).
- **Action**: Executed `src/data/fetch_metadata.py` targeting subject sets across multiple pages.
- **Result**: Successfully acquired 200 authentic subjects (100 from set `96361` and 100 from set `36656`) with complete S3 image URLs and JSON metadata.
- **Problem (FAILURE/CHALLENGE)**: Remote server closed connections (`SSLEOFError` / `ConnectionResetError 10054`) on subsequent pages, preventing full download of thousands of subjects.
- **Decision**: Work with the verified 200-subject sample obtained directly from the public Panoptes API rather than abandoning reproducibility or fabricating data.
- **Next Step**: Audit metadata and construct contamination-free splits.

---

### Entry 4: Metadata Parsing Discovery & Split Construction
- **Date**: 2026-09-30
- **Task**: Inspect retrieved Panoptes metadata and cross-match with active objects catalog.
- **Reason**: Assign ground-truth labels and verify whether cataloged active asteroids appear in the general stream.
- **Action**: Analyzed filenames and raw JSON payloads. Discovered that set `36656` cutouts encode MPC minor planet numbers in the filename prefix (e.g., `0002060U` -> 2060 Chiron, `0062412A` -> 62412 2000 SY178).
- **Result**: Uncovered 16 authentic catalog-matched active asteroid observations in set `36656` (1 observation of Chiron and 15 multi-epoch observations of Main-belt comet 62412). The remaining 84 subjects represent inactive asteroids (e.g., 65 Cybele, 704 Interamnia, 3552 Don Quixote).
- **Problem**: An initial naive split assigned all teaching positives to train, leaving validation and test sets with zero positives (degenerate evaluation metrics).
- **Decision**: Implemented `src/data/build_clean_splits.py` to create a stratified, leak-free partition:
  - Train: 100 teaching positives + 50 unmatched negatives (n=150)
  - Val: 8 catalog-matched positives + 17 unmatched negatives (n=25, 32% positive rate)
  - Test: 8 catalog-matched positives + 17 unmatched negatives (n=25, 32% positive rate)
- **Key Scientific Insight**: This cleanly models the real-world problem: the model trains on curated teaching cutouts and must generalize to general stream observations where active asteroids exhibit multi-epoch label uncertainty.
- **Next Step**: Execute automated data quality audit and train baseline models.

---

### Entry 5: Automated Data Quality Audit
- **Date**: 2026-09-30
- **Task**: Run comprehensive data quality and split integrity checks.
- **Reason**: Verify absence of duplicate subjects, split leakage, and corrupted fields before training.
- **Action**: Executed `src/preprocessing/data_quality.py`.
- **Result**: Produced `results/data_quality_report.json`. Confirmed 0 duplicate subject IDs, 0 duplicate image URLs, 0 train/val/test ID overlaps (`leakage_detected = false`), and exactly 16 catalog-matched uncertain positives.
- **Next Step**: Train logistic regression and fine-tune ResNet18.

---

### Entry 6: Model Training & Benchmark Execution
- **Date**: 2026-09-30
- **Task**: Train logistic regression baseline on 8 hand-crafted features and fine-tune ResNet18 on cutout representations.
- **Reason**: Establish baseline performance and evaluate CNN representation capacity.
- **Action**: Ran `src/models/train_baseline.py` with seed=42, BCEWithLogitsLoss (class-weighted), and CosineAnnealingLR.
- **Result**: Logistic regression baseline completed in 4.0s: Val ROC-AUC 0.875, Test ROC-AUC 0.956. ResNet18 executed across 10 epochs.
- **Next Step**: Evaluate label uncertainty experiments (Experiments B, C, D) and perform systematic error analysis.
