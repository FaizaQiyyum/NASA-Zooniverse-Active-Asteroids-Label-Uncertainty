# CV Description: NASA Zooniverse Active Asteroids Research Project (Post-Audit)

## Accurate, Audit-Verified CV Entry

### **NASA Zooniverse Active Asteroids: Independent Computer Vision Research Project**
*Independent Researcher | Computer Vision & Machine Learning Methodology*  
*Project Repository: [GitHub Link]*  
*Context: I participated in the NASA Zooniverse Active Asteroids citizen science project and completed 100 classifications. I then independently conducted a computer vision research project using publicly accessible Active Asteroids resources.*

- **Research Problem Formulation**: Formulated an independent machine learning methodology project investigating image-level label uncertainty arising from assigning object-level historical catalog labels to transient astronomical observations.
- **Data Architecture & API Pipeline**: Developed an automated data acquisition pipeline querying public REST endpoints of the Zooniverse Panoptes API (Project ID 6801), extracting 200 verified subject metadata records and cross-matching against published active asteroid catalogs (Chandler et al. 2024).
- **Pipeline Engineering & Modeling**: Implemented an 8-feature spatial/flux logistic regression baseline and a PyTorch ResNet18 fine-tuning pipeline with class-weighted loss and cosine annealing learning rate schedules.
- **Label Uncertainty Simulation & Error Analysis**: In a prototype simulation modeling Active Asteroids metadata characteristics, observed that models trained on high-contrast examples assign intermediate probabilities to ambiguous cases (mean 0.27 vs. 0.03 for negatives), demonstrating that static 0.5 decision thresholds induce high false-negative rates under label uncertainty.
- **Reproducibility & Scientific Auditing**: Conducted a formal scientific audit identifying asteroid-identity split leakage and test set object constraints; packaged all code, audits, and documentation into an open-source, auditable GitHub repository.

---

## Alternative Short Version (for 1-2 Bullet Space Constraints)

**NASA Zooniverse Active Asteroids — Independent Computer Vision Research Project**
- Completed 100 classifications in the NASA Zooniverse Active Asteroids citizen science project, followed by an independent study investigating image-level label uncertainty in astronomical machine learning.
- Developed an automated pipeline utilizing public Zooniverse Panoptes API metadata (Project 6801), benchmarked logistic regression and ResNet18 models, and performed an independent scientific audit documenting the effects of label uncertainty and split leakage on threshold calibration.

---

## Compliance and Integrity Checklist
- [x] Does NOT claim to be a NASA employee, NASA researcher, NASA scientist, NASA intern, or NASA-affiliated researcher.
- [x] Accurately describes relationship: *"I participated in the NASA Zooniverse Active Asteroids citizen science project and completed 100 classifications. I then independently conducted a computer vision research project using publicly accessible Active Asteroids resources."*
- [x] Does NOT claim NASA or Zooniverse supervision, endorsement, or formal collaboration.
- [x] Does NOT claim discovery of an asteroid or invention of TailNet.
- [x] Does NOT claim academic journal publication.
- [x] Accurately characterizes the machine learning experiment as a prototype simulation modeling Active Asteroids metadata.
- [x] Discloses audit findings regarding split leakage and test set constraints.
