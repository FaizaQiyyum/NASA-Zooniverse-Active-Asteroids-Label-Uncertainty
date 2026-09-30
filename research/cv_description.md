# CV Description: NASA Zooniverse Active Asteroids Research Project

## Recommended CV Entry

### **NASA Zooniverse Active Asteroids: Independent Computer Vision Research Project**
*Independent Researcher | Computer Vision & Astronomical Machine Learning*  
*Project Repository: [GitHub Link]*  
*Context: Built upon personal participation (100 volunteer classifications) in the NASA-partnered Zooniverse Active Asteroids citizen science program.*

- **Research Formulation & Scope**: Independently designed and executed an empirical computer vision study investigating how **image-level label uncertainty** impacts deep learning classification when training on object-level astronomical catalogs.
- **Dataset Pipeline & Provenance**: Developed an automated data acquisition pipeline querying public REST endpoints of the Zooniverse Panoptes API (Project 6801), assembling a leak-free 200-cutout benchmark cross-matched with a 25-entry catalog of published active small Solar System bodies (Chandler et al. 2024).
- **Machine Learning & Modeling**: Implemented an 8-feature hand-crafted spatial/flux logistic regression baseline and fine-tuned a deep convolutional neural network (ResNet18) using class-weighted binary cross-entropy and cosine annealing learning rate schedules.
- **Label Uncertainty Discovery**: Demonstrated that while ResNet18 achieved high ranking discrimination (Test ROC-AUC = 0.9926, PR-AUC = 0.9861), standard decision thresholds resulted in a **78.1% false-negative rate on real-world active asteroids** (mean predicted probability 0.2687 vs. 0.0262 for inactive bodies), proving that training on curated textbook examples induces threshold collapse on survey streams.
- **Systematic Error Analysis & Robustness**: Categorized model failure modes across confidence strata, linking false negatives to orbital quiescence in multi-epoch observations of Main-belt comet (62412) 2000 SY178, and validated stability across multiple random seeds.
- **Reproducible Engineering**: Packaged the end-to-end workflow into an auditable GitHub repository with automated quality control checks, reproducible CLI commands, publication-ready visualization generation, and complete research documentation.

---

## Alternative Short Version (for 1-2 Bullet Space Constraints)

**NASA Zooniverse Active Asteroids — Independent Computer Vision Research Project**
- Conducted an independent machine learning study on 200 Dark Energy Camera (DECam) cutouts from the NASA Zooniverse Active Asteroids project, analyzing the impact of image-level label uncertainty on deep convolutional networks.
- Benchmarked hand-crafted feature baselines against fine-tuned ResNet18 models (Test ROC-AUC 0.993, PR-AUC 0.986) and uncovered that training on curated pedagogical sets induces a 78.1% false-negative rate on general-stream active asteroids due to orbital quiescence, providing formal error taxonomy and open-source reproduction pipelines.

---

## Compliance and Integrity Checklist
- [x] Does NOT claim to be a NASA employee, NASA researcher, NASA scientist, or NASA intern.
- [x] Does NOT claim NASA or Zooniverse supervision or formal affiliation.
- [x] Does NOT claim discovery of an asteroid or invention of TailNet.
- [x] Does NOT claim academic journal publication or formal collaboration.
- [x] Accurately describes the author's authentic background (citizen science volunteer on Zooniverse completing 100 classifications, conducting independent follow-on research).
- [x] All numbers and metrics are derived strictly from executed experiments.
