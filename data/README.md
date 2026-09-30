# NASA Zooniverse Active Asteroids Dataset Provenance & Structure

## 1. Overview

This directory contains metadata and reference catalogs derived from public resources associated with the **NASA Zooniverse Active Asteroids** citizen science program (Zooniverse Project ID: `6801`).

Active Asteroids engages public volunteers to examine astronomical thumbnail cutouts extracted from the **Dark Energy Camera (DECam)** archive (Victor M. Blanco 4-meter Telescope at Cerro Tololo Inter-American Observatory, Chile) to search for active asteroids displaying comet-like features (tails or comae).

---

## 2. Directory Layout

```
data/
├── README.md                          # This documentation file
├── catalog/
│   └── active_objects_catalog.csv    # 25-entry catalog of verified active small bodies
└── metadata/
    ├── subjects.csv                  # Experimental subject metadata and split assignments (n=200)
    └── subjects_full.csv             # Full metadata including raw Zooniverse Panoptes API JSON payloads
```

---

## 3. Data Sources & Scientific Provenance

All data in this repository are derived strictly from publicly accessible sources without authentication bypass or private data usage:

1. **Zooniverse Panoptes API**:
   - Endpoint: `https://www.zooniverse.org/api/subjects`
   - Active Asteroids Project ID: `6801`
   - Subject sets utilized:
     - `96361`: **CSB00 Training Set 1** (curated teaching cutouts with verified active comets/centaurs)
     - `36656`: **General Classification Stream** (archival DECam observations of known minor planets)
2. **Published Astronomical Catalog**:
   - Source: *The Astronomical Journal*, 167:156 (Chandler et al. 2024, arXiv:2403.09768)
   - Contains 25 verified active small solar system bodies, including Centaur (2060) Chiron, Main-Belt Comet (62412) 2000 SY178, (6478) Gault, (7968) Elst-Pizarro, (10199) Chariklo, and (60558) Echeclus.

---

## 4. Metadata Fields in `subjects.csv`

| Column | Type | Description |
|---|---|---|
| `subject_id` | integer/string | Unique Zooniverse subject identifier |
| `subject_set_id` | integer/string | Zooniverse subject set ID (`96361` = teaching set; `36656` = general stream) |
| `designation` | string | Minor planet designation or comet name (e.g., `62412 (2000 SY178)`, `2060 Chiron`, `Asteroid_18916`) |
| `filename` | string | DECam cutout filename encoding MPC number, timestamp, filter, and exposure |
| `kind` | string | Object dynamical class if cataloged (e.g., `Comet`, `Centaur`) |
| `image_url` | string | Public URL hosted on Panoptes AWS S3 storage |
| `object_level_label` | integer | Binary label: `1` = Historically active object; `0` = No catalog match |
| `label_source` | string | Provenance of label: `train_set_teaching`, `catalog_match`, or `no_match` |
| `label_confidence` | string | Qualitative confidence: `high` (curated teaching), `medium` (catalog match), `low` (unmatched) |
| `uncertainty_category` | string | Stratification: `teaching_positive`, `catalog_matched_positive`, `unmatched_negative` |
| `mpc_number` | string | Extracted Minor Planet Center object number |
| `split` | string | Partition assignment: `train` (n=150), `val` (n=25), `test` (n=25) |

---

## 5. Distinction Between Object-Level Status and Image-Level Evidence

A central methodological contribution of this research is the explicit recognition of **label uncertainty**:
- **Object-Level Active (Catalog)**: An asteroid has been observed to display activity at some epoch in its orbit.
- **Image-Level Active (Observation)**: A specific photographic cutout displaying detectable dust emission or coma.

Because active asteroids may only be active near perihelion or during stochastic outburst/impact events, observations taken during quiescent phases will appear purely asteroidal (point-source-like). Assigning an object-level positive label to a quiescent cutout injects label noise. Conversely, the "unmatched negative" pool contains minor planets not currently cataloged as active, some of which may harbor faint uncataloged activity.

---

## 6. How to Download Raw Cutouts (Cloud / High-Compute Environments)

To avoid downloading large volumes to local disks, run the download script in Google Colab or cloud VMs:

```bash
python src/data/download_images.py \
  --metadata data/metadata/subjects.csv \
  --output_dir data/images \
  --max_images 200
```
