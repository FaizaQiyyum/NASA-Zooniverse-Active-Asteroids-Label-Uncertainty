# NASA Zooniverse Active Asteroids
# Re-parse Metadata with Full Filename & Subject Metadata Parsing

import re
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
full_meta_path = project_root / "data/metadata/subjects_full.csv"
catalog_path = project_root / "data/catalog/active_objects_catalog.csv"
output_path = project_root / "data/metadata/subjects.csv"

# Load catalog
cat_df = pd.read_csv(catalog_path)
known_numbers = {
    "2060": "2060 Chiron",
    "6478": "6478 Gault",
    "7968": "7968 Elst-Pizarro (133P)",
    "10199": "10199 Chariklo",
    "62412": "62412 (2000 SY178)",
    "248370": "248370 (433P)",
    "323137": "323137 (282P)",
    "588045": "588045 (2007 FZ18)",
    "60558": "60558 Echeclus (174P)",
}

df_full = pd.read_csv(full_meta_path)

records = []
for idx, row in df_full.iterrows():
    raw = json.loads(row["raw_metadata"])
    sid = str(row["subject_id"])
    set_id = str(row["subject_set_id"])
    img_url = str(row["image_url"])
    
    filename = raw.get("Thumbname", raw.get("filename", ""))
    name = raw.get("#Name", raw.get("name", ""))
    kind = raw.get("#Kind", "")
    feedback = str(raw.get("#feedback_1_answer", ""))
    
    # Parse asteroid number from filename
    # E.g. "0062412A_20140328-..." -> "62412"
    # E.g. "0002060U_20150921-..." -> "2060"
    num_match = re.match(r"^0*(\d+)", filename)
    mpc_num = num_match.group(1) if num_match else ""
    
    # Determine label, source, confidence, uncertainty category
    if set_id == "96361":
        # Teaching set: curated active comet/asteroid cutouts
        label = 1
        source = "train_set_teaching"
        conf = "high"
        unc_cat = "teaching_positive"
        designation = name if name else (known_numbers.get(mpc_num, mpc_num))
    elif mpc_num in known_numbers:
        # General stream, matches known active asteroid catalog!
        label = 1
        source = "catalog_match"
        conf = "medium"
        unc_cat = "catalog_matched_positive"
        designation = known_numbers[mpc_num]
    else:
        # General stream, inactive asteroid
        label = 0
        source = "no_match"
        conf = "low"
        unc_cat = "unmatched_negative"
        designation = f"Asteroid_{mpc_num}" if mpc_num else "Unknown_Inactive"
        
    records.append({
        "subject_id": sid,
        "subject_set_id": set_id,
        "designation": designation,
        "filename": filename,
        "kind": kind,
        "image_url": img_url,
        "feedback_answer": feedback,
        "object_level_label": label,
        "label_source": source,
        "label_confidence": conf,
        "uncertainty_category": unc_cat,
        "mpc_number": mpc_num,
    })

df_parsed = pd.DataFrame(records)

# Create train/val/test splits:
# - Teaching positives -> train
# - Catalog matched positives (16 total: 1 Chiron, 15 of 62412) -> split 50/50 val/test (8 val, 8 test)
# - Negatives (84 total) -> 60% train (50), 20% val (17), 20% test (17)
np.random.seed(42)
splits = []

cat_pos_indices = df_parsed[df_parsed["label_source"] == "catalog_match"].index.tolist()
np.random.shuffle(cat_pos_indices)
val_cat_pos = set(cat_pos_indices[:len(cat_pos_indices)//2])
test_cat_pos = set(cat_pos_indices[len(cat_pos_indices)//2:])

neg_indices = df_parsed[df_parsed["label_source"] == "no_match"].index.tolist()
np.random.shuffle(neg_indices)
n_neg = len(neg_indices)
train_neg = set(neg_indices[:int(0.6 * n_neg)])
val_neg = set(neg_indices[int(0.6 * n_neg):int(0.8 * n_neg)])
test_neg = set(neg_indices[int(0.8 * n_neg):])

for idx in range(len(df_parsed)):
    source = df_parsed.iloc[idx]["label_source"]
    if source == "train_set_teaching":
        splits.append("train")
    elif idx in val_cat_pos:
        splits.append("val")
    elif idx in test_cat_pos:
        splits.append("test")
    elif idx in train_neg:
        splits.append("train")
    elif idx in val_neg:
        splits.append("val")
    elif idx in test_neg:
        splits.append("test")
    else:
        splits.append("train")

df_parsed["split"] = splits

df_parsed.to_csv(output_path, index=False)
print("Updated subjects.csv saved.")
print("\n--- Summary ---")
print(df_parsed.groupby(["split", "object_level_label", "uncertainty_category"]).size())
