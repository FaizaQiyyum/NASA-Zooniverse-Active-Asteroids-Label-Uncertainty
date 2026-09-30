# NASA Zooniverse Active Asteroids
# Image Download Script
#
# Downloads DECam cutout images from Zooniverse subject URLs.
# Designed for cloud or high-storage environments (NOT recommended for local).
#
# Prerequisites: Run fetch_metadata.py first to get subjects.csv
#
# Usage (recommended in Google Colab or cloud environment):
#   python src/data/download_images.py \
#     --metadata data/metadata/subjects.csv \
#     --output_dir data/images \
#     --max_images 2100 \
#     --splits train,val,test

import os
import sys
import time
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO


def download_image(url: str, save_path: str, timeout: int = 30) -> dict:
    """
    Download a single image from URL.
    Returns dict with status, file path, and image dimensions.
    """
    result = {
        "url": url,
        "path": save_path,
        "status": "unknown",
        "width": None,
        "height": None,
        "size_bytes": None,
        "error": None,
    }
    
    if not url or url == "":
        result["status"] = "no_url"
        result["error"] = "No image URL available"
        return result
    
    if os.path.exists(save_path):
        try:
            img = Image.open(save_path)
            w, h = img.size
            result["status"] = "cached"
            result["width"] = w
            result["height"] = h
            result["size_bytes"] = os.path.getsize(save_path)
            return result
        except Exception:
            os.remove(save_path)
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research; active-asteroids-ml)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        
        img = Image.open(BytesIO(resp.content))
        w, h = img.size
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        
        result["status"] = "downloaded"
        result["width"] = w
        result["height"] = h
        result["size_bytes"] = len(resp.content)
        
    except requests.exceptions.RequestException as e:
        result["status"] = "failed"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "corrupt"
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download Active Asteroids cutout images from Zooniverse"
    )
    parser.add_argument(
        "--metadata", type=str, default="data/metadata/subjects.csv",
        help="Path to subjects metadata CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/images",
        help="Directory to save downloaded images"
    )
    parser.add_argument(
        "--max_images", type=int, default=2100,
        help="Maximum images to download (default: 2100, ~full dataset)"
    )
    parser.add_argument(
        "--splits", type=str, default="train,val,test",
        help="Splits to download (comma-separated)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.2,
        help="Delay between downloads in seconds (be polite)"
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    metadata_path = project_root / args.metadata
    output_dir = project_root / args.output_dir
    
    print("NASA Zooniverse Active Asteroids - Image Downloader")
    print(f"NOTE: This downloads images from Zooniverse public servers.")
    print(f"      Recommended to run in Google Colab or cloud environment.")
    print(f"      Local storage requirement: ~{args.max_images * 0.4:.0f} MB estimated")
    print()
    
    if not metadata_path.exists():
        print(f"ERROR: Metadata not found at {metadata_path}")
        print("Run fetch_metadata.py first.")
        sys.exit(1)
    
    df = pd.read_csv(metadata_path)
    splits = [s.strip() for s in args.splits.split(",")]
    df = df[df["split"].isin(splits)]
    
    # Filter to rows with valid URLs
    df_with_url = df[df["image_url"].notna() & (df["image_url"] != "")]
    print(f"Subjects with URLs: {len(df_with_url)} / {len(df)}")
    
    # Limit to max_images
    if len(df_with_url) > args.max_images:
        df_with_url = df_with_url.sample(n=args.max_images, random_state=42)
        print(f"Sampling {args.max_images} images")
    
    results = []
    n_downloaded = 0
    n_failed = 0
    n_cached = 0
    
    for i, (_, row) in enumerate(df_with_url.iterrows()):
        subject_id = str(row["subject_id"])
        split = str(row.get("split", "train"))
        label = int(row.get("object_level_label", 0))
        
        # Organize by split and label
        label_dir = "positive" if label == 1 else "negative"
        img_dir = output_dir / split / label_dir
        img_path = str(img_dir / f"{subject_id}.png")
        
        result = download_image(row["image_url"], img_path)
        result["subject_id"] = subject_id
        result["split"] = split
        result["label"] = label
        results.append(result)
        
        if result["status"] == "downloaded":
            n_downloaded += 1
        elif result["status"] == "cached":
            n_cached += 1
        else:
            n_failed += 1
        
        if (i + 1) % 50 == 0 or i == len(df_with_url) - 1:
            print(f"Progress: {i+1}/{len(df_with_url)} | "
                  f"Downloaded: {n_downloaded} | Cached: {n_cached} | Failed: {n_failed}")
        
        time.sleep(args.delay)
    
    # Save download report
    results_df = pd.DataFrame(results)
    report_path = project_root / "results" / "download_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(report_path, index=False)
    
    print(f"\n=== Download Summary ===")
    print(f"Downloaded: {n_downloaded}")
    print(f"Cached:     {n_cached}")
    print(f"Failed:     {n_failed}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
