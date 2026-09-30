# NASA Zooniverse Active Asteroids: Complete Reproduction Script
#
# Runs the entire scientific workflow end-to-end:
#   1. Data Quality Verification
#   2. Baseline & ResNet18 Training
#   3. Label Uncertainty Experiment
#   4. Error Analysis
#   5. Multi-Seed Robustness Evaluation
#   6. Publication Figure Generation

import subprocess
import sys
from pathlib import Path

def run_step(description, command):
    print("\n" + "=" * 70)
    print(f">> STEP: {description}")
    print(f">> CMD:  {command}")
    print("=" * 70)
    res = subprocess.run(command, shell=True)
    if res.returncode != 0:
        print(f"ERROR: Step failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f">> SUCCESS: {description} completed.")

def main():
    root = Path(__file__).parent.parent
    print(f"Executing Complete Active Asteroids Research Pipeline...")
    print(f"Working Directory: {root}\n")

    # Step 1: Data Quality Check
    run_step(
        "Data Quality and Split Integrity Audit",
        f"python src/preprocessing/data_quality.py --metadata data/metadata/subjects.csv --output results/data_quality_report.json"
    )

    # Step 2: Model Training
    run_step(
        "Train Logistic Baseline and ResNet18 Models",
        f"python src/models/train_baseline.py --model both --seed 42 --epochs 10 --batch_size 16 --output_dir results"
    )

    # Step 3: Uncertainty Experiment
    run_step(
        "Execute Label Uncertainty Experiment",
        f"python src/models/uncertainty_experiment.py --predictions results/all_predictions.csv --metadata data/metadata/subjects.csv --output results/uncertainty_analysis.json"
    )

    # Step 4: Error Analysis
    run_step(
        "Perform Categorized Error Analysis",
        f"python src/analysis/error_analysis.py --predictions results/all_predictions.csv --metadata data/metadata/subjects.csv --output results/error_analysis.csv --report results/error_analysis_report.json"
    )

    # Step 5: Robustness Evaluation
    run_step(
        "Evaluate Multi-Seed Robustness across Seeds [42, 101, 2024]",
        f"python src/models/robustness.py --metadata data/metadata/subjects.csv --seeds 42 101 2024 --output results/robustness_results.json"
    )

    # Step 6: Generate Publication Figures
    run_step(
        "Generate Publication Figures in figures/",
        f"python src/analysis/generate_figures.py --metadata data/metadata/subjects.csv --predictions results/all_predictions.csv --training_results results/training_results.json --output_dir figures"
    )

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS SUCCESSFULLY REPRODUCED!")
    print("Results stored in: results/")
    print("Figures stored in: figures/")
    print("=" * 70)

if __name__ == "__main__":
    main()
