"""
run_all.py

Runs the full pipeline end-to-end:
  1. Simulate the synthetic GLP-1 patient population -> data/raw/
  2. GMM segmentation (BIC model selection + stability check) -> data/processed/
  3. Cox proportional hazards survival analysis -> output/
  4. Figures -> output/figures/

Run with:  python run_all.py
"""

import simulate_patient_data
import segment_patients
import survival_analysis
import make_figures


def main():
    print("--- Simulating synthetic GLP-1 patient population ---")
    simulate_patient_data.main()

    print("\n--- GMM segmentation ---")
    segment_patients.main()

    print("\n--- Cox proportional hazards survival analysis ---")
    survival_analysis.main()

    print("\n--- Generating figures ---")
    make_figures.main()


if __name__ == "__main__":
    main()
