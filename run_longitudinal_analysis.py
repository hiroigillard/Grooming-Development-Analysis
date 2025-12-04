import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

try:
    import POMM as pomm_lib
except ImportError:
    print("Error: Could not import 'POMM.py'.")
    print("Please ensure 'POMM.py' and 'libPOMM.so' are in the same folder as this script.")
    exit()

# --- Configuration ---
INDIVIDUAL_MATRICES_DIR = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/Longitudinal_PBeta/individual_matrix'
SEQUENCE_FILES_DIR = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/Longitudinal_PBeta/Seqs'
RESULTS_DIR = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/Longitudinal_PBeta/results' # New output folder for this analysis

# Set to True if you want a plot for every comparison.
GENERATE_PLOTS = True

N_SAMPLES = 10000
N_PROC = 4

def load_sequences_as_numeric(file_path):
    """Reads a .txt file and converts sequences to the required numeric format."""
    sequences = []
    try:
        with open(file_path, 'r', encoding='latin1') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    sequences.append([np.int32(char) for char in stripped_line])
    except FileNotFoundError:
        print(f"Warning: Sequence file not found at {file_path}")
        return None
    return sequences

def plot_pbeta_distribution(dist, observed_pbeta, p_value, title, output_path):
    """Generates and saves a histogram of the P-beta distribution."""
    plt.figure(figsize=(8, 6))
    sns.histplot(dist, bins=50, color='gray', stat='density')
    plt.axvline(observed_pbeta, color='red', linestyle='--', linewidth=2,
                label=f'Observed Pβ = {observed_pbeta:.3f}')
    plt.title(title, fontsize=14, wrap=True)
    plt.xlabel('P-beta (Pβ)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.text(0.05, 0.95, f'p-value = {p_value:.4f}', transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """
    Main function to run the longitudinal P-beta analysis.
    """
    dist_dir = os.path.join(RESULTS_DIR, 'distributions_and_plots')
    os.makedirs(dist_dir, exist_ok=True)

    S_model = np.array([0, -1, 1, 2, 3, 4, 5], dtype=np.int32)
    
    # --- Step 1: Discover all animals and organize their files by age ---
    animal_data = defaultdict(lambda: {'sequences': {}, 'matrices': {}})
    # Regex to capture Group, Age (digits), and Mouse ID from filenames
    # Example: "Rop_P18_Mouse13_clean_sequences.txt" -> ('Rop', '18', 'Mouse13')
    pattern = re.compile(r"^(Veh|SKF|Rop)_P(\d+)_([^_]+)_.*")

    # Scan sequence files
    for filename in os.listdir(SEQUENCE_FILES_DIR):
        if filename.endswith('.txt') and not filename.startswith('._'):
            match = pattern.match(filename)
            if match:
                group, age, mouse_id = match.groups()
                animal_id = f"{group}_{mouse_id}"
                animal_data[animal_id]['sequences'][int(age)] = os.path.join(SEQUENCE_FILES_DIR, filename)

    # Scan matrix files
    for filename in os.listdir(INDIVIDUAL_MATRICES_DIR):
        if filename.endswith('.csv') and not filename.startswith('._'):
            match = pattern.match(filename)
            if match:
                group, age, mouse_id = match.groups()
                animal_id = f"{group}_{mouse_id}"
                animal_data[animal_id]['matrices'][int(age)] = os.path.join(INDIVIDUAL_MATRICES_DIR, filename)

    print(f"Discovered data for {len(animal_data)} unique animals.")
    
    results_summary = []

    # --- Step 2: Loop through each animal for longitudinal analysis ---
    for animal_id, data in animal_data.items():
        print(f"\n{'='*50}")
        print(f"Analyzing Animal: {animal_id}")
        
        # 1. Find and load the P21 matrix to use as the reference model
        if 21 not in data['matrices']:
            print(f"  -> Warning: P21 matrix not found for {animal_id}. Skipping animal.")
            continue
        
        matrix_path = data['matrices'][21]
        print(f"  -> Using P21 matrix as model: {os.path.basename(matrix_path)}")
        try:
            P_model_df = pd.read_csv(matrix_path, index_col=0)
            ordered_states = ['Start', 'End', '1', '2', '3', '4', '5']
            P_model_df.columns = P_model_df.columns.astype(str)
            P_model_df.index = P_model_df.index.astype(str)
            P_model = P_model_df.loc[ordered_states, ordered_states].to_numpy()
            
            row_sums = P_model.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P_model = P_model / row_sums
        except Exception as e:
            print(f"  -> Error loading or processing P21 matrix for {animal_id}. Error: {e}. Skipping.")
            continue

        # 2. Test the P21 model against sequences from all available ages for this animal
        for age, seq_path in sorted(data['sequences'].items()):
            print(f"    -> Testing sequences from age P{age}...")
            observed_sequences = load_sequences_as_numeric(seq_path)

            if not observed_sequences:
                print(f"       ...skipping P{age} due to empty sequence file.")
                continue

            p_value, p_beta_dist, p_beta_observed = pomm_lib.getPVSampledSeqsPOMM(
                S_model, P_model, observed_sequences, nSample=N_SAMPLES, nProc=N_PROC
            )

            results_summary.append({
                'animal_id': animal_id,
                'age': age,
                'p_beta_observed': p_beta_observed,
                'p_value': p_value,
                'n_sequences': len(observed_sequences)
            })

            if GENERATE_PLOTS:
                base_name = f"{animal_id}_P21model_vs_P{age}seq"
                plot_title = f"Model: {animal_id} (P21) vs Data: P{age}"
                plot_path = os.path.join(dist_dir, f"{base_name}_plot.png")
                plot_pbeta_distribution(
                    p_beta_dist, p_beta_observed, p_value,
                    title=plot_title,
                    output_path=plot_path
                )

    # --- Save final summary ---
    summary_df = pd.DataFrame(results_summary)
    summary_df = summary_df.sort_values(by=['animal_id', 'age']).reset_index(drop=True)
    summary_path = os.path.join(RESULTS_DIR, 'longitudinal_pbeta_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*50}")
    print(f"Analysis complete! Longitudinal results saved to:\n{summary_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("Reminder: Ensure you have edited your 'POMM.py' file to use the pure-Python fallback functions.")
    main()
