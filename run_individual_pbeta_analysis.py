import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import POMM as pomm_lib
except ImportError:
    print("Error: Could not import 'POMM.py'.")
    print("Please ensure 'POMM.py' and 'libPOMM.so' are in the same folder as this script.")
    exit()

# --- Configuration ---
INDIVIDUAL_MATRICES_DIR = '/Volumes/Storage/Summer_Studentship/P18 Data clean/ind_matrix_from_seq'
SEQUENCE_FILES_DIR = '/Volumes/Storage/Summer_Studentship/P18 Data clean/bout_analysis'
RESULTS_DIR = '/Volumes/Storage/Summer_Studentship/P18 Data clean/pb_ind'

# --- IMPORTANT ---
# Set to True if you want a plot for every comparison.
# WARNING: This can generate thousands of files (e.g., 38 models x 38 sequences = 1444 plots).
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
    Main function to run the paired P-beta analysis.
    """
    dist_dir = os.path.join(RESULTS_DIR, 'distributions_and_plots')
    os.makedirs(dist_dir, exist_ok=True)

    S_model = np.array([0, -1, 1, 2, 3, 4, 5], dtype=np.int32)
    
    # Get a list of all sequence files, which will drive the analysis
    all_sequence_files = [f for f in os.listdir(SEQUENCE_FILES_DIR) if f.endswith('.txt') and not f.startswith('._')]
    
    results_summary = []

    print(f"\n{'='*50}")
    print(f"Starting Paired Analysis for {len(all_sequence_files)} animals.")
    print(f"{'='*50}")

    # --- MODIFIED: Single loop for paired analysis ---
    for seq_filename in all_sequence_files:
        base_name = os.path.splitext(seq_filename)[0]
        print(f"\n-> Processing pair: {base_name}")

        # 1. Define and load the corresponding matrix file for the current sequence file
        model_filename = f"{base_name}_matrix.csv"
        matrix_path = os.path.join(INDIVIDUAL_MATRICES_DIR, model_filename)
        
        try:
            P_model_df = pd.read_csv(matrix_path, index_col=0)
            
            ordered_states = ['Start', 'End', '1', '2', '3', '4', '5']
            P_model_df.columns = P_model_df.columns.astype(str)
            P_model_df.index = P_model_df.index.astype(str)
            P_model = P_model_df.loc[ordered_states, ordered_states].to_numpy()
            
            row_sums = P_model.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            P_model = P_model / row_sums

        except (FileNotFoundError, KeyError) as e:
            print(f"   Warning: Could not find or parse matching matrix '{model_filename}'. Skipping pair. Error: {e}")
            continue

        # 2. Load the observed sequences for the current animal
        seq_path = os.path.join(SEQUENCE_FILES_DIR, seq_filename)
        observed_sequences = load_sequences_as_numeric(seq_path)

        if not observed_sequences:
            print(f"   ...skipping due to empty sequence file.")
            continue

        # 3. Run the P-beta analysis for the pair
        p_value, p_beta_dist, p_beta_observed = pomm_lib.getPVSampledSeqsPOMM(
            S_model, P_model, observed_sequences, nSample=N_SAMPLES, nProc=N_PROC
        )

        results_summary.append({
            'animal_id': base_name,
            'p_beta_observed': p_beta_observed,
            'p_value': p_value,
            'n_sequences': len(observed_sequences)
        })

        if GENERATE_PLOTS:
            plot_title = f"Pβ Distribution: '{base_name}'\n(Model vs. Own Data)"
            plot_path = os.path.join(dist_dir, f"{base_name}_plot.png")
            plot_pbeta_distribution(
                p_beta_dist, p_beta_observed, p_value,
                title=plot_title,
                output_path=plot_path
            )

    # --- Save final summary ---
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(RESULTS_DIR, 'paired_pbeta_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*50}")
    print(f"Analysis complete! Paired results saved to:\n{summary_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    print("Reminder: Ensure you have edited your 'POMM.py' file to use the pure-Python fallback functions.")
    main()
