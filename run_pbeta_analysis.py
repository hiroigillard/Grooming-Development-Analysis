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

GROUP_PREFIXES = ['Veh', 'SKF', 'Rop']
MEDIAN_MATRICES_DIR = '/Volumes/Storage/Summer_Studentship/P18 Data clean/median_matrix'
SEQUENCE_FILES_DIR = '/Volumes/Storage/Summer_Studentship/P18 Data clean/bout_analysis'
RESULTS_DIR = '/Volumes/Storage/Summer_Studentship/P18 Data clean/PBeta_Group'

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
    plt.title(title, fontsize=14)
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
    Main function to run the entire P-beta analysis pipeline.
    """
    dist_dir = os.path.join(RESULTS_DIR, 'distributions_and_plots')
    os.makedirs(dist_dir, exist_ok=True)

    S_model = np.array([0, -1, 1, 2, 3, 4, 5], dtype=np.int32)
    all_sequence_files = [f for f in os.listdir(SEQUENCE_FILES_DIR) if f.endswith('.txt') and not f.startswith('._')]
    
    results_summary = []

    for model_prefix in GROUP_PREFIXES:
        print(f"\n{'='*50}")
        print(f"MODEL: Using '{model_prefix}' group's median transition matrix.")
        print(f"{'='*50}")

        matrix_path = os.path.join(MEDIAN_MATRICES_DIR, f'Group_{model_prefix}_median_matrix.csv')
        try:
            P_model_df = pd.read_csv(matrix_path, index_col=0)
            
            # --- THIS IS THE CORRECTED LOGIC ---
            # The order of rows/columns MUST match the order in the S_model vector: [Start, End, 1, 2, 3, 4, 5]
            ordered_states = ['Start', 'End', '1', '2', '3', '4', '5']
            P_model_df.columns = P_model_df.columns.astype(str)
            P_model_df.index = P_model_df.index.astype(str)
            P_model = P_model_df.loc[ordered_states, ordered_states].to_numpy()
            
            # Re-normalize the matrix to ensure rows sum exactly to 1.0
            row_sums = P_model.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1 # Avoid division by zero for rows that are all zeros
            P_model = P_model / row_sums

        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not find or parse matrix file for '{model_prefix}'. Error: {e}. Skipping.")
            continue

        print(f"\n--- Testing individual animal sequences against '{model_prefix}' model ---")
        for seq_filename in all_sequence_files:
            print(f"  -> Testing: {seq_filename}")
            seq_path = os.path.join(SEQUENCE_FILES_DIR, seq_filename)
            observed_sequences = load_sequences_as_numeric(seq_path)

            if not observed_sequences:
                print(f"     ...skipping due to empty sequence file.")
                continue

            p_value, p_beta_dist, p_beta_observed = pomm_lib.getPVSampledSeqsPOMM(
                S_model, P_model, observed_sequences, nSample=N_SAMPLES, nProc=N_PROC
            )

            results_summary.append({
                'model_group': model_prefix,
                'test_data': seq_filename,
                'p_beta_observed': p_beta_observed,
                'p_value': p_value,
                'n_sequences': len(observed_sequences)
            })

            base_name = f"{model_prefix}_model_vs_{os.path.splitext(seq_filename)[0]}"
            np.save(os.path.join(dist_dir, f"{base_name}_pbeta_dist.npy"), p_beta_dist)
            plot_pbeta_distribution(
                p_beta_dist, p_beta_observed, p_value,
                title=f"Pβ Distribution: '{model_prefix}' Model vs '{seq_filename}' Data",
                output_path=os.path.join(dist_dir, f"{base_name}_plot.png")
            )

        print(f"\n--- Testing combined '{model_prefix}' group sequences against '{model_prefix}' model ---")
        group_seq_files = [f for f in all_sequence_files if f.startswith(f"{model_prefix}_")]
        
        if not group_seq_files:
            print(f"  No sequence files found for group '{model_prefix}'. Skipping group analysis.")
            continue
            
        combined_sequences = []
        for seq_filename in group_seq_files:
            seq_path = os.path.join(SEQUENCE_FILES_DIR, seq_filename)
            combined_sequences.extend(load_sequences_as_numeric(seq_path))
        
        print(f"  Loaded {len(combined_sequences)} sequences from {len(group_seq_files)} files for group '{model_prefix}'.")

        p_value, p_beta_dist, p_beta_observed = pomm_lib.getPVSampledSeqsPOMM(
            S_model, P_model, combined_sequences, nSample=N_SAMPLES, nProc=N_PROC
        )

        results_summary.append({
            'model_group': model_prefix,
            'test_data': f'{model_prefix}_GROUP_COMBINED',
            'p_beta_observed': p_beta_observed,
            'p_value': p_value,
            'n_sequences': len(combined_sequences)
        })

        base_name = f"{model_prefix}_model_vs_{model_prefix}_GROUP"
        np.save(os.path.join(dist_dir, f"{base_name}_pbeta_dist.npy"), p_beta_dist)
        plot_pbeta_distribution(
            p_beta_dist, p_beta_observed, p_value,
            title=f"Pβ Distribution: '{model_prefix}' Model vs Combined '{model_prefix}' Data",
            output_path=os.path.join(dist_dir, f"{base_name}_plot.png")
        )

    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(RESULTS_DIR, 'pbeta_analysis_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*50}")
    print(f"Analysis complete! Summary of results saved to:\n{summary_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
