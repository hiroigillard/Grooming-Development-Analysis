import os
import pandas as pd
import numpy as np
import re
from scipy.stats import ks_2samp


## -------------------------------- ##
## --- DATA HELPER FUNCTIONS ---
## -------------------------------- ##

def generate_sequence(matrix_df, max_length=500):
    """Generates a single sequence based on a transition probability matrix."""
    states = matrix_df.columns.tolist()
    sequence, current_state = [], 'Start'
    while current_state != 'End' and len(sequence) < max_length:
        # Ensure probabilities sum to 1 to avoid np.random.choice errors
        probabilities = matrix_df.loc[current_state].values
        prob_sum = np.sum(probabilities)
        if not np.isclose(prob_sum, 1.0):
            probabilities /= prob_sum  # Re-normalize the row if needed

        next_state = np.random.choice(states, p=probabilities)
        if next_state == 'End':
            current_state = 'End'
            continue
        sequence.append(next_state)
        current_state = next_state
    return sequence


def calculate_metrics_from_sequences(sequences):
    """Calculates metrics for a list of sequences."""
    if not sequences:
        return pd.DataFrame({'length': [], 'unique_phases': []})
    lengths = [len(s) for s in sequences]
    unique_phases = [len(set(s)) for s in sequences]
    return pd.DataFrame({'length': lengths, 'unique_phases': unique_phases})


def get_longitudinal_id(filename):
    """
    Extracts a consistent animal ID, ignoring age (e.g., P21).
    Example: 'Rop_P21_Mouse13_...' -> 'Rop_Mouse13'
    """
    match = re.search(r'(Veh|SKF|Rop)_.*?_(Mouse\d+)', filename)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    return "Unknown_Animal"


## -------------------------------- ##
## --- MAIN EXECUTION BLOCK ---
## -------------------------------- ##

def main():
    """
    Main function to validate each individual animal's matrix against its
    own sequence data.
    """
    # --- Configuration ---
    individual_matrix_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/self sequences labels/P21/individual_matrix'  # Folder with individual CSV matrices
    real_sequence_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/self sequences labels/P21/bout_analysis'  # Folder with original TXT sequences
    output_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/self sequences labels/P21/individual_comparison'  # Folder for the results

    os.makedirs(output_dir, exist_ok=True)

    num_sequences_to_generate = 1000
    all_results = []

    # Find all individual matrix files
    matrix_files = [f for f in os.listdir(individual_matrix_dir) if f.endswith("_matrix.csv")]

    if not matrix_files:
        print(f"Error: No matrix files found in '{individual_matrix_dir}'. Please check the path.")
        return

    print(f"Found {len(matrix_files)} individual matrices to process...")

    for matrix_filename in matrix_files:
        print(f"\n--- Processing: {matrix_filename} ---")

        # 1. Match matrix file to sequence file
        base_name = matrix_filename.replace('_matrix.csv', '')
        sequence_filename = f"{base_name}.txt"
        sequence_filepath = os.path.join(real_sequence_dir, sequence_filename)

        if not os.path.exists(sequence_filepath):
            print(f"  Warning: Corresponding sequence file not found. Skipping.")
            continue

        # 2. Load data
        matrix_filepath = os.path.join(individual_matrix_dir, matrix_filename)
        animal_matrix_df = pd.read_csv(matrix_filepath, index_col=0)

        with open(sequence_filepath, 'r', encoding='latin1') as f:
            real_sequences = [list(line.strip()) for line in f if line.strip()]

        if not real_sequences:
            print(f"  Warning: Sequence file is empty. Skipping.")
            continue

        # 3. Generate sequences from this animal's model
        print(f"  Generating {num_sequences_to_generate} sequences from this matrix...")
        model_sequences = [generate_sequence(animal_matrix_df) for _ in range(num_sequences_to_generate)]

        # 4. Calculate metrics for both distributions
        model_metrics = calculate_metrics_from_sequences(model_sequences)
        real_metrics = calculate_metrics_from_sequences(real_sequences)

        # 5. Perform K-S test for each metric
        print("  Performing K-S tests...")
        # Length
        ks_len, p_len = ks_2samp(model_metrics['length'], real_metrics['length'])
        interp_len = "Different" if p_len < 0.05 else "Not Different"

        # Unique Phases
        ks_unique, p_unique = ks_2samp(model_metrics['unique_phases'], real_metrics['unique_phases'])
        interp_unique = "Different" if p_unique < 0.05 else "Not Different"

        # 6. Store results
        long_id = get_longitudinal_id(matrix_filename)
        all_results.append({
            'longitudinal_id': long_id,
            'matrix_file': matrix_filename,
            'sequence_file': sequence_filename,
            'num_real_sequences': len(real_sequences),
            'length_p_value': p_len,
            'length_interpretation': interp_len,
            'unique_phases_p_value': p_unique,
            'unique_phases_interpretation': interp_unique
        })

    # --- Save all results to a final CSV file ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(output_dir, "individual_model_fit_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Analysis complete. All results saved to: {results_path}")
    else:
        print("\nNo data was processed. Please check your file paths and names.")


if __name__ == "__main__":
    main()