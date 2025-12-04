import pandas as pd
import numpy as np
import argparse
from scipy.stats import entropy
from itertools import permutations
import os


def calculate_individual_kl_divergence(input_csv, output_dir, group_prefixes=None):
    """
    Calculates KL divergence for each individual animal against the mean of other groups.

    Args:
        input_csv (str): Path to the 'bout_avg_occupancy_summary.csv' file.
        output_dir (str): Directory to save the output files.
        group_prefixes (list): A list of filename prefixes that define the groups.
    """
    if group_prefixes is None:
        group_prefixes = ['SKF', 'Rop', 'Veh']

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_csv}'")
        return

    phase_columns = [col for col in df.columns if col.startswith('Phase')]
    if not phase_columns:
        print("Error: No 'Phase' columns found in the CSV.")
        return

    # --- 1. Store individual distributions and calculate group averages ---
    individual_data = {prefix: {} for prefix in group_prefixes}
    group_avg_distributions = {}
    epsilon = 1e-9  # Add to prevent log(0) errors

    for _, row in df.iterrows():
        for prefix in group_prefixes:
            if row['filename'].startswith(prefix):
                dist = row[phase_columns].values.astype(float)
                dist += epsilon
                dist /= np.sum(dist)
                individual_data[prefix][row['filename']] = dist
                break

    for prefix, individuals in individual_data.items():
        if individuals:
            avg_dist = np.mean(list(individuals.values()), axis=0)
            avg_dist /= np.sum(avg_dist)  # Re-normalize
            group_avg_distributions[prefix] = avg_dist

    if len(group_avg_distributions) < 2:
        print("\nCannot calculate KL Divergence: Fewer than two groups have data.")
        return

    # --- 2. Calculate KL divergence for each individual vs. other group means ---
    all_results_df = pd.DataFrame()
    summary_text = []

    group_pairs = permutations(group_avg_distributions.keys(), 2)

    for p_group_name, q_group_name in group_pairs:

        q_group_mean_dist = group_avg_distributions[q_group_name]

        kl_values = []
        filenames = []

        # Iterate through each animal in the "P" group
        for filename, p_individual_dist in individual_data[p_group_name].items():
            # Calculate KL divergence of this individual vs. the mean of group Q
            kl_div = entropy(pk=p_individual_dist, qk=q_group_mean_dist)

            kl_values.append(kl_div)
            filenames.append(filename)

        if not kl_values:
            continue

        # --- 3. Summarize and store results ---
        mean_kl = np.mean(kl_values)
        std_kl = np.std(kl_values)

        summary_line = (
            f"Comparing individual animals from '{p_group_name}' against the mean of '{q_group_name}':\n"
            f"  Mean D_KL = {mean_kl:.6f}\n"
            f"  Std Dev   = {std_kl:.6f}\n"
        )
        print("\n" + summary_line)
        summary_text.append(summary_line)

        # Add to the DataFrame for CSV export
        temp_df = pd.DataFrame({
            'filename': filenames,
            'comparison': f"D_KL({p_group_name}_indiv || {q_group_name}_mean)",
            'kl_divergence': kl_values
        })
        all_results_df = pd.concat([all_results_df, temp_df], ignore_index=True)

    # --- 4. Save results to files ---
    os.makedirs(output_dir, exist_ok=True)

    # Save the detailed CSV
    csv_output_path = os.path.join(output_dir, 'individual_kl_divergence.csv')
    try:
        all_results_df.to_csv(csv_output_path, index=False, float_format='%.8f')
        print(f"\nDetailed individual KL divergence scores saved to: {csv_output_path}")
    except Exception as e:
        print(f"\nError saving CSV file: {e}")

    # Save the text summary
    txt_output_path = os.path.join(output_dir, 'individual_kl_summary.txt')
    try:
        with open(txt_output_path, 'w') as f:
            f.write("--- Summary of Individual KL Divergence Analysis ---\n\n")
            f.write(
                "Each section compares individual animals from the first group (P) against the average distribution of the second group (Q).\n\n")
            f.writelines(summary_text)
        print(f"Summary report saved to: {txt_output_path}")
    except Exception as e:
        print(f"\nError saving summary text file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate KL Divergence for each individual animal against the mean of other groups.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'input_csv',
        help="Path to the input CSV file (e.g., 'bout_analysis/bout_avg_occupancy_summary.csv')."
    )
    parser.add_argument(
        '--output-dir',
        default='bout_analysis',
        help="Directory to save the output files.\n(default: 'bout_analysis')"
    )
    parser.add_argument(
        '--groups',
        nargs='+',
        default=['SKF', 'Rop', 'Veh'],
        help="A list of filename prefixes to define the groups.\n(default: SKF Rop Veh)"
    )
    args = parser.parse_args()

    calculate_individual_kl_divergence(args.input_csv, args.output_dir, args.groups)


if __name__ == '__main__':
    main()
