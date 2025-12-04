import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from scipy.stats import ks_2samp
from itertools import combinations


## -------------------------------- ##
## --- DATA HELPER FUNCTIONS ---
## -------------------------------- ##

def calculate_average_n_for_group(directory, group_prefix):
    """Calculates the average number of sequences per file for a specific group."""
    sequence_counts = []
    files = [f for f in os.listdir(directory) if f.startswith(group_prefix) and f.endswith(".txt")]
    if not files: return 20
    for filename in files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='latin1') as f:
            count = sum(1 for line in f if line.strip())
            sequence_counts.append(count)
    return math.ceil(np.mean(sequence_counts)) if sequence_counts else 20


def generate_sequence(matrix_df, max_length=500):
    """Generates a single sequence based on a transition probability matrix."""
    states = matrix_df.columns.tolist()
    sequence, current_state = [], 'Start'
    while current_state != 'End' and len(sequence) < max_length:
        probabilities = matrix_df.loc[current_state].values
        next_state = np.random.choice(states, p=probabilities)
        if next_state == 'End':
            current_state = 'End'
            continue
        sequence.append(next_state)
        current_state = next_state
    return sequence


def generate_sampling_distribution(matrix_df, n, num_samples=1000):
    """Generates a sampling distribution of the mean for sequence metrics."""
    sample_avg_lengths, sample_avg_uniques = [], []
    for _ in range(num_samples):
        sequences = [generate_sequence(matrix_df) for _ in range(n)]
        if not sequences: continue
        lengths = [len(s) for s in sequences]
        uniques = [len(set(s)) for s in sequences]
        sample_avg_lengths.append(np.mean(lengths))
        sample_avg_uniques.append(np.mean(uniques))
    return pd.DataFrame({'length': sample_avg_lengths, 'unique_phases': sample_avg_uniques})


def calculate_metrics_per_file(directory, group_prefix):
    """Calculates the average metric for each file and includes the filename."""
    results = []
    group_files = [f for f in os.listdir(directory) if f.startswith(group_prefix) and f.endswith(".txt")]
    for filename in group_files:
        file_path = os.path.join(directory, filename)
        sequences_in_file = []
        with open(file_path, 'r', encoding='latin1') as f:
            for line in f:
                if sequence := line.strip(): sequences_in_file.append(list(sequence))
        if not sequences_in_file: continue
        lengths = [len(s) for s in sequences_in_file]
        uniques = [len(set(s)) for s in sequences_in_file]
        results.append({
            'filename': filename,
            'length': np.mean(lengths),
            'unique_phases': np.mean(uniques)
        })
    return pd.DataFrame(results)


## -------------------------------- ##
## --- PLOTTING FUNCTIONS ---
## -------------------------------- ##

def plot_combined_comparison(all_gen_metrics, all_real_metrics, metric_col, output_path):
    """Plots overlaid KDEs for all metrics."""
    plt.figure(figsize=(12, 7))
    colors = {'Veh': 'black', 'SKF': 'blue', 'Rop': 'red'}
    plot_kwargs = {'linewidth': 2.5}
    for group, color in colors.items():
        if group in all_gen_metrics and not all_gen_metrics[group].empty:
            sns.kdeplot(data=all_gen_metrics[group], x=metric_col, color=color, linestyle='-',
                        label=f'{group} Generated (Model)', **plot_kwargs)
        if group in all_real_metrics and not all_real_metrics[group].empty:
            sns.kdeplot(data=all_real_metrics[group], x=metric_col, color=color, linestyle='--',
                        label=f'{group} Real (Animal Avg.)', **plot_kwargs)
    metric_title = metric_col.replace("_", " ").title()
    plt.title(f'Comparison of {metric_title} Distributions', fontsize=16)
    plt.xlabel(metric_title, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Group & Data Type')
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.xlim(left=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined KDE plot to: {output_path}")


def plot_individual_comparison(gen_metrics, real_metrics_per_file, metric_col, group_name, output_path, xlim, ylim):
    """Plots the comparison for a single group with synchronized axes."""
    plt.figure(figsize=(12, 7))
    metric_title = metric_col.replace("_", " ").title()
    sns.histplot(data=gen_metrics, x=metric_col, stat='density', color='lightblue', bins=30, label=f'Generated (Model)')
    if not real_metrics_per_file.empty:
        sns.kdeplot(data=real_metrics_per_file, x=metric_col, color='red', linewidth=3, label=f'Real (Animal Avg.)')
        real_mean = real_metrics_per_file[metric_col].mean()
        plt.axvline(x=real_mean, color='darkred', linestyle='--', linewidth=2,
                    label=f'Mean of Real Avgs. ({real_mean:.2f})')
    plt.title(f'Comparison of Sequence {metric_title} for {group_name} Group', fontsize=16)
    plt.xlabel(metric_title, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.xlim(left=0, right=xlim)
    plt.ylim(bottom=0, top=ylim)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved individual plot to: {output_path}")


## -------------------------------- ##
## --- MAIN EXECUTION BLOCK ---
## -------------------------------- ##

def main():
    """Main function to run the entire analysis and plotting pipeline."""
    matrix_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/sequence_analysis(min_len=2)/P18/median_matrix'
    # Directory where your ORIGINAL sequence files are stored
    real_sequence_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/sequence_analysis(min_len=2)/P18/bout_analysis'
    combined_output_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/sequence_analysis(min_len=2)/P18/group_gen'
    individual_output_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/sequence_analysis(min_len=2)/P18/ind_gen'

    os.makedirs(combined_output_dir, exist_ok=True)
    os.makedirs(individual_output_dir, exist_ok=True)
    groups = ['Veh', 'SKF', 'Rop']
    test_results = []

    all_generated_metrics, all_real_metrics = {}, {}
    for group in groups:
        print(f"\n--- Processing Group: {group} ---")
        n_for_sampling = calculate_average_n_for_group(real_sequence_dir, f"{group}_")
        print(f"Average sequences for this group (n) is: {n_for_sampling}")
        matrix_path = os.path.join(matrix_dir, f"Group_{group}_median_matrix.csv")
        if not os.path.exists(matrix_path): continue
        matrix_df = pd.read_csv(matrix_path, index_col=0)
        print(f"Generating model's sampling distribution with n={n_for_sampling}...")
        all_generated_metrics[group] = generate_sampling_distribution(matrix_df, n=n_for_sampling)
        print("Calculating metrics from real animal files...")
        all_real_metrics[group] = calculate_metrics_per_file(real_sequence_dir, f"{group}_")

        print(f"--- Goodness-of-Fit Testing for {group} ---")
        for metric in ['length', 'unique_phases']:
            generated_data, real_data = all_generated_metrics[group][metric], all_real_metrics[group][metric]
            if not real_data.empty:
                ks_stat, p_val = ks_2samp(generated_data, real_data)
                interpretation = "Different" if p_val < 0.05 else "Not Different"
                test_results.append({
                    'test_type': 'goodness_of_fit', 'metric': metric,
                    'group1': f'{group}_Model', 'group2': f'{group}_Real',
                    'ks_statistic': ks_stat, 'p_value': p_val, 'interpretation': interpretation
                })
                print(f"  K-S Test for '{metric}': p-value = {p_val:.4f} ({interpretation})")

    print("\n--- Performing Between-Group Hypothesis Tests ---")
    for group1, group2 in combinations(groups, 2):
        for metric in ['length', 'unique_phases']:
            # Compare Real Data
            real1, real2 = all_real_metrics[group1][metric], all_real_metrics[group2][metric]
            if not real1.empty and not real2.empty:
                ks_stat, p_val = ks_2samp(real1, real2)
                interpretation = "Different" if p_val < 0.05 else "Not Different"
                test_results.append({'test_type': 'between_group', 'metric': metric, 'group1': f'{group1}_Real',
                                     'group2': f'{group2}_Real', 'ks_statistic': ks_stat, 'p_value': p_val,
                                     'interpretation': interpretation})
                print(f"  Real Data: {group1} vs {group2} ('{metric}'): p-value = {p_val:.4f} ({interpretation})")
            # Compare Model Data
            model1, model2 = all_generated_metrics[group1][metric], all_generated_metrics[group2][metric]
            ks_stat, p_val = ks_2samp(model1, model2)
            interpretation = "Different" if p_val < 0.05 else "Not Different"
            test_results.append({'test_type': 'between_group', 'metric': metric, 'group1': f'{group1}_Model',
                                 'group2': f'{group2}_Model', 'ks_statistic': ks_stat, 'p_value': p_val,
                                 'interpretation': interpretation})
            print(f"  Model Data: {group1} vs {group2} ('{metric}'): p-value = {p_val:.4f} ({interpretation})")

    results_df = pd.DataFrame(test_results)
    results_path = os.path.join(combined_output_dir, "hypothesis_test_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved all test results to: {results_path}")

    # --- NEW: Calculate deviation scores (Z-scores) and export all animal metrics ---
    print("\n--- Exporting Average Metrics & Deviation Scores per Animal File ---")
    model_stats = {}
    for group, df in all_generated_metrics.items():
        model_stats[group] = {
            'length_mean': df['length'].mean(), 'length_std': df['length'].std(),
            'unique_phases_mean': df['unique_phases'].mean(), 'unique_phases_std': df['unique_phases'].std()
        }

    all_animal_metrics_list = []
    for group, df in all_real_metrics.items():
        if not df.empty:
            df_copy = df.copy()
            df_copy['group'] = group
            # Calculate Z-scores for each animal relative to its group's model
            df_copy['length_z_score'] = (df_copy['length'] - model_stats[group]['length_mean']) / model_stats[group][
                'length_std']
            df_copy['unique_phases_z_score'] = (df_copy['unique_phases'] - model_stats[group]['unique_phases_mean']) / \
                                               model_stats[group]['unique_phases_std']
            all_animal_metrics_list.append(df_copy)

    if all_animal_metrics_list:
        final_animal_metrics_df = pd.concat(all_animal_metrics_list, ignore_index=True)
        output_cols = ['group', 'filename', 'length', 'length_z_score', 'unique_phases', 'unique_phases_z_score']
        final_animal_metrics_df = final_animal_metrics_df[output_cols]
        animal_metrics_path = os.path.join(combined_output_dir, "real_animal_average_metrics.csv")
        final_animal_metrics_df.to_csv(animal_metrics_path, index=False)
        print(f"Saved average metrics and Z-scores per file to: {animal_metrics_path}")

    print("\n--- Creating Plots ---")
    plot_combined_comparison(all_generated_metrics, all_real_metrics, 'length',
                             os.path.join(combined_output_dir, "Combined_Comparison_Length.png"))
    plot_combined_comparison(all_generated_metrics, all_real_metrics, 'unique_phases',
                             os.path.join(combined_output_dir, "Combined_Comparison_Unique_Phases.png"))

    data_for_limits = list(all_generated_metrics.values()) + list(all_real_metrics.values())
    global_xlims, global_ylims = {}, {}
    for metric in ['length', 'unique_phases']:
        max_x = max((df[metric].max() for df in data_for_limits if not df.empty and metric in df), default=1)
        global_xlims[metric] = max_x * 1.05
        max_y = 0
        for group in groups:
            if group not in all_generated_metrics: continue
            fig, ax = plt.subplots();
            sns.histplot(data=all_generated_metrics[group], x=metric, stat='density', bins=30, ax=ax)
            if not all_real_metrics[group].empty: sns.kdeplot(data=all_real_metrics[group], x=metric, ax=ax)
            max_y = max(max_y, ax.get_ylim()[1]);
            plt.close(fig)
        global_ylims[metric] = max_y * 1.05

    for group in groups:
        if group not in all_generated_metrics: continue
        plot_individual_comparison(all_generated_metrics[group], all_real_metrics[group], 'length', group,
                                   os.path.join(individual_output_dir, f"{group}_length_comparison.png"),
                                   global_xlims['length'], global_ylims['length'])
        plot_individual_comparison(all_generated_metrics[group], all_real_metrics[group], 'unique_phases', group,
                                   os.path.join(individual_output_dir, f"{group}_unique_phases_comparison.png"),
                                   global_xlims['unique_phases'], global_ylims['unique_phases'])

    print("\n✅ All analyses and plotting complete!")


if __name__ == "__main__":
    main()