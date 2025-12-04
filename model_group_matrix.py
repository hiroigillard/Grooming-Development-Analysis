import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def normalize_matrix(df):
    """
    --- NEW FUNCTION ---
    Renormalizes the rows of a DataFrame to sum to 1.
    This is crucial for the median matrix, as the median of normalized
    rows is not guaranteed to be normalized.
    """
    # Make a copy to avoid modifying the original DataFrame in-place
    df_normalized = df.copy()
    row_sums = df_normalized.sum(axis=1)

    # Set row sums that are 0 to 1 to avoid division by zero.
    # The elements in these rows are all 0, so they will remain 0.
    row_sums[row_sums == 0] = 1

    # Divide each element by its row sum
    df_normalized = df_normalized.div(row_sums, axis=0)

    return df_normalized


def calculate_transition_matrix(file_path, states):
    """
    Calculates the transition counts and probabilities for a single sequence file.
    """
    counts_df = pd.DataFrame(0, index=states, columns=states)

    with open(file_path, 'r', encoding='latin1') as f:
        for line in f:
            sequence = line.strip()
            if not sequence:
                continue

            first_char = sequence[0]
            if first_char in counts_df.columns:
                counts_df.loc['Start', first_char] += 1

            for i in range(len(sequence) - 1):
                from_state = sequence[i]
                to_state = sequence[i + 1]
                if from_state in counts_df.index and to_state in counts_df.columns:
                    counts_df.loc[from_state, to_state] += 1

            last_char = sequence[-1]
            if last_char in counts_df.index:
                counts_df.loc[last_char, 'End'] += 1

    # This part already normalizes the individual matrices correctly
    row_sums = counts_df.sum(axis=1)
    prob_df = counts_df.div(row_sums, axis=0).fillna(0)

    return prob_df


def plot_heatmap(matrix_df, title, output_path):
    """
    Generates and saves a heatmap for a given transition matrix.
    """
    plot_df = matrix_df.drop(columns=['Start'], errors='ignore').drop(index=['End'], errors='ignore')

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        plot_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=.5,
        square=True
    )
    plt.title(title, fontsize=16)
    plt.xlabel("To State", fontsize=12)
    plt.ylabel("From State", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def plot_combined_heatmap(group_matrices, output_path):
    """
    Generates and saves a single figure with three heatmaps side-by-side,
    using ImageGrid to ensure all plots are square and identically sized.
    """
    plot_order = ['Veh', 'SKF', 'Rop']

    valid_matrices = [m for group in plot_order if (m := group_matrices.get(group)) is not None]
    if not valid_matrices:
        print("No data to plot for combined heatmap.")
        return

    # Drop non-numeric columns for calculation of min/max
    plot_dfs = [df.drop(columns=['Start'], errors='ignore').drop(index=['End'], errors='ignore') for df in
                valid_matrices]
    global_min = min(df.min().min() for df in plot_dfs)
    global_max = max(df.max().max() for df in plot_dfs)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle('Comparison of Group Median Transition Matrices', fontsize=20)

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 3),
                     axes_pad=0.5,
                     cbar_mode='single',
                     cbar_location='right',
                     cbar_pad=0.2)

    heatmap_artist = None
    for i, ax in enumerate(grid):
        group_name = plot_order[i]
        matrix_df = group_matrices.get(group_name)

        if matrix_df is None:
            ax.set_title(f"{group_name} (No Data)", fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        plot_df = matrix_df.drop(columns=['Start'], errors='ignore').drop(index=['End'], errors='ignore')

        g = sns.heatmap(
            plot_df,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            linewidths=.5,
            vmin=global_min,
            vmax=global_max,
            cbar=False,
            square=True
        )
        if g.get_visible():
            heatmap_artist = g.collections[0]

        ax.set_title(group_name, fontsize=16)
        ax.set_xlabel("To State", fontsize=12)

    grid[0].set_ylabel("From State", fontsize=12)

    if heatmap_artist:
        cbar = grid.cbar_axes[0].colorbar(heatmap_artist)
        cbar.set_label('Probability', rotation=270, labelpad=20)

    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined heatmap to: {output_path}")


def main():
    """
    Main function to orchestrate the entire analysis pipeline.
    """
    input_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/self sequences labels/P21/bout_analysis'
    individual_output_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/self sequences labels/P21/individual_matrix'
    group_output_dir = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/self sequences labels/P21/median_matrix'

    os.makedirs(individual_output_dir, exist_ok=True)
    os.makedirs(group_output_dir, exist_ok=True)

    states = ['Start', '1', '2', '3', '4', '5', 'End']
    print(f"Using fixed states: {states}")

    matrices_by_group = {'Veh': [], 'SKF': [], 'Rop': []}

    all_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".txt") and not f.startswith('._')
    ]
    print(f"\nProcessing {len(all_files)} individual sequence files...")

    for filename in all_files:
        file_path = os.path.join(input_dir, filename)
        prob_matrix = calculate_transition_matrix(file_path, states)
        individual_csv_path = os.path.join(individual_output_dir, f"{os.path.splitext(filename)[0]}_matrix.csv")
        prob_matrix.to_csv(individual_csv_path)

        if filename.startswith('Veh_'):
            matrices_by_group['Veh'].append(prob_matrix)
        elif filename.startswith('SKF_'):
            matrices_by_group['SKF'].append(prob_matrix)
        elif filename.startswith('Rop_'):
            matrices_by_group['Rop'].append(prob_matrix)

    print("\nFinished processing individual files.")

    print("\nCalculating and normalizing group median matrices...")

    group_median_matrices = {}
    for group_name, matrices in matrices_by_group.items():
        if not matrices:
            print(f"Warning: No matrices found for group '{group_name}'. Skipping.")
            group_median_matrices[group_name] = None
            continue

        stacked_matrices = np.stack([df.to_numpy() for df in matrices])
        median_matrix_np = np.median(stacked_matrices, axis=0)
        median_matrix_df = pd.DataFrame(median_matrix_np, index=states, columns=states)

        # --- MODIFICATION: Renormalize the median matrix ---
        normalized_median_matrix = normalize_matrix(median_matrix_df)

        group_median_matrices[group_name] = normalized_median_matrix

        group_csv_path = os.path.join(group_output_dir, f"Group_{group_name}_median_matrix.csv")

        # --- MODIFICATION: Save the normalized matrix ---
        normalized_median_matrix.to_csv(group_csv_path)
        print(f"Saved normalized median matrix for group '{group_name}' to: {group_csv_path}")

        heatmap_path = os.path.join(group_output_dir, f"Group_{group_name}_heatmap.png")

        # --- MODIFICATION: Plot the normalized matrix ---
        plot_heatmap(normalized_median_matrix, f"Median Transition Matrix - {group_name} Group", heatmap_path)

    if group_median_matrices:
        print("\nPlotting combined group heatmap...")
        combined_heatmap_path = os.path.join(group_output_dir, "Group_Comparison_heatmap.png")
        plot_combined_heatmap(group_median_matrices, combined_heatmap_path)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()