# --- [Includes and other functions remain the same] ---
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib

try:
    matplotlib.use('Agg')
    print("Using Agg backend for Matplotlib.")
except ImportError:
    print("Warning: Matplotlib not found. Plotting functions will fail.")
except Exception as e:
    print(f"Warning: Could not set Matplotlib backend to Agg. Error: {e}")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Needed for colorbar placement

try:
    import seaborn as sns

    print("Seaborn found.")
except ImportError:
    print("Warning: Seaborn not found. Heatmaps will not be generated.")
    sns = None
except Exception as e:
    print(f"Error importing Seaborn: {e}")
    sns = None

from collections import defaultdict


# --- [compute_transitions_improved, create_transition_matrices, plot_heatmap functions are unchanged] ---
# --- [Copy the previous versions of these functions here] ---

# --- [ START: Previous compute_transitions_improved function ] ---
def compute_transitions_improved(df, max_gap=60):
    """
    Improved transition detection between behaviors with gaps up to max_gap frames.
    This version uses a more thorough approach to detect all transitions.

    Args:
        df (pd.DataFrame): DataFrame with frame number and behavior columns
        max_gap (int): Maximum frame gap to consider as a transition

    Returns:
        dict: Transition count matrix and behavior occurrences
    """
    # Get column names, excluding the first column (frame number) and background
    frame_col = df.columns[0]
    behavior_cols = [col for col in df.columns[1:] if col != 'background']

    # Extract behavior blocks with their start and end frames
    behavior_blocks = {}

    for behavior in behavior_cols:
        behavior_blocks[behavior] = []
        if behavior not in df.columns:
            print(f"Warning: Behavior column '{behavior}' not found in DataFrame. Skipping.")
            continue

        behavior_series = df[behavior].astype(int)
        transitions = np.diff(np.hstack([[0], behavior_series.values, [0]]))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0] - 1

        if len(starts) == 0 or len(ends) == 0:
            continue

        min_len = min(len(starts), len(ends))
        starts = starts[:min_len]
        ends = ends[:min_len]

        for i in range(len(starts)):
            if starts[i] < len(df) and ends[i] < len(df) and starts[i] <= ends[i]:
                start_frame = int(df.iloc[starts[i]][frame_col])
                end_frame = int(df.iloc[ends[i]][frame_col])
                behavior_blocks[behavior].append((start_frame, end_frame))
            else:
                print(
                    f"Warning: Invalid indices start={starts[i]}, end={ends[i]} for behavior '{behavior}'. Skipping block.")

    behavior_counts = {behavior: len(blocks) for behavior, blocks in behavior_blocks.items()}
    transitions = defaultdict(lambda: defaultdict(int))
    total_calculated_transitions = 0

    for behavior_from, blocks_from in behavior_blocks.items():
        for start_from, end_from in blocks_from:
            for behavior_to, blocks_to in behavior_blocks.items():
                if behavior_from == behavior_to:
                    continue
                for start_to, end_to in blocks_to:
                    gap = start_to - end_from
                    if 0 < gap <= max_gap:
                        transitions[behavior_from][behavior_to] += 1
                        total_calculated_transitions += 1

    return {
        'transitions': dict(transitions),
        'behavior_counts': behavior_counts
    }


# --- [ END: Previous compute_transitions_improved function ] ---


# --- [ START: Previous create_transition_matrices function ] ---
def create_transition_matrices(transition_data, behaviors):
    """
    Create raw count, conditional probability, and joint probability transition matrices.
    Uses original behavior names for indexing.

    Args:
        transition_data (dict): Dictionary with transition counts and behavior counts
        behaviors (list): List of *original* behavior names (column headers) to include

    Returns:
        tuple: (counts_matrix, conditional_probability_matrix, joint_probability_matrix)
               Matrices are indexed/columned by the original behavior names.
    """
    counts_matrix = pd.DataFrame(0, index=behaviors, columns=behaviors)
    conditional_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)
    joint_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)

    # Fill in the transition counts using original behavior names
    for from_behavior, to_behaviors in transition_data['transitions'].items():
        if from_behavior in behaviors:  # Check if the 'from' behavior is one we are analyzing
            for to_behavior, count in to_behaviors.items():
                if to_behavior in behaviors:  # Check if the 'to' behavior is one we are analyzing
                    # Use .loc for safe assignment even if index/columns don't exist initially (though they should here)
                    counts_matrix.loc[from_behavior, to_behavior] = count

    # Calculate conditional probability (rows sum to 1)
    conditional_probability_matrix = counts_matrix.copy().astype(float)
    for from_behavior in behaviors:
        row_sum = conditional_probability_matrix.loc[from_behavior].sum()
        if row_sum > 0:
            conditional_probability_matrix.loc[from_behavior] = conditional_probability_matrix.loc[
                                                                    from_behavior] / row_sum

    # Calculate joint probability (all cells sum to 1)
    total_transitions = counts_matrix.values.sum()
    if total_transitions > 0:
        joint_probability_matrix = counts_matrix.astype(float) / total_transitions

    # Return matrices with original behavior names as index/columns
    return counts_matrix, conditional_probability_matrix, joint_probability_matrix


# --- [ END: Previous create_transition_matrices function ] ---


# --- [ START: Previous plot_heatmap function ] ---
def plot_heatmap(matrix, title, matrix_type='conditional_prob', output_path=None, vmin=None, vmax=None, ax=None,
                 add_colorbar=False):
    """
    Create a heatmap visualization of the transition matrix on a given Axes object.
    Assumes matrix index/columns are the desired display labels.

    Args:
        matrix (pd.DataFrame): Transition matrix (with desired display labels)
        title (str): Title for the plot
        matrix_type (str): Type of matrix ('count', 'conditional_prob', 'joint_prob')
        output_path (str, optional): Path to save the plot (if ax is None).
        vmin (float, optional): Minimum value for the color scale.
        vmax (float, optional): Maximum value for the color scale.
        ax (matplotlib.axes.Axes, optional): The Axes object to draw the heatmap on.
                                             If None, a new figure and axes are created.
        add_colorbar (bool): Whether to add a colorbar to the axes (relevant when ax is provided).
    """
    if sns is None:
        print("Seaborn not available, skipping heatmap generation.")
        return None, None  # Return None for ax and mappable

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone_plot = True
        add_colorbar_in_heatmap = True  # Let heatmap add cbar for standalone
    else:
        standalone_plot = False
        fig = ax.figure
        add_colorbar_in_heatmap = add_colorbar  # Use flag for existing axes

    fmt = ".2f"
    cmap = "YlGnBu"
    cbar_label = "Probability"
    auto_vmin = 0.0
    auto_vmax = 1.0

    # Use np.nanmax to handle potential NaNs if matrix comes from reindexing
    if matrix_type == 'count':
        fmt = ".0f"
        cbar_label = "Transition Count"
        auto_vmax = np.nanmax(matrix.values) if np.any(np.isfinite(matrix.values)) else 1
        auto_vmin = 0
    elif matrix_type == 'conditional_prob':
        cbar_label = "Conditional Probability P(To|From)"
        auto_vmax = 1.0
        auto_vmin = 0.0
    elif matrix_type == 'joint_prob':
        cbar_label = "Joint Probability P(From, To)"
        fmt = ".3f"
        auto_vmax = np.nanmax(matrix.values) if np.any(np.isfinite(matrix.values)) else 1e-9
        auto_vmin = 0.0

    final_vmin = vmin if vmin is not None else auto_vmin
    final_vmax = vmax if vmax is not None else auto_vmax
    if final_vmax <= final_vmin:
        final_vmax = final_vmin + 1e-6

    g = sns.heatmap(
        matrix,
        annot=True,
        cmap=cmap,
        vmin=final_vmin,
        vmax=final_vmax,
        fmt=fmt,
        linewidths=.5,
        cbar=add_colorbar_in_heatmap,  # Control based on flag
        cbar_kws={'label': cbar_label} if add_colorbar_in_heatmap else {},
        ax=ax, annot_kws={"size": 13, 'fontweight': 'bold'}, square=True
    )

    ax.set_title(title)
    # These labels are now dynamically set based on matrix index/columns (e.g., 'Phase X')
    ax.set_ylabel(f"From {matrix.index.name if matrix.index.name else 'Behavior'}")
    ax.set_xlabel(f"To {matrix.columns.name if matrix.columns.name else 'Behavior'}")
    # Rotate x-axis labels if they are long (like 'Phase X')
    ax.tick_params(axis='x', rotation=45)
    # Adjust bottom margin if labels rotate
    if standalone_plot:
        plt.subplots_adjust(bottom=0.2)

    if standalone_plot:
        if output_path:
            # Apply tight layout before saving standalone plot
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
            print(f"Heatmap saved to: {output_path}")
        else:
            plt.close(fig)  # Close after showing or if not saving

    mappable = ax.collections[0] if ax.collections else None
    return ax, mappable


# --- [ END: Previous plot_heatmap function ] ---


# --- [ MODIFIED function ] ---
def process_group_files(directory, group_prefix, max_gap=60, output_dir=None):
    """
    Process all CSV files for a specific experimental group based on filename prefix.
    Works with original behavior names found in CSVs.

    Args:
        directory (str): Directory containing CSV files
        group_prefix (str): Prefix identifying the experimental group (e.g., "Veh_")
        max_gap (int): Maximum frame gap to consider as a transition
        output_dir (str, optional): Directory to save output files

    Returns:
        dict: Aggregated transition data for the group, keyed by original behavior names.
    """
    group_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # MODIFICATION: Changed from endswith(suffix) to startswith(prefix)
            if file.lower().startswith(group_prefix.lower()):
                full_path = os.path.join(root, file)
                if os.path.isfile(full_path):
                    group_files.append(full_path)

    # MODIFICATION: Updated print statement to refer to "prefix"
    print(
        f"Found {len(group_files)} files for group prefix: '{group_prefix}' in directory '{directory}' and subdirectories.")

    # Combined transition data for the group (using original behavior names as keys)
    group_transitions = defaultdict(lambda: defaultdict(int))
    group_behavior_counts = defaultdict(int)
    total_files_processed = 0

    for file_path in group_files:
        try:
            df = pd.read_csv(file_path)

            if len(df.columns) < 2:
                print(f"Warning: Skipping {file_path} - Less than 2 columns found.")
                continue

            if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                print(
                    f"Warning: First column in {file_path} does not appear to be numeric frame numbers. Check file format.")

            transitions_data = compute_transitions_improved(df, max_gap)

            for from_behavior, to_behaviors in transitions_data['transitions'].items():
                for to_behavior, count in to_behaviors.items():
                    group_transitions[from_behavior][to_behavior] += count

            for behavior, count in transitions_data['behavior_counts'].items():
                group_behavior_counts[behavior] += count

            total_files_processed += 1

        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file {file_path}")
        except FileNotFoundError:
            print(f"Error: File not found {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # MODIFICATION: Updated print statement to refer to "prefix"
    print(f"Finished processing. Total files successfully processed for group {group_prefix}: {total_files_processed}")

    if output_dir:
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            output_dir = None

    total_group_transitions = sum(sum(to_dict.values())
                                  for to_dict in group_transitions.values())
    # MODIFICATION: Updated print statement to refer to "prefix"
    print(f"Total aggregated transitions for group {group_prefix}: {total_group_transitions}")

    return {
        'transitions': dict(group_transitions),
        'behavior_counts': dict(group_behavior_counts)
    }


def main():
    BEHAVIOR_TO_PHASE_MAP = {
        'nose': 'Phase 1',
        'whiskers': 'Phase 2',
        'eyes': 'Phase 3',
        'ears': 'Phase 4',
        'body': 'Phase 5'
    }
    PHASE_DISPLAY_ORDER = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
    DEFAULT_BEHAVIORS = list(BEHAVIOR_TO_PHASE_MAP.keys())

    parser = argparse.ArgumentParser(
        description='Compute and visualize behavior transition counts and probabilities (conditional and joint). Uses Phase 1-5 labels for standard behaviors.')
    parser.add_argument('directory', help='Directory containing CSV files (searches recursively)')
    parser.add_argument('--max-gap', type=int, default=60, help='Maximum frame gap for transitions (default: 60)')
    parser.add_argument('--output-dir',
                        help='Directory to save output files (defaults to DIRECTORY/transition_analysis)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output')

    # MODIFICATION: Changed argument defaults and help text for prefixes.
    parser.add_argument('--groups', nargs='+', default=['Veh_', 'SKF_', 'Rop_'],
                        help='List of file prefixes identifying the groups (e.g., Veh_ SKF_)')

    parser.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS,
                        help=f'List of original behavior names (column headers) to include in the analysis. Defaults to: {DEFAULT_BEHAVIORS}. Only these defaults will be renamed to Phase 1-5.')
    args = parser.parse_args()

    groups = args.groups
    behaviors_to_process = args.behaviors
    print(f"Analyzing original behaviors: {behaviors_to_process}")
    # MODIFICATION: Updated print statement to refer to "prefixes"
    print(f"Looking for group prefixes: {groups}")

    output_dir = args.output_dir or os.path.join(args.directory, 'transition_analysis')
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir}. Output will not be saved. Error: {e}")
        output_dir = None

    if sns is None and output_dir:
        print("Warning: Seaborn is not installed. Heatmaps cannot be generated.")

    all_group_data = {}
    group_matrices_renamed = {}

    # MODIFICATION: Renamed loop variable for clarity
    for group_prefix in groups:
        # MODIFICATION: Updated print statement
        print(f"\n--- Processing Group: {group_prefix} ---")
        # MODIFICATION: Deriving group name from prefix
        group_name = group_prefix.strip('_')

        # MODIFICATION: Passing group_prefix to the processing function
        group_data = process_group_files(args.directory, group_prefix, args.max_gap, output_dir)

        total_group_transitions = sum(sum(to_dict.values()) for to_dict in group_data['transitions'].values())

        if total_group_transitions == 0:
            # MODIFICATION: Updated print statement
            print(
                f"Warning: No transitions found for group: {group_prefix}. Skipping matrix generation and plotting for this group.")
            all_group_data[group_name] = group_data
            group_matrices_renamed[group_name] = None
            continue

        all_group_data[group_name] = group_data

        counts_matrix_orig, cond_prob_matrix_orig, joint_prob_matrix_orig = create_transition_matrices(group_data,
                                                                                                       behaviors_to_process)

        rename_map = {orig: phase for orig, phase in BEHAVIOR_TO_PHASE_MAP.items() if orig in behaviors_to_process}
        output_labels_unordered = [rename_map.get(b, b) for b in behaviors_to_process]
        output_labels_ordered = [p for p in PHASE_DISPLAY_ORDER if p in output_labels_unordered]

        counts_matrix = counts_matrix_orig.rename(index=rename_map, columns=rename_map)
        cond_prob_matrix = cond_prob_matrix_orig.rename(index=rename_map, columns=rename_map)
        joint_prob_matrix = joint_prob_matrix_orig.rename(index=rename_map, columns=rename_map)

        counts_matrix = counts_matrix.reindex(index=output_labels_ordered, columns=output_labels_ordered, fill_value=0)
        cond_prob_matrix = cond_prob_matrix.reindex(index=output_labels_ordered, columns=output_labels_ordered,
                                                    fill_value=0.0)
        joint_prob_matrix = joint_prob_matrix.reindex(index=output_labels_ordered, columns=output_labels_ordered,
                                                      fill_value=0.0)

        group_matrices_renamed[group_name] = {
            'counts': counts_matrix,
            'conditional': cond_prob_matrix,
            'joint': joint_prob_matrix
        }

        if output_dir:
            try:
                counts_path = os.path.join(output_dir, f"transition_counts_{group_name}.csv")
                counts_matrix.to_csv(counts_path)
                print(f"Transition counts saved to: {counts_path}")

                cond_prob_path = os.path.join(output_dir, f"transition_conditional_probabilities_{group_name}.csv")
                cond_prob_matrix.to_csv(cond_prob_path)
                print(f"Conditional transition probabilities saved to: {cond_prob_path}")

                joint_prob_path = os.path.join(output_dir, f"transition_joint_probabilities_{group_name}.csv")
                joint_prob_matrix.to_csv(joint_prob_path)
                print(f"Joint transition probabilities saved to: {joint_prob_path}")

                if sns:
                    counts_heatmap_path = os.path.join(output_dir, f"transition_counts_heatmap_{group_name}.png")
                    plot_heatmap(counts_matrix, f"Transition Counts: {group_name}",
                                 matrix_type='count', output_path=counts_heatmap_path)

                    cond_prob_heatmap_path = os.path.join(output_dir,
                                                          f"transition_conditional_probabilities_heatmap_{group_name}.png")
                    plot_heatmap(cond_prob_matrix, f"Conditional Transition Prob P(To|From): {group_name}",
                                 matrix_type='conditional_prob', output_path=cond_prob_heatmap_path)

                    joint_prob_heatmap_path = os.path.join(output_dir,
                                                           f"transition_joint_probabilities_heatmap_{group_name}.png")
                    plot_heatmap(joint_prob_matrix, f"Joint Transition Prob P(From, To): {group_name}",
                                 matrix_type='joint_prob', output_path=joint_prob_heatmap_path)

            except Exception as e:
                print(f"Error during saving files/plots for group {group_name}: {e}")

        print(f"\n--- Summary for Group: {group_name} ---")
        print("Behavior occurrences (number of times behavior block appears):")
        for behavior in behaviors_to_process:
            count = group_data['behavior_counts'].get(behavior, 0)
            display_name = rename_map.get(behavior, behavior)
            print(f"  {display_name}: {count}")

        print(f"\nTotal Transitions: {total_group_transitions}")
        print("\nTransition Counts Matrix:")
        print(counts_matrix)
        print("\nConditional Transition Probability Matrix P(To|From):")
        print(cond_prob_matrix.round(3))
        print("\nJoint Transition Probability Matrix P(From, To):")
        print(joint_prob_matrix.round(4))
        print(f"(Sum of joint probabilities: {joint_prob_matrix.values.sum():.4f})")

    # --- [ Group Comparison and Combined Figures section is unchanged ] ---

    valid_groups_for_plotting = [name for name, matrices in group_matrices_renamed.items() if matrices is not None]
    num_valid_groups = len(valid_groups_for_plotting)

    if sns and output_dir and num_valid_groups > 0:
        print(f"\n--- Generating Combined Heatmaps with CONSISTENT SCALES for {num_valid_groups} groups ---")

        plot_types = [
            ('counts', 'Transition Counts', 'combined_transition_counts_heatmaps_scaled.png', '.0f', 'count'),
            ('conditional', 'Conditional Probability P(To|From)',
             'combined_transition_conditional_probabilities_heatmaps_scaled.png', '.2f', 'conditional_prob'),
            ('joint', 'Joint Probability P(From, To)', 'combined_transition_joint_probabilities_heatmaps_scaled.png',
             '.3f', 'joint_prob')
        ]

        target_subplot_width_inches = 6.0
        target_subplot_height_inches = 5.0
        top_margin_inches = 1.0
        bottom_margin_inches = 0.8
        colorbar_width_inches = 0.5
        side_padding = 0.5
        total_fig_width = (target_subplot_width_inches * num_valid_groups) + side_padding * 2
        total_fig_height = target_subplot_height_inches + top_margin_inches + bottom_margin_inches

        for matrix_key, plot_title_prefix, filename, fmt_str, matrix_plot_type in plot_types:
            global_vmin = 0.0
            global_vmax = 0.0
            all_max_values = []
            for group_name in valid_groups_for_plotting:
                matrix = group_matrices_renamed[group_name][matrix_key]
                current_max = np.nanmax(matrix.values) if matrix.size > 0 and np.any(np.isfinite(matrix.values)) else 0
                if np.isfinite(current_max):
                    all_max_values.append(current_max)

            if not all_max_values:
                print(f"Warning: No valid data found to determine scale for combined {matrix_key} plot.")
                global_vmax = 1.0
            else:
                global_vmax = max(all_max_values)
            if matrix_plot_type == 'count':
                global_vmax = max(global_vmax, 1)
            elif matrix_plot_type == 'conditional_prob':
                global_vmax = 1.0
            elif matrix_plot_type == 'joint_prob':
                global_vmax = max(global_vmax, 1e-9)
            if global_vmax <= global_vmin:
                global_vmax = global_vmin + 1e-6

            print(f"Combined plot for '{matrix_key}': Using scale vmin={global_vmin:.3f}, vmax={global_vmax:.3f}")

            fig, axes = plt.subplots(1, num_valid_groups,
                                     figsize=(total_fig_width, total_fig_height),
                                     squeeze=False)
            fig.suptitle(f"Combined Heatmaps: {plot_title_prefix}", fontsize=16)
            last_mappable = None

            for i, group_name in enumerate(valid_groups_for_plotting):
                ax = axes[0, i]
                matrix = group_matrices_renamed[group_name][matrix_key]
                _, mappable = plot_heatmap(matrix,
                                           title=group_name,
                                           matrix_type=matrix_plot_type,
                                           vmin=global_vmin,
                                           vmax=global_vmax,
                                           ax=ax,
                                           add_colorbar=False)
                if mappable:
                    last_mappable = mappable
                if i > 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis='y', which='both', left=False)

            if last_mappable:
                last_ax = axes[0, num_valid_groups - 1]
                divider = make_axes_locatable(last_ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar_label = "Count" if matrix_plot_type == 'count' else \
                    ("Cond. Prob." if matrix_plot_type == 'conditional_prob' else "Joint Prob.")
                fig.colorbar(last_mappable, cax=cax, label=cbar_label)
            else:
                print("Warning: Could not add colorbar as no plottable data was found.")

            try:
                fig.tight_layout(rect=[0, 0.03, 0.98, 0.95])
                combined_path = os.path.join(output_dir, filename)
                fig.savefig(combined_path)
                plt.close(fig)
                print(f"Combined {matrix_key} heatmap saved to: {combined_path}")
            except Exception as e:
                print(f"\nError saving combined {matrix_key} heatmap: {e}")
                plt.close(fig)

    # --- [ Debug comparison section is unchanged ] ---
    if args.debug and num_valid_groups > 1:
        print("\n---- Debug Group Comparison (Transition Counts) ----")
        present_phases = []
        if valid_groups_for_plotting:
            first_valid_group = valid_groups_for_plotting[0]
            if group_matrices_renamed[first_valid_group]:
                present_phases = group_matrices_renamed[first_valid_group]['counts'].index.tolist()

        if not present_phases:
            print("No common phases found to compare.")
        else:
            print(f"Comparing phases: {present_phases}")
            for phase_from in present_phases:
                for phase_to in present_phases:
                    transition_exists = any(
                        group_matrices_renamed[name] is not None and
                        phase_from in group_matrices_renamed[name]['counts'].index and
                        phase_to in group_matrices_renamed[name]['counts'].columns and
                        group_matrices_renamed[name]['counts'].loc[phase_from, phase_to] > 0
                        for name in valid_groups_for_plotting
                    )

                    if transition_exists:
                        print(f"\nTransitions {phase_from} -> {phase_to}:")
                        for group_name in valid_groups_for_plotting:
                            count = 0
                            try:
                                if group_matrices_renamed[group_name] is not None:
                                    count = group_matrices_renamed[group_name]['counts'].loc[phase_from, phase_to]
                                else:
                                    count = 0
                            except KeyError:
                                count = 0
                            print(f"  {group_name}: {int(count)}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    if 'pd' not in globals() or 'np' not in globals() or 'matplotlib' not in globals() or 'plt' not in globals():
        print("Error: Missing critical libraries (Pandas, NumPy, Matplotlib). Please install them.")
    else:
        main()