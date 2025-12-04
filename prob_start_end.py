import pandas as pd
import numpy as np
import os
import argparse
import matplotlib

try:
    # Try a non-interactive backend suitable for scripts
    matplotlib.use('Agg')
    print("Using Agg backend for Matplotlib.")
except ImportError:
    print("Warning: Matplotlib not found. Plotting functions might fail.")
except Exception as e:
    # Fallback or inform the user
    print(f"Warning: Could not set Matplotlib backend to Agg. Error: {e}. Plotting might fail or require a display.")
    pass  # Allow matplotlib to use its default backend

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
import fnmatch  # For pattern matching filenames


def compute_transitions_improved(df, max_gap=60):
    """
    Correctly detects transitions, start events, and end events.
    - A behavior block transitions to the *single closest* subsequent block if the gap is <= max_gap.
    - A "Start Event" is a block not preceded by any other block within max_gap frames.
    - An "End Event" is a block not followed by any other block within max_gap frames.

    Args:
        df (pd.DataFrame): DataFrame with frame number and behavior columns.
        max_gap (int): Maximum frame gap for transitions and for defining start/end events.

    Returns:
        dict: Contains transition counts, behavior occurrences, start counts, and end counts.
    """
    frame_col = df.columns[0]
    behavior_cols = [col for col in df.columns[1:] if col != 'background']

    # 1. Extract all behavior blocks
    behavior_blocks = {}
    for behavior in behavior_cols:
        behavior_blocks[behavior] = []
        if behavior not in df.columns: continue
        try:
            behavior_series = pd.to_numeric(df[behavior], errors='coerce').fillna(0).astype(int)
            transitions_diff = np.diff(np.hstack([[0], behavior_series.values, [0]]))
            starts = np.where(transitions_diff == 1)[0]
            ends = np.where(transitions_diff == -1)[0] - 1
            if len(starts) == 0: continue
            min_len = min(len(starts), len(ends))
            for i in range(min_len):
                if 0 <= starts[i] < len(df) and 0 <= ends[i] < len(df):
                    start_frame = int(df.iloc[starts[i]][frame_col])
                    end_frame = int(df.iloc[ends[i]][frame_col])
                    behavior_blocks[behavior].append((start_frame, end_frame))
        except Exception as e:
            print(f"Error processing blocks for '{behavior}': {e}.")

    # 2. Initialize counts and create flat lists for efficient lookups
    behavior_counts = {b: len(blocks) for b, blocks in behavior_blocks.items()}
    start_counts = defaultdict(int)
    end_counts = defaultdict(int)
    transitions = defaultdict(lambda: defaultdict(int))

    # Create a flat list of all blocks with their name and start/end times
    all_blocks_flat = []
    for behavior_name, blocks in behavior_blocks.items():
        for start_f, end_f in blocks:
            all_blocks_flat.append({'behavior': behavior_name, 'start': start_f, 'end': end_f})

    # 3. For each block, determine its predecessor and successor
    for block_from in all_blocks_flat:
        start_from = block_from['start']
        end_from = block_from['end']
        behavior_from = block_from['behavior']

        # --- Start Check ---
        # Find any block that ends in the window before this one starts
        is_preceded = any(
            start_from - max_gap <= other_block['end'] < start_from
            for other_block in all_blocks_flat
            if other_block != block_from
        )
        if not is_preceded:
            start_counts[behavior_from] += 1

        # --- End/Transition Check ---
        # Find all valid successor blocks within the max_gap
        successors = []
        for block_to in all_blocks_flat:
            if block_to == block_from: continue
            gap = block_to['start'] - end_from
            if 0 < gap <= max_gap:
                successors.append(block_to)

        if successors:
            # If there are successors, find the closest one
            closest_successor = min(successors, key=lambda b: b['start'] - end_from)
            behavior_to = closest_successor['behavior']
            transitions[behavior_from][behavior_to] += 1
        else:
            # If no successors are found, it's an end event
            end_counts[behavior_from] += 1

    total_transitions = sum(sum(t.values()) for t in transitions.values())

    return {
        'transitions': dict(transitions),
        'behavior_counts': behavior_counts,
        'start_counts': dict(start_counts),
        'end_counts': dict(end_counts),
        'total_transitions': total_transitions
    }


def create_transition_matrices(transition_data, behaviors):
    """
    Create raw count and conditional probability transition matrices.
    """
    counts_matrix = pd.DataFrame(0, index=behaviors, columns=behaviors)
    for from_behavior, to_behaviors in transition_data.get('transitions', {}).items():
        if from_behavior in behaviors:
            for to_behavior, count in to_behaviors.items():
                if to_behavior in behaviors:
                    counts_matrix.loc[from_behavior, to_behavior] = count

    # Calculate conditional probability (rows sum to 1)
    # The sum of transitions out of a state + end events for that state = total occurrences of that state
    # The denominator for P(To|From) is the total number of transitions out of 'From' state
    outgoing_transitions = counts_matrix.sum(axis=1)
    # Avoid division by zero for states that only have end events
    row_sums = outgoing_transitions.replace(0, np.nan)
    conditional_probability_matrix = counts_matrix.div(row_sums, axis=0).fillna(0.0)

    return counts_matrix, conditional_probability_matrix


# --- [plot_heatmap function remains the same] ---
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
    cmap = "YlGnBu"  # Consider "viridis", "plasma", "magma" as alternatives
    cbar_label = "Probability"
    auto_vmin = 0.0
    auto_vmax = 1.0

    # Use np.nanmax to handle potential NaNs if matrix comes from reindexing or empty data
    if matrix.empty:
        print(f"Warning: Matrix for '{title}' is empty. Cannot generate heatmap.")
        if standalone_plot: plt.close(fig)
        return ax, None  # Return the axis but no mappable

    finite_values = matrix.values[np.isfinite(matrix.values)]  # Get only finite values

    if matrix_type == 'count':
        fmt = ".0f"
        cbar_label = "Transition Count"
        # Ensure vmax is at least 1 for counts if data exists
        auto_vmax = np.max(finite_values) if finite_values.size > 0 else 1
        auto_vmin = 0
    elif matrix_type == 'conditional_prob':
        cbar_label = "Conditional Probability P(To|From)"
        auto_vmax = 1.0
        auto_vmin = 0.0
    elif matrix_type == 'joint_prob':
        cbar_label = "Joint Probability P(From, To)"
        fmt = ".3f"
        # Ensure vmax is at least slightly positive if data exists
        auto_vmax = np.max(finite_values) if finite_values.size > 0 else 1e-9
        auto_vmin = 0.0

    final_vmin = vmin if vmin is not None else auto_vmin
    final_vmax = vmax if vmax is not None else auto_vmax
    # Prevent vmin == vmax issue
    if final_vmax <= final_vmin:
        final_vmax = final_vmin + 1e-6

    try:
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
            ax=ax
        )

        ax.set_title(title, wrap=True)  # Wrap title if too long
        # These labels are now dynamically set based on matrix index/columns (e.g., 'Phase X')
        ax.set_ylabel(f"From {matrix.index.name if matrix.index.name else 'Behavior'}")
        ax.set_xlabel(f"To {matrix.columns.name if matrix.columns.name else 'Behavior'}")
        # Rotate x-axis labels if they are long (like 'Phase X')
        ax.tick_params(axis='x', rotation=45, labelsize=8)  # Adjust label size
        ax.tick_params(axis='y', labelsize=8)  # Adjust label size
        # Adjust bottom margin if labels rotate
        # Use tight_layout before saving standalone plot

    except Exception as e:
        print(f"Error generating heatmap for '{title}': {e}")
        if standalone_plot: plt.close(fig)
        return ax, None

    if standalone_plot:
        if output_path:
            # Apply tight layout before saving standalone plot
            try:
                fig.tight_layout()
                fig.savefig(output_path)
                plt.close(fig)
                print(f"Heatmap saved to: {output_path}")
            except Exception as e:
                print(f"Error saving heatmap '{output_path}': {e}")
                plt.close(fig)
        else:
            # If not saving, potentially show, or just close if running non-interactively
            # plt.show() # Uncomment if you want plots to display interactively
            plt.close(fig)  # Close after showing or if not saving

    mappable = ax.collections[0] if ax.collections else None
    return ax, mappable


def main():
    BEHAVIOR_TO_PHASE_MAP = {
        'nose': 'Phase 1', 'whiskers': 'Phase 2', 'eyes': 'Phase 3',
        'ears': 'Phase 4', 'body': 'Phase 5'
    }
    PHASE_DISPLAY_ORDER = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
    DEFAULT_BEHAVIORS = list(BEHAVIOR_TO_PHASE_MAP.keys())

    parser = argparse.ArgumentParser(
        description='Compute behavior transition matrices, including start/end probabilities.')
    parser.add_argument('directory', help='Directory containing CSV files (searches recursively)')
    parser.add_argument('--file-pattern', default='*.csv',
                        help='Pattern to match files (e.g., "*.csv"). Default: *.csv')
    parser.add_argument('--max-gap', type=int, default=60,
                        help='Maximum frame gap for transitions and for start/end event detection (default: 60)')
    parser.add_argument('--output-dir',
                        help='Directory to save output files (defaults to DIRECTORY/transition_analysis)')
    parser.add_argument('--generate-heatmaps', action='store_true', help='Generate and save individual heatmap PNGs.')
    parser.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS,
                        help=f'List of behavior names to analyze. Defaults to: {DEFAULT_BEHAVIORS}.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (prints matrices to console).')

    args = parser.parse_args()

    print(f"Analyzing behaviors: {args.behaviors}")

    output_dir = args.output_dir or os.path.join(args.directory, 'transition_analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    rename_map = {orig: phase for orig, phase in BEHAVIOR_TO_PHASE_MAP.items() if orig in args.behaviors}
    output_labels_ordered = [p for p in PHASE_DISPLAY_ORDER if p in rename_map.values()]
    phase_numbers = [p.split(' ')[1] for p in output_labels_ordered]

    files_found = [os.path.join(root, f) for root, _, files in os.walk(args.directory) for f in
                   fnmatch.filter(files, args.file_pattern)]
    print(f"Found {len(files_found)} files to process.")

    all_files_summary_data = []

    for file_path in files_found:
        print(f"\n--- Processing File: {os.path.basename(file_path)} ---")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df.columns) < 2:
                print("Warning: File is empty or invalid. Skipping.")
                continue

            transitions_data = compute_transitions_improved(df, args.max_gap)

            # Unpack all data
            behavior_counts = transitions_data['behavior_counts']
            start_counts = transitions_data['start_counts']
            end_counts = transitions_data['end_counts']

            # Create transition matrices
            counts_matrix_orig, cond_prob_matrix_orig = create_transition_matrices(
                transitions_data, args.behaviors
            )

            # Rename and reorder for output
            counts_matrix = counts_matrix_orig.rename(index=rename_map, columns=rename_map).reindex(
                index=output_labels_ordered, columns=output_labels_ordered, fill_value=0)
            cond_prob_matrix = cond_prob_matrix_orig.rename(index=rename_map, columns=rename_map).reindex(
                index=output_labels_ordered, columns=output_labels_ordered, fill_value=0.0)

            # --- Save Individual Outputs ---
            counts_path = os.path.join(output_dir, f"{base_filename}_transition_counts.csv")
            counts_matrix.to_csv(counts_path)
            cond_prob_path = os.path.join(output_dir, f"{base_filename}_transition_conditional_probabilities.csv")
            cond_prob_matrix.to_csv(cond_prob_path, float_format='%.6f')
            print(f"Saved count and probability tables for {base_filename}")

            if args.generate_heatmaps and sns:
                plot_heatmap(counts_matrix, f"Transition Counts: {base_filename}", 'count',
                             os.path.join(output_dir, f"{base_filename}_counts_heatmap.png"))
                plot_heatmap(cond_prob_matrix, f"Conditional Prob P(To|From): {base_filename}", 'conditional_prob',
                             os.path.join(output_dir, f"{base_filename}_cond_prob_heatmap.png"))

            # --- Collect data for the summary table ---
            file_summary = {'filename': base_filename}
            # Add Start/End probabilities
            # P(Start|behavior) = number of start events / total occurrences of that behavior
            for behavior in args.behaviors:
                display_name = rename_map.get(behavior)
                if not display_name: continue

                b_count = behavior_counts.get(behavior, 0)
                s_count = start_counts.get(behavior, 0)
                e_count = end_counts.get(behavior, 0)

                start_prob = s_count / b_count if b_count > 0 else 0
                end_prob = e_count / b_count if b_count > 0 else 0

                phase_num = display_name.split(' ')[1]
                file_summary[f'Start-{phase_num}'] = start_prob
                file_summary[f'End-{phase_num}'] = end_prob

            # Add inter-behavior transition probabilities
            for from_phase in output_labels_ordered:
                for to_phase in output_labels_ordered:
                    from_num = from_phase.split(' ')[1]
                    to_num = to_phase.split(' ')[1]
                    col_name = f"{from_num}-{to_num}"
                    file_summary[col_name] = cond_prob_matrix.loc[from_phase, to_phase]

            all_files_summary_data.append(file_summary)

            # --- Print summary for this file ---
            print(f"Summary for: {base_filename}")
            for behavior in args.behaviors:
                display_name = rename_map.get(behavior, behavior)
                print(f"  {display_name}: "
                      f"Occurrences={behavior_counts.get(behavior, 0)}, "
                      f"Starts={start_counts.get(behavior, 0)}, "
                      f"Ends={end_counts.get(behavior, 0)}")

            if args.debug:
                print("\nConditional Transition Probability Matrix P(To|From):")
                print(cond_prob_matrix.round(3))

        except Exception as e:
            print(f"FATAL Error processing {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc()

    # --- Generate and Save Full Summary Table ---
    if all_files_summary_data:
        print("\n--- Generating Final Summary Table ---")
        summary_df = pd.DataFrame(all_files_summary_data)

        # Define final column order
        start_cols = [f'Start-{p}' for p in phase_numbers]
        end_cols = [f'End-{p}' for p in phase_numbers]
        transition_cols = [f"{f}-{t}" for f in phase_numbers for t in phase_numbers]
        summary_column_order = ['filename'] + start_cols + transition_cols + end_cols

        summary_df = summary_df.reindex(columns=summary_column_order, fill_value=0.0)

        summary_csv_path = os.path.join(output_dir, "all_files_transition_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
        print(f"Comprehensive summary table saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()