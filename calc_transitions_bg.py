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
from mpl_toolkits.axes_grid1 import make_axes_locatable # Needed for colorbar placement

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

# --- [ compute_transitions_improved, create_transition_matrices, process_group_files functions are unchanged from the previous version ] ---
# --- [ Copy the previous versions of these functions here ] ---

# --- [ START: Previous compute_transitions_improved function (with background exit) ] ---
def compute_transitions_improved(df, max_gap=60):
    """
    Improved transition detection between behaviors, including transitions to background.
    A transition to background occurs if a behavior ends and no other tracked
    behavior starts within max_gap frames.

    Args:
        df (pd.DataFrame): DataFrame with frame number and behavior columns
        max_gap (int): Maximum frame gap to consider as a transition or start of background

    Returns:
        dict: Transition count matrix and behavior occurrences
    """
    # Get column names, excluding the first column (frame number) and background
    frame_col = df.columns[0]
    # Important: Ensure 'background' column is excluded if it exists
    behavior_cols = [col for col in df.columns[1:] if col != 'background']

    # Extract behavior blocks with their start and end frames
    behavior_blocks = {}
    all_start_frames = set() # Keep track of all start frames for background check

    for behavior in behavior_cols:
        behavior_blocks[behavior] = []
        if behavior not in df.columns:
            print(f"Warning: Behavior column '{behavior}' not found in DataFrame. Skipping.")
            continue

        behavior_series = df[behavior].astype(int)
        # Find where behavior turns on (0 to 1) and off (1 to 0)
        diffs = np.diff(np.hstack([[0], behavior_series.values, [0]]))
        starts_idx = np.where(diffs == 1)[0]
        ends_idx = np.where(diffs == -1)[0] - 1 # Index of the last frame of the block

        if len(starts_idx) == 0 or len(ends_idx) == 0:
            continue

        min_len = min(len(starts_idx), len(ends_idx))
        starts_idx = starts_idx[:min_len]
        ends_idx = ends_idx[:min_len]

        for i in range(len(starts_idx)):
            # Ensure indices are within DataFrame bounds
            if starts_idx[i] < len(df) and ends_idx[i] < len(df) and starts_idx[i] <= ends_idx[i]:
                start_frame = int(df.iloc[starts_idx[i]][frame_col])
                end_frame = int(df.iloc[ends_idx[i]][frame_col]) # Frame number where behavior ends
                behavior_blocks[behavior].append((start_frame, end_frame))
                all_start_frames.add(start_frame) # Add start frame to the set
            else:
                 print(f"Warning: Invalid indices start_idx={starts_idx[i]}, end_idx={ends_idx[i]} for behavior '{behavior}'. Skipping block.")


    behavior_counts = {behavior: len(blocks) for behavior, blocks in behavior_blocks.items()}
    transitions = defaultdict(lambda: defaultdict(int))
    # Use a set of all start frames for efficient checking later
    all_start_frames_list = sorted(list(all_start_frames))

    for behavior_from, blocks_from in behavior_blocks.items():
        for start_from, end_from in blocks_from:
            found_subsequent_behavior = False
            # Check for transitions to other defined behaviors
            for behavior_to, blocks_to in behavior_blocks.items():
                # Skip checking transitions to self unless specifically required (currently skipped)
                if behavior_from == behavior_to:
                    continue
                for start_to, end_to in blocks_to:
                    gap = start_to - end_from
                    # Is behavior_to starting within the max_gap window after behavior_from ends?
                    if 0 < gap <= max_gap:
                        transitions[behavior_from][behavior_to] += 1
                        found_subsequent_behavior = True
                        # Note: Must continue checking all possible 'to' behaviors

            # Check for transition to background *after* checking all other behaviors
            is_transition_to_background = True
            # Check if any behavior starts in the gap window
            for start_check in all_start_frames_list:
                 # Optimization: If start_check is already beyond the window, stop checking
                 if start_check > end_from + max_gap:
                     break
                 # Check if a start frame falls *strictly within* the gap window
                 if end_from < start_check <= end_from + max_gap:
                     is_transition_to_background = False
                     break # Found a behavior starting in the gap

            # If after checking all starts, none were found in the gap, it's a transition to background
            if is_transition_to_background:
                 if max_gap > 0:
                    transitions[behavior_from]['background_exit'] += 1

    return {
        'transitions': dict(transitions),
        'behavior_counts': behavior_counts
    }
# --- [ END: Previous compute_transitions_improved function (with background exit) ] ---


# --- [ START: Previous create_transition_matrices function (with background exit) ] ---
def create_transition_matrices(transition_data, behaviors):
    """
    Create raw count, conditional probability, and joint probability transition matrices.
    Includes a column for transitions to 'background_exit'.

    Args:
        transition_data (dict): Dictionary with transition counts (may include 'background_exit')
                                and behavior counts. Keys are original behavior names.
        behaviors (list): List of *original* behavior names (column headers) to include
                          as *source* behaviors (rows).

    Returns:
        tuple: (counts_matrix, conditional_probability_matrix, joint_probability_matrix)
               Matrices index = original behavior names.
               Matrices columns = original behavior names + 'background_exit'.
    """
    # Define columns including the potential background exit state
    columns_with_background = behaviors + ['background_exit']

    # Initialize matrices: Rows are the source behaviors, Columns include background exit
    counts_matrix = pd.DataFrame(0, index=behaviors, columns=columns_with_background)
    conditional_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=columns_with_background)
    joint_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=columns_with_background)

    # Fill in the transition counts using original behavior names + 'background_exit'
    total_transitions = 0
    for from_behavior, to_dict in transition_data['transitions'].items():
        # Only consider transitions *from* the specified behaviors
        if from_behavior in behaviors:
            for to_state, count in to_dict.items():
                # Check if the 'to' state is one of the target behaviors OR background_exit
                if to_state in behaviors or to_state == 'background_exit':
                    # Use .loc for safe assignment
                    counts_matrix.loc[from_behavior, to_state] = count
                    total_transitions += count # Accumulate total transitions

    # Calculate conditional probability (rows sum to 1, including background exit)
    conditional_probability_matrix = counts_matrix.copy().astype(float)
    for from_behavior in behaviors:
        # Sum across all possible 'to' states (including background_exit)
        row_sum = conditional_probability_matrix.loc[from_behavior].sum()
        if row_sum > 0:
            conditional_probability_matrix.loc[from_behavior] = conditional_probability_matrix.loc[from_behavior] / row_sum

    # Calculate joint probability (all cells sum to 1)
    if total_transitions > 0:
        joint_probability_matrix = counts_matrix.astype(float) / total_transitions
    else:
        # Avoid division by zero if no transitions occurred
        joint_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=columns_with_background)


    # Return matrices with original behavior names as index and columns including background_exit
    return counts_matrix, conditional_probability_matrix, joint_probability_matrix
# --- [ END: Previous create_transition_matrices function (with background exit) ] ---


# --- [ START: MODIFIED plot_heatmap function ] ---
def plot_heatmap(matrix, title, matrix_type='conditional_prob', output_path=None, vmin=None, vmax=None, ax=None, add_colorbar=False, title_fontsize=14, title_fontweight='bold'):
    """
    Create a heatmap visualization of the transition matrix on a given Axes object.
    Sets specific axis labels ("From Phase", "To Phase") and formats the title.

    Args:
        matrix (pd.DataFrame): Transition matrix (with desired display labels)
        title (str): Title for the plot (will be formatted)
        matrix_type (str): Type of matrix ('count', 'conditional_prob', 'joint_prob')
        output_path (str, optional): Path to save the plot (if ax is None).
        vmin (float, optional): Minimum value for the color scale.
        vmax (float, optional): Maximum value for the color scale.
        ax (matplotlib.axes.Axes, optional): The Axes object to draw the heatmap on.
                                             If None, a new figure and axes are created.
        add_colorbar (bool): Whether to add a colorbar to the axes (relevant when ax is provided).
        title_fontsize (int): Font size for the plot title.
        title_fontweight (str): Font weight for the plot title (e.g., 'bold', 'normal').
    """
    if sns is None:
         print("Seaborn not available, skipping heatmap generation.")
         return None, None # Return None for ax and mappable

    if ax is None:
        # Estimate figsize based on matrix dimensions to prevent squishing
        aspect_ratio = matrix.shape[1] / matrix.shape[0] if matrix.shape[0] > 0 else 1
        fig_height = 6
        fig_width = max(6, fig_height * aspect_ratio * 1.1) # Adjust width
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        standalone_plot = True
        add_colorbar_in_heatmap = True # Let heatmap add cbar for standalone
    else:
        standalone_plot = False
        fig = ax.figure
        add_colorbar_in_heatmap = add_colorbar # Use flag for existing axes

    fmt = ".2f"
    cmap = "YlGnBu"
    cbar_label = "Probability"
    auto_vmin = 0.0
    auto_vmax = 1.0

    # Use np.nanmax to handle potential NaNs if matrix comes from reindexing
    matrix_values_finite = matrix.values[np.isfinite(matrix.values)] if matrix.size > 0 else np.array([])

    if matrix_type == 'count':
        fmt = ".0f"
        cbar_label = "Transition Count"
        auto_vmax = np.max(matrix_values_finite) if matrix_values_finite.size > 0 else 1.0
        auto_vmin = 0
    elif matrix_type == 'conditional_prob':
        cbar_label = "Conditional Probability P(To|From)"
        auto_vmax = 1.0
        auto_vmin = 0.0
    elif matrix_type == 'joint_prob':
        cbar_label = "Joint Probability P(From, To)"
        fmt = ".3f"
        # Max joint prob could be small, find actual max
        auto_vmax = np.max(matrix_values_finite) if matrix_values_finite.size > 0 else 1e-9 # Avoid vmax=0
        auto_vmin = 0.0

    final_vmin = vmin if vmin is not None else auto_vmin
    final_vmax = vmax if vmax is not None else auto_vmax
    # Ensure vmax is strictly greater than vmin for color mapping
    if final_vmax <= final_vmin:
        final_vmax = final_vmin + 1e-6 # Add a small epsilon


    g = sns.heatmap(
        matrix,
        annot=True,
        cmap=cmap,
        vmin=final_vmin,
        vmax=final_vmax,
        fmt=fmt,
        linewidths=.5,
        cbar=add_colorbar_in_heatmap, # Control based on flag
        cbar_kws={'label': cbar_label} if add_colorbar_in_heatmap else {},
        ax=ax,
        annot_kws={"size": 10} # Adjust annotation font size if needed
    )

    # --- Apply requested formatting ---
    ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
    ax.set_ylabel("From Phase") # Set explicitly as requested
    ax.set_xlabel("To Phase")   # Set explicitly as requested
    # --- End formatting changes ---

    # Rotate x-axis labels if they are long
    ax.tick_params(axis='x', rotation=45, labelsize=10) # Adjust label size
    # Ensure y-axis labels aren't cut off
    ax.tick_params(axis='y', rotation=0, labelsize=10) # Adjust label size


    # Adjust layout if standalone - tight_layout usually works well
    if standalone_plot:
         # Before saving, apply tight layout
         try:
             # Add padding to prevent labels/title being cut off
             fig.tight_layout(pad=1.5)
         except ValueError:
              print("Warning: tight_layout failed, plot margins might be suboptimal.")
         if output_path:
            fig.savefig(output_path, dpi=300) # Increase DPI for better quality
            plt.close(fig)
            print(f"Heatmap saved to: {output_path}")
         else:
            # If not saving standalone, still close the figure
            plt.close(fig) # Close after showing or if not saving

    mappable = ax.collections[0] if ax.collections else None
    return ax, mappable
# --- [ END: MODIFIED plot_heatmap function ] ---


# --- [ START: Previous process_group_files function (unchanged) ] ---
def process_group_files(directory, group_suffix, max_gap=60, output_dir=None):
    """
    Process all CSV files for a specific experimental group.
    Works with original behavior names found in CSVs.

    Args:
        directory (str): Directory containing CSV files
        group_suffix (str): Suffix identifying the experimental group (e.g., "_veh.csv")
        max_gap (int): Maximum frame gap to consider as a transition
        output_dir (str, optional): Directory to save output files

    Returns:
        dict: Aggregated transition data for the group, keyed by original behavior names.
              Transitions dictionary may include 'background_exit' as a 'to' key.
    """
    group_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # Make comparison case-insensitive and check endswith
            if file.lower().endswith(group_suffix.lower()):
                full_path = os.path.join(root, file)
                if os.path.isfile(full_path):
                    group_files.append(full_path)


    print(f"Found {len(group_files)} files for group suffix: '{group_suffix}' in directory '{directory}' and subdirectories.")

    # Combined transition data for the group (using original behavior names as keys)
    group_transitions = defaultdict(lambda: defaultdict(int))
    group_behavior_counts = defaultdict(int)
    total_files_processed = 0

    for file_path in group_files:
        try:
            # print(f"Processing file: {file_path}") # Verbose
            df = pd.read_csv(file_path)

            if len(df.columns) < 2:
                print(f"Warning: Skipping {file_path} - Less than 2 columns found.")
                continue

            # Check if first column seems like frame numbers (e.g., numeric, increasing)
            if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                 print(f"Warning: First column in {file_path} does not appear to be numeric frame numbers. Check file format.")

            # Compute transitions for this file using the MODIFIED function
            # This now includes 'background_exit' transitions
            transitions_data = compute_transitions_improved(df, max_gap)

            # Combine with group data (using original behavior names and 'background_exit')
            for from_behavior, to_states in transitions_data['transitions'].items():
                for to_state, count in to_states.items():
                    group_transitions[from_behavior][to_state] += count

            # Combine behavior counts (using original behavior names)
            for behavior, count in transitions_data['behavior_counts'].items():
                group_behavior_counts[behavior] += count

            total_files_processed += 1

        except pd.errors.EmptyDataError:
             print(f"Warning: Skipping empty file {file_path}")
        except FileNotFoundError:
             print(f"Error: File not found {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    print(f"Finished processing. Total files successfully processed for group {group_suffix}: {total_files_processed}")

    # Create output directory if needed (check existence *before* creating)
    if output_dir:
        try:
             if not os.path.exists(output_dir):
                 os.makedirs(output_dir)
                 print(f"Created output directory: {output_dir}")
        except OSError as e:
             print(f"Error creating output directory {output_dir}: {e}")
             output_dir = None # Reset output dir if creation failed

    # Calculate total transitions for the group (for debugging/verification)
    # This now correctly includes transitions *to* background_exit
    total_group_transitions = sum(sum(to_dict.values())
                                  for to_dict in group_transitions.values())
    print(f"Total aggregated transitions for group {group_suffix}: {total_group_transitions}")

    # Return aggregated data keyed by original behavior names
    # The 'transitions' dict may contain 'background_exit' as a key in inner dicts
    return {
        'transitions': dict(group_transitions),
        'behavior_counts': dict(group_behavior_counts)
    }
# --- [ END: Previous process_group_files function (unchanged) ] ---


# --- [ START: MODIFIED main function ] ---
def main():
    # --- Mappings and Display Orders ---
    BEHAVIOR_TO_PHASE_MAP = {
        'nose': 'Phase 1',
        'whiskers': 'Phase 2',
        'eyes': 'Phase 3',
        'ears': 'Phase 4',
        'body': 'Phase 5'
    }
    PHASE_ROW_ORDER = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
    BACKGROUND_EXIT_LABEL = 'Background'
    # Define the desired order for the COLUMNS (Phases + Background)
    # This is used for reindexing before plotting
    COLUMN_ORDER = PHASE_ROW_ORDER + [BACKGROUND_EXIT_LABEL]

    # *** NEW: Map derived group names to desired display names ***
    GROUP_DISPLAY_NAME_MAP = {
        'veh': 'Vehicle',
        'skf': 'SKF-38393',
        'rop': 'Ropinirole'
        # Add other mappings here if needed, e.g., 'drugX': 'Drug X Name'
    }

    DEFAULT_BEHAVIORS = list(BEHAVIOR_TO_PHASE_MAP.keys())

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compute and visualize behavior transition counts and probabilities (conditional and joint), including transitions to background. Uses Phase 1-5 labels for standard behaviors.')
    parser.add_argument('directory', help='Directory containing CSV files (searches recursively)')
    parser.add_argument('--max-gap', type=int, default=60, help='Maximum frame gap for transitions AND for defining transition to background (default: 60)')
    parser.add_argument('--output-dir', help='Directory to save output files (defaults to DIRECTORY/transition_analysis)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output')
    parser.add_argument('--groups', nargs='+', default=['_veh.csv', '_skf.csv', '_rop.csv'],
                        help='List of file suffixes identifying the groups (e.g., _veh.csv _drugA.csv)')
    parser.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS,
                        help=f'List of original behavior names (column headers) to include in the analysis ROWS. Defaults to: {DEFAULT_BEHAVIORS}. Only these defaults will be renamed to Phase 1-5.')
    args = parser.parse_args()

    # --- Setup ---
    groups = args.groups
    behaviors_to_process = args.behaviors # Rows
    print(f"Analyzing transitions FROM original behaviors: {behaviors_to_process}")
    print(f"Looking for group suffixes: {groups}")
    print(f"Using max_gap = {args.max_gap} frames for transitions and background detection.")

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

    # --- Group processing loop ---
    all_group_data = {}
    group_matrices_renamed = {} # Stores renamed matrices for combined plots/debug

    for group_suffix in groups:
        print(f"\n--- Processing Group: {group_suffix} ---")
        # Derive base name (e.g., 'veh', 'skf', 'rop')
        group_name_derived = group_suffix.replace('.csv', '').strip('_').lower() # Use lower for consistent map keys
        # *** Use the map to get the desired display name ***
        display_group_name = GROUP_DISPLAY_NAME_MAP.get(group_name_derived, group_name_derived.capitalize()) # Fallback to capitalized derived name

        # Process files
        group_data = process_group_files(args.directory, group_suffix, args.max_gap, output_dir)

        # Calculate total transitions
        total_group_transitions = sum(
            sum(to_dict.values())
            for from_b, to_dict in group_data['transitions'].items()
            if from_b in behaviors_to_process
        )

        if total_group_transitions == 0:
            print(f"Warning: No transitions found originating from {behaviors_to_process} for group: {display_group_name}. Skipping matrix generation and plotting for this group.")
            all_group_data[display_group_name] = group_data # Use display name as key here too
            group_matrices_renamed[display_group_name] = None
            continue

        all_group_data[display_group_name] = group_data

        # Create matrices (original names + background_exit)
        counts_matrix_orig, cond_prob_matrix_orig, joint_prob_matrix_orig = create_transition_matrices(group_data, behaviors_to_process)

        # --- Rename and Reorder Matrix Index/Columns ---
        row_rename_map = {orig: phase for orig, phase in BEHAVIOR_TO_PHASE_MAP.items() if orig in behaviors_to_process}
        col_rename_map = row_rename_map.copy()
        col_rename_map['background_exit'] = BACKGROUND_EXIT_LABEL

        current_row_labels = [row_rename_map.get(b, b) for b in counts_matrix_orig.index]
        current_col_labels = [col_rename_map.get(c, c) for c in counts_matrix_orig.columns]

        # Define desired ROW order (filter standard phases, add others)
        desired_row_order = [p for p in PHASE_ROW_ORDER if p in current_row_labels]
        desired_row_order += [lbl for lbl in current_row_labels if lbl not in desired_row_order]

        # Define desired COLUMN order (use standard COLUMN_ORDER, filter based on presence)
        desired_col_order = [p for p in COLUMN_ORDER if p in current_col_labels]
        desired_col_order += [lbl for lbl in current_col_labels if lbl not in desired_col_order] # Add any others present


        counts_matrix = counts_matrix_orig.rename(index=row_rename_map, columns=col_rename_map)
        cond_prob_matrix = cond_prob_matrix_orig.rename(index=row_rename_map, columns=col_rename_map)
        joint_prob_matrix = joint_prob_matrix_orig.rename(index=row_rename_map, columns=col_rename_map)

        final_row_order = [r for r in desired_row_order if r in counts_matrix.index]
        final_col_order = [c for c in desired_col_order if c in counts_matrix.columns]

        counts_matrix = counts_matrix.reindex(index=final_row_order, columns=final_col_order, fill_value=0)
        cond_prob_matrix = cond_prob_matrix.reindex(index=final_row_order, columns=final_col_order, fill_value=0.0)
        joint_prob_matrix = joint_prob_matrix.reindex(index=final_row_order, columns=final_col_order, fill_value=0.0)

        # Store RENAMED/REORDERED matrices using display_group_name as key
        group_matrices_renamed[display_group_name] = {
            'counts': counts_matrix,
            'conditional': cond_prob_matrix,
            'joint': joint_prob_matrix
        }

        # --- Save and Plot using RENAMED matrices & display_group_name ---
        if output_dir:
            try:
                # Save CSVs (use derived name for file safety, display name in content)
                safe_group_name = group_name_derived # Use 'veh', 'skf', etc. for filenames
                counts_path = os.path.join(output_dir, f"transition_counts_{safe_group_name}.csv")
                counts_matrix.to_csv(counts_path)
                print(f"Transition counts saved to: {counts_path}")

                cond_prob_path = os.path.join(output_dir, f"transition_conditional_probabilities_{safe_group_name}.csv")
                cond_prob_matrix.to_csv(cond_prob_path)
                print(f"Conditional transition probabilities saved to: {cond_prob_path}")

                joint_prob_path = os.path.join(output_dir, f"transition_joint_probabilities_{safe_group_name}.csv")
                joint_prob_matrix.to_csv(joint_prob_path)
                print(f"Joint transition probabilities saved to: {joint_prob_path}")

                # Create and save individual heatmaps using **display_group_name** for the title
                if sns:
                    counts_heatmap_path = os.path.join(output_dir, f"transition_counts_heatmap_{safe_group_name}.png")
                    plot_heatmap(counts_matrix, title=display_group_name, # Use full name for title
                                matrix_type='count', output_path=counts_heatmap_path)

                    cond_prob_heatmap_path = os.path.join(output_dir, f"transition_conditional_probabilities_heatmap_{safe_group_name}.png")
                    plot_heatmap(cond_prob_matrix, title=display_group_name, # Use full name for title
                                matrix_type='conditional_prob', output_path=cond_prob_heatmap_path)

                    joint_prob_heatmap_path = os.path.join(output_dir, f"transition_joint_probabilities_heatmap_{safe_group_name}.png")
                    plot_heatmap(joint_prob_matrix, title=display_group_name, # Use full name for title
                                matrix_type='joint_prob', output_path=joint_prob_heatmap_path)

            except Exception as e:
                print(f"Error during saving files/plots for group {display_group_name}: {e}")

        # --- Print summary using RENAMED labels & display_group_name ---
        print(f"\n--- Summary for Group: {display_group_name} ---") # Use display name
        print("Behavior occurrences (number of times behavior block appears):")
        for behavior in behaviors_to_process:
            count = group_data['behavior_counts'].get(behavior, 0)
            display_name = row_rename_map.get(behavior, behavior)
            print(f"  {display_name}: {count}")

        print(f"\nTotal Transitions Originating From Analyzed Behaviors: {total_group_transitions}")
        print("\nTransition Counts Matrix:")
        print(counts_matrix)
        print("\nConditional Transition Probability Matrix P(To|From):")
        print(cond_prob_matrix.round(3))
        print("(Sum of conditional probabilities per row:")
        print(cond_prob_matrix.sum(axis=1).round(3))
        print(")")
        print("\nJoint Transition Probability Matrix P(From, To):")
        print(joint_prob_matrix.round(4))
        print(f"(Sum of joint probabilities: {joint_prob_matrix.values.sum():.4f})")


    # --- [ Group Comparison and Combined Figures - Use RENAMED matrices & display names ] ---

    # Get list of groups that have matrix data (using display names as keys now)
    valid_groups_for_plotting = [name for name, matrices in group_matrices_renamed.items() if matrices is not None]
    num_valid_groups = len(valid_groups_for_plotting)

    if sns and output_dir and num_valid_groups > 0:
        print(f"\n--- Generating Combined Heatmaps with CONSISTENT SCALES for {num_valid_groups} groups ---")

        plot_types = [
            ('counts', 'Transition Counts', 'combined_transition_counts_heatmaps_scaled.png', '.0f', 'count'),
            ('conditional', 'Conditional Probability P(To|From)', 'combined_transition_conditional_probabilities_heatmaps_scaled.png', '.2f', 'conditional_prob'),
            ('joint', 'Joint Probability P(From, To)', 'combined_transition_joint_probabilities_heatmaps_scaled.png', '.3f', 'joint_prob')
        ]

        # Adjust subplot dimensions based on typical matrix shape
        matrix_example = group_matrices_renamed[valid_groups_for_plotting[0]]['counts']
        aspect_ratio = matrix_example.shape[1] / matrix_example.shape[0] if matrix_example.shape[0] > 0 else 1
        target_subplot_height_inches = 5.0 # Base height
        # Make width proportional, ensure minimum width
        target_subplot_width_inches = max(5.0, target_subplot_height_inches * aspect_ratio * 0.9)

        top_margin_inches = 1.0  # Space for suptitle
        bottom_margin_inches = 1.2 # Increased space for rotated x-labels
        colorbar_width_inches = 0.5
        inter_plot_padding = 0.7 # Padding between plots
        side_padding = 0.5

        # Calculate total figure dimensions
        total_fig_width = (target_subplot_width_inches * num_valid_groups) + \
                          (inter_plot_padding * (num_valid_groups -1)) + \
                           colorbar_width_inches + side_padding * 2 + 0.5 # Extra padding for cbar label
        total_fig_height = target_subplot_height_inches + top_margin_inches + bottom_margin_inches

        # --- Loop through plot types ---
        for matrix_key, plot_title_prefix, filename, fmt_str, matrix_plot_type in plot_types:

            # Calculate global vmin and vmax (logic unchanged)
            global_vmin = 0.0
            global_vmax = 0.0
            all_max_values = []
            for group_name in valid_groups_for_plotting: # group_name is now display name
                matrix = group_matrices_renamed[group_name][matrix_key]
                matrix_values_finite = matrix.values[np.isfinite(matrix.values)] if matrix.size > 0 else np.array([])
                if matrix_values_finite.size > 0:
                    all_max_values.append(np.max(matrix_values_finite))

            if not all_max_values:
                 print(f"Warning: No valid data found to determine scale for combined {matrix_key} plot.")
                 global_vmax = 1.0 if matrix_plot_type != 'count' else 1
            else:
                 global_vmax = max(all_max_values) if all_max_values else 0

            if matrix_plot_type == 'count':
                 global_vmax = max(global_vmax, 1)
                 global_vmin = 0.0
            elif matrix_plot_type == 'conditional_prob':
                 global_vmax = 1.0
                 global_vmin = 0.0
            elif matrix_plot_type == 'joint_prob':
                 global_vmax = max(global_vmax, 1e-9)
                 global_vmin = 0.0

            if global_vmax <= global_vmin:
                global_vmax = global_vmin + 1e-6


            print(f"Combined plot for '{matrix_key}': Using scale vmin={global_vmin:.3f}, vmax={global_vmax:.3f}")

            # Create the figure and axes
            fig, axes = plt.subplots(1, num_valid_groups,
                                     figsize=(total_fig_width, total_fig_height),
                                     squeeze=False)
            fig.suptitle(f"Combined Heatmaps: {plot_title_prefix}", fontsize=18, fontweight='bold') # Larger overall title

            last_mappable = None

            # --- Plot each group's RENAMED heatmap using display_group_name ---
            for i, group_name in enumerate(valid_groups_for_plotting): # group_name is display name
                ax = axes[0, i]
                matrix = group_matrices_renamed[group_name][matrix_key]

                # Call plot_heatmap with the display name and formatting options
                _, mappable = plot_heatmap(matrix,
                                           title=group_name, # Use display name here
                                           matrix_type=matrix_plot_type,
                                           vmin=global_vmin,
                                           vmax=global_vmax,
                                           ax=ax,
                                           add_colorbar=False, # Add colorbar separately
                                           title_fontsize=14, # Specific title size for subplots
                                           title_fontweight='bold') # Specific title weight
                if mappable:
                     last_mappable = mappable

                if i > 0: # Hide Y labels for subplots other than the first
                     ax.set_ylabel("")
                     ax.tick_params(axis='y', which='both', left=False, labelleft=False)

                # Ensure x-labels don't overlap (rotation already handled in plot_heatmap)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


            # Add single colorbar (logic unchanged)
            if last_mappable:
                last_ax = axes[0, num_valid_groups - 1]
                divider = make_axes_locatable(last_ax)
                cax_width = f"{colorbar_width_inches/target_subplot_width_inches*100:.1f}%"
                cax = divider.append_axes("right", size=cax_width, pad=0.2) # Adjust pad
                cbar_label = "Count" if matrix_plot_type == 'count' else \
                             ("Cond. Prob. P(To|From)" if matrix_plot_type=='conditional_prob' else "Joint Prob. P(From, To)")
                fig.colorbar(last_mappable, cax=cax, label=cbar_label)
            else:
                print("Warning: Could not add colorbar as no plottable data was found.")

            # Final adjustments and saving (adjust rect for new layout)
            try:
                # rect=[left, bottom, right, top]
                fig.tight_layout(rect=[0.03, bottom_margin_inches/total_fig_height * 0.8, 0.97, 1 - (top_margin_inches/total_fig_height * 0.9)])
                combined_path = os.path.join(output_dir, filename)
                fig.savefig(combined_path, dpi=300) # Save combined at higher DPI
                plt.close(fig)
                print(f"Combined {matrix_key} heatmap saved to: {combined_path}")
            except Exception as e:
                print(f"\nError saving combined {matrix_key} heatmap: {e}")
                plt.close(fig)


    # --- [ Debug comparison section - Use display names ] ---
    if args.debug and num_valid_groups > 1:
        print("\n---- Debug Group Comparison (Transition Counts) ----")
        # Determine common labels across groups (using display names as keys)
        common_row_labels = set()
        common_col_labels = set()
        first_group = True
        ref_row_order = []
        ref_col_order = []

        for group_name in valid_groups_for_plotting: # display names
            if group_matrices_renamed[group_name]:
                matrix = group_matrices_renamed[group_name]['counts']
                if first_group:
                    common_row_labels = set(matrix.index)
                    common_col_labels = set(matrix.columns)
                    ref_row_order = matrix.index.tolist() # Use first group's order as ref
                    ref_col_order = matrix.columns.tolist()
                    first_group = False
                else:
                    common_row_labels.intersection_update(matrix.index)
                    common_col_labels.intersection_update(matrix.columns)

        # Filter reference order to only common labels (optional, could show all from ref)
        # ref_row_order = [r for r in ref_row_order if r in common_row_labels]
        # ref_col_order = [c for c in ref_col_order if c in common_col_labels]


        if not ref_row_order or not ref_col_order:
             print("No common states found across groups to compare based on first group's structure.")
        else:
            print(f"Comparing common transitions between states...")
            print(f"(Rows compared based on first group: {ref_row_order})")
            print(f"(Columns compared based on first group: {ref_col_order})")

            for label_from in ref_row_order:
                for label_to in ref_col_order:
                    transition_exists = any(
                        group_matrices_renamed[name] is not None and
                        label_from in group_matrices_renamed[name]['counts'].index and
                        label_to in group_matrices_renamed[name]['counts'].columns and
                        group_matrices_renamed[name]['counts'].loc[label_from, label_to] > 0
                        for name in valid_groups_for_plotting # display names
                    )

                    if transition_exists:
                        print(f"\nTransitions {label_from} -> {label_to}:")
                        for group_name in valid_groups_for_plotting: # display names
                            count = 0
                            try:
                                matrix_dict = group_matrices_renamed[group_name]
                                if matrix_dict is not None and \
                                   label_from in matrix_dict['counts'].index and \
                                   label_to in matrix_dict['counts'].columns:
                                     count = matrix_dict['counts'].loc[label_from, label_to]
                            except KeyError:
                                count = 0
                            print(f"  {group_name}: {int(count)}") # Use display name

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    # --- [ Library availability checks remain the same ] ---
    if 'pd' not in globals() or 'np' not in globals() or 'matplotlib' not in globals() or 'plt' not in globals():
         print("Error: Missing critical libraries (Pandas, NumPy, Matplotlib). Please install them.")
    else:
         main()
# --- [ END: MODIFIED main function ] ---