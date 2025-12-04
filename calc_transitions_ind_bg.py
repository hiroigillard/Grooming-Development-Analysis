# --- Includes ---
import pandas as pd
import numpy as np
import os
import argparse
import matplotlib
try:
    # Attempt to use a non-interactive backend suitable for scripts
    matplotlib.use('Agg')
    print("Using Agg backend for Matplotlib.")
except ImportError:
    print("Warning: Matplotlib not found. Plotting functions might fail.")
except Exception as e:
    # Catch other potential exceptions like backend not available
    print(f"Warning: Could not set Matplotlib backend to Agg. Error: {e}. Plotting might fail or require a display.")
    pass

# Import pyplot after setting the backend
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable # Often used with heatmaps
except ImportError:
    print("Warning: Matplotlib components (pyplot, axes_grid1) not found or failed to import after backend setting.")
    plt = None
    make_axes_locatable = None


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
import fnmatch
import re # For extracting numbers from phase names

# --- [ START: compute_transitions_improved function (WITH background detection) ] ---
def compute_transitions_improved(df, max_gap=60):
    """
    Improved transition detection between behaviors, INCLUDING transitions to background.
    A transition to background occurs if a behavior ends and no other tracked
    behavior starts within max_gap frames.

    Args:
        df (pd.DataFrame): DataFrame with frame number and behavior columns
        max_gap (int): Maximum frame gap to consider as a transition or start of background

    Returns:
        dict: Contains:
              'transitions': Nested dict {from_behavior: {to_state: count}}. to_state can be another behavior or 'background_exit'.
              'behavior_counts': Dict {behavior: occurrence_count}.
              'total_transitions': Total count of all detected transitions (behavior-behavior + behavior-background).
    """
    frame_col = df.columns[0]
    # Assume behavior columns are all columns except the first (frame) one.
    # Filter out any explicitly named 'background' column if present from previous versions.
    behavior_cols = [col for col in df.columns[1:] if col != 'background']
    behavior_blocks = {}
    all_start_frames = set()

    # --- 1. Identify behavior blocks and all start frames ---
    for behavior in behavior_cols:
        behavior_blocks[behavior] = []
        if behavior not in df.columns:
            print(f"Warning: Behavior column '{behavior}' not found in DataFrame. Skipping.")
            continue
        try:
            # Ensure column is numeric (0/1)
            if pd.api.types.is_numeric_dtype(df[behavior]):
                 behavior_series = df[behavior].astype(int)
            else:
                 # Try conversion, coerce non-numeric to 0
                 behavior_series = pd.to_numeric(df[behavior], errors='coerce').fillna(0).astype(int)
                 print(f"Warning: Column '{behavior}' was not strictly numeric. Coerced non-numeric values to 0.")

            # Use diff to find start (0->1) and end (1->0) points
            # Pad with 0 at start/end to catch transitions at the very beginning/end of the series
            diffs = np.diff(np.hstack([[0], behavior_series.values, [0]]))
            starts_idx = np.where(diffs == 1)[0]
            # End index is the index *before* the change from 1 to 0
            ends_idx = np.where(diffs == -1)[0] - 1

            if len(starts_idx) == 0 or len(ends_idx) == 0: continue # No blocks

            # Pair start and end indices carefully
            paired_blocks = []
            end_idx_ptr = 0
            for start_i in starts_idx:
                # Find the first end_idx >= start_i
                while end_idx_ptr < len(ends_idx) and ends_idx[end_idx_ptr] < start_i:
                    end_idx_ptr += 1
                # If a valid end is found
                if end_idx_ptr < len(ends_idx):
                     # Check array bounds and ensure start <= end index
                    if 0 <= start_i < len(df) and 0 <= ends_idx[end_idx_ptr] < len(df): # Redundant check on start_i <= end_idx[ptr] as end always follows start here
                        start_frame = int(df.iloc[start_i][frame_col])
                        end_frame = int(df.iloc[ends_idx[end_idx_ptr]][frame_col])
                        paired_blocks.append((start_frame, end_frame))
                        all_start_frames.add(start_frame) # Record start frame
                        end_idx_ptr += 1 # Move to next potential end
                    else:
                        # This condition should be rare with the diff logic but acts as a safeguard
                        print(f"Warning: Invalid DataFrame indices derived (start_idx={start_i}, end_idx={ends_idx[end_idx_ptr]}) for behavior '{behavior}'. Skipping block.")
                        # Don't advance end_idx_ptr as this end might match a later start
            behavior_blocks[behavior] = paired_blocks

        except KeyError: print(f"Warning: Column '{behavior}' not found during processing. Skipping.")
        except ValueError: print(f"Warning: Could not convert frame numbers to int for behavior '{behavior}'. Check first column data type.")
        except Exception as e: print(f"Error processing blocks for behavior '{behavior}': {e}. Skipping behavior.")

    behavior_counts = {behavior: len(blocks) for behavior, blocks in behavior_blocks.items()}
    transitions = defaultdict(lambda: defaultdict(int))
    total_calculated_transitions = 0
    all_start_frames_list = sorted(list(all_start_frames)) # Sort for efficient searching

    # --- 2. Identify transitions between behaviors and to background ---
    for behavior_from, blocks_from in behavior_blocks.items():
        for start_from, end_from in blocks_from:
            # --- Check for transitions to OTHER defined behaviors ---
            found_subsequent_behavior_within_gap = False
            for behavior_to, blocks_to in behavior_blocks.items():
                if behavior_from == behavior_to: continue # Don't count self-transitions here
                for start_to, end_to in blocks_to:
                    try:
                        gap = int(start_to) - int(end_from)
                        # Check if behavior_to starts strictly after behavior_from ends, within the max_gap
                        if 0 < gap <= max_gap:
                            transitions[behavior_from][behavior_to] += 1
                            total_calculated_transitions += 1
                            found_subsequent_behavior_within_gap = True
                            # Note: We count *all* transitions within the gap, even if multiple behaviors start.
                            # The 'background_exit' check below is independent.
                    except (ValueError, TypeError): pass # Ignore type errors during int conversion/comparison

            # --- Check for transition to background ---
            # A transition to background happens if NO tracked behavior starts within the gap.
            is_transition_to_background = True
            if max_gap > 0: # Only makes sense if gap > 0
                for start_check in all_start_frames_list:
                    # Optimization: if we are looking at starts far beyond the gap, stop checking
                    if start_check > end_from + max_gap:
                        break
                    # Check if a behavior start occurs *strictly after* the end_from frame and within the gap window
                    if end_from < start_check <= end_from + max_gap:
                        is_transition_to_background = False
                        break # Found a behavior starting, so it's not a background transition
                if is_transition_to_background:
                    transitions[behavior_from]['background_exit'] += 1
                    total_calculated_transitions += 1 # Also count background exits in the total

    return {
        'transitions': dict(transitions), # Convert back to regular dict
        'behavior_counts': behavior_counts,
        'total_transitions': total_calculated_transitions
    }
# --- [ END: compute_transitions_improved function (WITH background detection) ] ---


# --- [ START: create_transition_matrices function (WITH background column) ] ---
def create_transition_matrices(transition_data, behaviors):
    """
    Create raw count, conditional probability, and joint probability transition matrices.
    Includes a column for transitions to 'background_exit'.

    Args:
        transition_data (dict): Dictionary from compute_transitions_improved.
                                Contains 'transitions', 'behavior_counts', 'total_transitions'.
        behaviors (list): List of *original* behavior names (column headers from CSV)
                          to include as *source* behaviors (rows of the matrices).

    Returns:
        tuple: (counts_matrix, conditional_probability_matrix, joint_probability_matrix)
               Matrices index = original behavior names provided in `behaviors`.
               Matrices columns = original behavior names + 'background_exit'.
               Returns DataFrames filled with 0s if no transitions occurred.
    """
    total_transitions = transition_data.get('total_transitions', 0)
    transitions_dict = transition_data.get('transitions', {})

    # Define potential columns: all specified behaviors + the background state
    all_possible_states = behaviors + ['background_exit']

    # Initialize matrices with specified behaviors as rows and all states as columns
    # Use pd.Index for potentially better performance and clarity
    row_index = pd.Index(behaviors, name="From Behavior")
    col_index = pd.Index(all_possible_states, name="To State")

    counts_matrix = pd.DataFrame(0, index=row_index, columns=col_index)
    conditional_probability_matrix = pd.DataFrame(0.0, index=row_index, columns=col_index)
    joint_probability_matrix = pd.DataFrame(0.0, index=row_index, columns=col_index)

    # If no transitions were found at all, return the zero matrices
    if total_transitions == 0 and not transitions_dict:
        print("Warning: No transitions found in the data. Returning zero matrices.")
        return counts_matrix, conditional_probability_matrix, joint_probability_matrix

    # Fill counts matrix from the transitions dictionary
    for from_behavior, to_dict in transitions_dict.items():
        if from_behavior in counts_matrix.index: # Only consider specified source behaviors
            for to_state, count in to_dict.items():
                if to_state in counts_matrix.columns: # Target state must be a behavior or 'background_exit'
                    counts_matrix.loc[from_behavior, to_state] = count

    # Recalculate total transitions *from the matrix content* to ensure consistency
    # This accounts for only transitions *from* the specified `behaviors` list
    total_matrix_transitions = counts_matrix.values.sum()

    # Conditional Probability P(To | From) - Rows sum to 1
    # Sum counts for each 'From' behavior (row sums)
    row_sums = counts_matrix.sum(axis=1)
    # Avoid division by zero for behaviors that never occurred or never transitioned
    # Where row_sum is 0, the conditional probability is 0 for all 'To' states.
    conditional_probability_matrix = counts_matrix.div(row_sums, axis=0).fillna(0.0)

    # Joint Probability P(From, To) - All cells sum to 1
    if total_matrix_transitions > 0:
        joint_probability_matrix = counts_matrix.astype(float) / total_matrix_transitions
    # else: joint_probability_matrix remains DataFrame of 0.0

    return counts_matrix, conditional_probability_matrix, joint_probability_matrix
# --- [ END: create_transition_matrices function (WITH background column) ] ---


# --- [ START: plot_heatmap function (MODIFIED for non-square, label change) ] ---
def plot_heatmap(matrix, title, matrix_type='conditional_prob', output_path=None, vmin=None, vmax=None, ax=None, add_colorbar=False):
    """
    Create a heatmap visualization of the transition matrix on a given Axes object.
    Sets specific axis labels ("From Phase", "To State") and formats the title.
    Does NOT assume square matrix.

    Args:
        matrix (pd.DataFrame): Transition matrix (index/columns should have desired display labels)
        title (str): Title for the plot
        matrix_type (str): Type of matrix ('count', 'conditional_prob', 'joint_prob') used for colorbar label/format.
        output_path (str, optional): Path to save the plot (if ax is None). Ignored if ax is provided.
        vmin (float, optional): Minimum value for the color scale. Auto-calculated if None.
        vmax (float, optional): Maximum value for the color scale. Auto-calculated if None.
        ax (matplotlib.axes.Axes, optional): The Axes object to draw the heatmap on.
                                             If None, a new figure and axes are created.
        add_colorbar (bool): Whether to add a colorbar to the axes (relevant when ax is provided). Ignored if ax is None.
    """
    # --- Setup and Error Handling ---
    if sns is None or plt is None:
        print("Seaborn or Matplotlib not available, skipping heatmap.")
        return None, None
    if matrix is None or matrix.empty:
        print(f"Warning: Matrix for '{title}' is None or empty. Cannot generate heatmap.")
        # If ax was provided, return it unchanged. If not, return None.
        return ax, None

    create_new_figure = ax is None
    if create_new_figure:
        # Adjust initial figsize based on aspect ratio, ensuring minimum size
        aspect_ratio = matrix.shape[1] / matrix.shape[0] if matrix.shape[0] > 0 else 1
        fig_height = 6
        fig_width = max(6, min(15, fig_height * aspect_ratio * 0.9)) # Avoid excessively wide plots
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        add_colorbar_in_heatmap = True # Add colorbar for standalone plots
    else:
        fig = ax.figure
        add_colorbar_in_heatmap = add_colorbar # Respect caller's choice for provided axes

    # --- Color/Format Setup based on matrix type ---
    fmt = ".2f"; cmap = "YlGnBu"; cbar_label = "Probability"; auto_vmin = 0.0; auto_vmax = 1.0
    # Use numpy to safely handle potential NaNs or Infs when calculating ranges
    finite_values = matrix.values[np.isfinite(matrix.values)]

    if matrix_type == 'count':
        fmt = ".0f"; cbar_label = "Transition Count"
        auto_vmax = np.max(finite_values) if finite_values.size > 0 else 1
        auto_vmin = 0
    elif matrix_type == 'conditional_prob':
        cbar_label = "Conditional Probability P(To|From)"
        auto_vmax = 1.0; auto_vmin = 0.0
    elif matrix_type == 'joint_prob':
        cbar_label = "Joint Probability P(From, To)"
        fmt = ".3f" # Higher precision often useful for joint probs
        auto_vmax = np.max(finite_values) if finite_values.size > 0 else 1e-9 # Handle all-zero case
        auto_vmin = 0.0

    # Use provided vmin/vmax if available, otherwise use auto-calculated values
    final_vmin = vmin if vmin is not None else auto_vmin
    final_vmax = vmax if vmax is not None else auto_vmax
    # Ensure vmax is strictly greater than vmin for color mapping
    if final_vmax <= final_vmin:
        final_vmax = final_vmin + 1e-6 # Add a tiny offset

    # --- Plotting ---
    try:
        heatmap_obj = sns.heatmap(
            matrix,
            annot=True,             # Show values on cells
            cmap=cmap,              # Color map
            vmin=final_vmin,        # Min value for color scale
            vmax=final_vmax,        # Max value for color scale
            fmt=fmt,                # String format for annotations
            linewidths=.5,          # Lines between cells
            cbar=add_colorbar_in_heatmap, # Whether to draw the color bar
            cbar_kws={'label': cbar_label} if add_colorbar_in_heatmap else {}, # Color bar label
            ax=ax,                  # Axes to draw on
            annot_kws={"size": 8},  # Font size for annotations
            square=False            # Allow rectangular heatmap based on matrix shape
        )

        ax.set_title(title, wrap=True, fontsize=14, fontweight='bold')
        ax.set_ylabel("From Phase") # Y-axis label (Source)
        ax.set_xlabel("To State")   # X-axis label (Destination)

        # Improve tick label readability
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)
        # Adjust x-tick alignment for rotated labels
        plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

    except Exception as e:
        print(f"Error generating heatmap for '{title}': {e}")
        # If we created the figure, close it on error. If ax was provided, leave it.
        if create_new_figure: plt.close(fig)
        return ax, None # Return original ax (or None) and no mappable

    # --- Saving / Closing (only if figure was created here) ---
    mappable = heatmap_obj.collections[0] if hasattr(heatmap_obj, 'collections') and heatmap_obj.collections else None

    if create_new_figure:
        if output_path:
            try:
                # Adjust layout to prevent labels overlapping plot area/title
                fig.tight_layout(pad=1.5)
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Heatmap saved to: {output_path}")
            except Exception as e:
                print(f"Error saving heatmap '{output_path}': {e}")
            finally:
                 plt.close(fig) # Ensure figure is closed even if saving fails
        else:
             plt.close(fig) # Close if no output path specified

    # Return the axes and the heatmap mappable (for potential external colorbar)
    return ax, mappable
# --- [ END: plot_heatmap function (MODIFIED for non-square, label change) ] ---


# --- [ START: main function (incorporating background logic and BOTH summaries) ] ---
def main():
    # --- Mappings and Display Orders ---
    BEHAVIOR_TO_PHASE_MAP = {'nose': 'Phase 1', 'whiskers': 'Phase 2', 'eyes': 'Phase 3', 'ears': 'Phase 4', 'body': 'Phase 5'}
    # Default order for rows in matrices/plots if these phases exist
    PHASE_ROW_ORDER = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
    # Label for the background state in matrices/plots
    BACKGROUND_EXIT_LABEL = 'Background'
    # Abbreviation for background state used in summary table column names (e.g., "1-Bkg")
    BACKGROUND_SUMMARY_ABBR = 'Bkg'
    # Default order for columns in matrices/plots (Phases + Background)
    COLUMN_ORDER = PHASE_ROW_ORDER + [BACKGROUND_EXIT_LABEL]
    # Default behavior names expected in input CSV files
    DEFAULT_BEHAVIORS = list(BEHAVIOR_TO_PHASE_MAP.keys())

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Compute and visualize behavior transition counts and probabilities (joint & conditional, including to background) for *each* input CSV file. Generates individual outputs and two summary tables.')
    parser.add_argument('directory', help='Directory containing CSV files (searches recursively)')
    parser.add_argument('--file-pattern', default='*.csv', help='Pattern to match files (default: *.csv)')
    parser.add_argument('--max-gap', type=int, default=60, help='Maximum frame gap for transitions AND background detection (default: 60)')
    parser.add_argument('--output-dir', help='Directory to save output files (defaults to DIRECTORY/individual_transition_analysis_with_bkg)')
    parser.add_argument('--generate-heatmaps', action='store_true', help='Generate individual heatmap PNG files for counts, conditional, and joint probabilities.')
    # Allow user to specify which original behaviors to use as ROWS (sources)
    parser.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS, help=f'List of original behavior names (CSV columns) to analyze as transition sources (rows). Defaults: {DEFAULT_BEHAVIORS}.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (prints matrices to console).')
    args = parser.parse_args()

    behaviors_to_process = args.behaviors # These define the ROWS of the matrices
    print(f"Analyzing transitions FROM original behaviors: {behaviors_to_process}")
    print(f"Looking for files matching pattern: '{args.file_pattern}' in directory '{args.directory}'")
    print(f"Including transitions to background (max_gap = {args.max_gap})")

    # --- Setup Output Dir ---
    output_dir = args.output_dir or os.path.join(args.directory, 'individual_transition_analysis_with_bkg')
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir}. Error: {e}")
        output_dir = None # Disable saving if directory creation fails
    if args.generate_heatmaps and (sns is None or plt is None) and output_dir:
        print("Warning: Seaborn and/or Matplotlib needed for heatmaps. Heatmaps will not be generated.")
        args.generate_heatmaps = False # Disable heatmap generation

    # --- Find Files ---
    files_processed_count = 0
    files_found = []
    for root, _, files in os.walk(args.directory):
        for filename in files:
            if fnmatch.fnmatch(filename, args.file_pattern):
                files_found.append(os.path.join(root, filename))
    print(f"Found {len(files_found)} files matching pattern '{args.file_pattern}'.")

    # --- Data structures for summaries ---
    all_files_joint_summary_data = []
    all_files_cond_summary_data = []
    processed_files_order = [] # Keep track of file order for consistent summary rows

    # --- Helper to get summary table column label (e.g., "1", "Bkg") ---
    def get_summary_label(phase_label, background_abbr=BACKGROUND_SUMMARY_ABBR):
        if phase_label == BACKGROUND_EXIT_LABEL:
            return background_abbr
        # Try to extract trailing number (Phase number)
        match = re.search(r'\d+$', phase_label)
        return match.group(0) if match else phase_label # Return number or original label if no number found

    # --- Process Files ---
    for file_path in files_found:
        print(f"\n--- Processing File: {os.path.basename(file_path)} ---")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        try:
            df = pd.read_csv(file_path)
            if df.empty: print("Warning: File is empty. Skipping."); continue
            if len(df.columns) < 2: print(f"Warning: Skipping {os.path.basename(file_path)} - needs frame column + >=1 behavior column."); continue
            # Basic check for numeric first column (frames)
            if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                try:
                    # Attempt conversion to check if it's convertible
                    pd.to_numeric(df.iloc[:, 0], errors='raise')
                except (ValueError, TypeError):
                     print(f"Warning: First column in {os.path.basename(file_path)} is not numeric. Assuming it's frame numbers, but errors might occur.")

            # --- Core Computation ---
            # Compute transitions (including background) using the improved function
            transitions_data = compute_transitions_improved(df, args.max_gap)

            # Print behavior occurrences found
            print("Behavior occurrences (blocks found):")
            active_rename_map = {orig: phase for orig, phase in BEHAVIOR_TO_PHASE_MAP.items() if orig in behaviors_to_process}
            for behavior in behaviors_to_process:
                occurrence_count = transitions_data['behavior_counts'].get(behavior, 0)
                phase_label = active_rename_map.get(behavior, behavior) # Use phase label if mapped
                print(f"  {phase_label} ({behavior}): {occurrence_count}")

            if transitions_data['total_transitions'] == 0:
                print("Warning: No transitions (incl. to background) detected in this file. Skipping matrix generation and summary data collection for this file.")
                # Add empty rows to summaries? Or skip? Skipping for now.
                continue

            # Create all three matrices (Counts, Conditional P, Joint P)
            counts_matrix_orig, cond_prob_matrix_orig, joint_prob_matrix_orig = create_transition_matrices(
                transitions_data, behaviors_to_process # Pass only the requested source behaviors
            )
            # create_transition_matrices handles the no-transition case internally now
            # but we check the sum of the resulting matrix just in case.
            if counts_matrix_orig.values.sum() == 0:
                 print("Warning: Transition matrix calculation resulted in zero counts (might occur if specified behaviors had no outgoing transitions). Skipping further processing for this file.")
                 continue


            # --- Rename and Reorder Matrix Index/Columns for Clarity ---
            # Map original behavior names (rows) to Phase labels
            row_rename_map = {orig: phase for orig, phase in BEHAVIOR_TO_PHASE_MAP.items() if orig in counts_matrix_orig.index}
            # Map original behavior names AND 'background_exit' (columns) to Phase/Background labels
            col_rename_map = row_rename_map.copy() # Start with behavior->phase mappings
            col_rename_map['background_exit'] = BACKGROUND_EXIT_LABEL # Add mapping for background state

            # Determine the actual labels present after renaming
            current_row_labels = [row_rename_map.get(b, b) for b in counts_matrix_orig.index]
            current_col_labels = [col_rename_map.get(c, c) for c in counts_matrix_orig.columns]

            # Define the desired display order based on standards, filtering by what's actually present
            # Rows: Use PHASE_ROW_ORDER, filtered by current_row_labels
            desired_row_order = [p for p in PHASE_ROW_ORDER if p in current_row_labels]
            # Add any other unexpected row labels found (maintaining relative order)
            desired_row_order += [lbl for lbl in current_row_labels if lbl not in desired_row_order]

            # Columns: Use COLUMN_ORDER (Phases + Background), filtered by current_col_labels
            desired_col_order = [p for p in COLUMN_ORDER if p in current_col_labels]
             # Add any other unexpected col labels found
            desired_col_order += [lbl for lbl in current_col_labels if lbl not in desired_col_order]

            # Apply renaming and reordering (reindexing) to all three matrices
            counts_matrix = counts_matrix_orig.rename(index=row_rename_map, columns=col_rename_map)\
                                              .reindex(index=desired_row_order, columns=desired_col_order, fill_value=0)
            cond_prob_matrix = cond_prob_matrix_orig.rename(index=row_rename_map, columns=col_rename_map)\
                                                .reindex(index=desired_row_order, columns=desired_col_order, fill_value=0.0)
            joint_prob_matrix = joint_prob_matrix_orig.rename(index=row_rename_map, columns=col_rename_map)\
                                                  .reindex(index=desired_row_order, columns=desired_col_order, fill_value=0.0)
            # --- End Rename/Reorder ---


            # --- Save Individual Outputs (Matrices and optional Heatmaps) ---
            if output_dir:
                try:
                    # Save matrices as CSV
                    counts_path = os.path.join(output_dir, f"{base_filename}_transition_counts.csv")
                    counts_matrix.to_csv(counts_path)
                    cond_prob_path = os.path.join(output_dir, f"{base_filename}_transition_conditional_probabilities.csv")
                    cond_prob_matrix.to_csv(cond_prob_path, float_format='%.6f') # Save probabilities with precision
                    joint_prob_path = os.path.join(output_dir, f"{base_filename}_transition_joint_probabilities.csv")
                    joint_prob_matrix.to_csv(joint_prob_path, float_format='%.6f')
                    print(f"Saved counts, conditional P, and joint P tables for {base_filename}")

                    # Generate heatmaps if requested and possible
                    if args.generate_heatmaps:
                        plot_heatmap(counts_matrix, f"Transition Counts: {base_filename}", matrix_type='count',
                                     output_path=os.path.join(output_dir, f"{base_filename}_transition_counts_heatmap.png"))
                        plot_heatmap(cond_prob_matrix, f"Conditional Prob P(To|From): {base_filename}", matrix_type='conditional_prob',
                                     output_path=os.path.join(output_dir, f"{base_filename}_transition_conditional_probabilities_heatmap.png"))
                        plot_heatmap(joint_prob_matrix, f"Joint Prob P(From, To): {base_filename}", matrix_type='joint_prob',
                                     output_path=os.path.join(output_dir, f"{base_filename}_transition_joint_probabilities_heatmap.png"))
                        print(f"Saved heatmaps for {base_filename}")
                except Exception as e:
                    print(f"Error saving individual files/plots for {base_filename}: {e}")


            # --- Collect data for BOTH summary tables ---
            processed_files_order.append(base_filename) # Store filename order
            file_joint_summary = {'filename': base_filename}
            file_cond_summary = {'filename': base_filename}

            # Iterate through the final rows (From Phase) and columns (To State) of the processed matrices
            for from_phase_label in desired_row_order: # Use the reordered index
                from_summary_label = get_summary_label(from_phase_label) # Get '1', '2', etc.
                for to_state_label in desired_col_order: # Use the reordered columns
                    to_summary_label = get_summary_label(to_state_label, BACKGROUND_SUMMARY_ABBR) # Get '1', 'Bkg', etc.
                    col_name = f"{from_summary_label}-{to_summary_label}" # e.g., "1-1", "1-Bkg"

                    try:
                        # Get JOINT probability
                        joint_prob = joint_prob_matrix.loc[from_phase_label, to_state_label]
                        file_joint_summary[col_name] = joint_prob

                        # Get CONDITIONAL probability
                        cond_prob = cond_prob_matrix.loc[from_phase_label, to_state_label]
                        file_cond_summary[col_name] = cond_prob

                    except KeyError: # Should not happen with reindex+fillna, but as safety
                        file_joint_summary[col_name] = np.nan # Or 0.0?
                        file_cond_summary[col_name] = np.nan # Or 0.0?
            # Add this file's data to the overall lists
            all_files_joint_summary_data.append(file_joint_summary)
            all_files_cond_summary_data.append(file_cond_summary)


            # --- Print summary stats for this file ---
            print(f"Total Transitions Detected (matrix sum): {counts_matrix.values.sum()}") # Use matrix sum post-filtering
            # print(f"(Original total transitions reported: {transitions_data['total_transitions']})") # Optional comparison
            print(f"Sum of joint probabilities in matrix: {joint_prob_matrix.values.sum():.4f}") # Should be close to 1.0

            if args.debug:
                print("\n--- Debug Matrices (Post Renaming/Reordering) ---")
                print("Transition Counts Matrix:")
                print(counts_matrix)
                print("\nConditional Transition Probability Matrix P(To|From):")
                print(cond_prob_matrix.round(4))
                print("\nJoint Transition Probability Matrix P(From, To):")
                print(joint_prob_matrix.round(4))
                print("----------------------------------------------------")

            files_processed_count += 1

        # --- Error Handling for File Processing Loop ---
        except pd.errors.EmptyDataError: print(f"Warning: Skipping empty file {os.path.basename(file_path)}")
        except FileNotFoundError: print(f"Error: File not found {file_path}")
        except Exception as e:
            print(f"\n!!! Error processing file {os.path.basename(file_path)}: {e} !!!")
            import traceback
            traceback.print_exc()
            print("!!! Attempting to continue with next file... !!!")
        # --- End Error Handling ---

    # --- Analysis Complete ---
    print(f"\n--- Analysis Complete ---")
    print(f"Successfully processed {files_processed_count} out of {len(files_found)} files found.")

    # --- Generate and Save SUMMARY Tables ---
    if output_dir:
        # --- Generate and Save JOINT Probability Summary ---
        if all_files_joint_summary_data:
            print("\n--- Generating Joint Probability Summary Table ---")
            try:
                summary_df = pd.DataFrame(all_files_joint_summary_data)
                summary_df.set_index('filename', inplace=True)

                # Define column order based on unique labels found across all files
                phase_numbers_present = sorted(list(set(
                    col.split('-')[0] for col in summary_df.columns if '-' in col and col.split('-')[0] != BACKGROUND_SUMMARY_ABBR
                )), key=int) # Sort numerically if possible
                summary_column_order = []
                for from_num in phase_numbers_present:
                    for to_num in phase_numbers_present: # Phase-to-Phase
                        summary_column_order.append(f"{from_num}-{to_num}")
                    bkg_col_name = f"{from_num}-{BACKGROUND_SUMMARY_ABBR}" # Phase-to-Background
                    if bkg_col_name in summary_df.columns: # Only add if background transitions occurred
                        summary_column_order.append(bkg_col_name)

                # Reindex DataFrame to ensure consistent columns and order, fill missing with 0.0
                summary_df = summary_df.reindex(columns=summary_column_order, fill_value=0.0)
                # Reorder rows based on the order files were processed
                summary_df = summary_df.reindex(index=processed_files_order)
                # Reset index so 'filename' becomes a column
                summary_df.reset_index(inplace=True)

                summary_csv_path = os.path.join(output_dir, "all_files_joint_transition_probabilities_summary_with_bkg.csv")
                summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f')
                print(f"Joint probability summary table saved to: {summary_csv_path}")

            except Exception as e:
                print(f"Error generating or saving the JOINT probability summary table: {e}")
                import traceback; traceback.print_exc()
        else:
            print("\nNo data collected for joint probability summary table.")

        # --- Generate and Save CONDITIONAL Probability Summary ---
        if all_files_cond_summary_data:
            print("\n--- Generating Conditional Probability Summary Table ---")
            try:
                cond_summary_df = pd.DataFrame(all_files_cond_summary_data)
                cond_summary_df.set_index('filename', inplace=True)

                # Define column order (can reuse logic, based on columns present in *this* df before reindex)
                phase_numbers_present_cond = sorted(list(set(
                    col.split('-')[0] for col in cond_summary_df.columns if '-' in col and col.split('-')[0] != BACKGROUND_SUMMARY_ABBR
                 )), key=int) # Sort numerically
                summary_column_order_cond = []
                for from_num in phase_numbers_present_cond:
                    for to_num in phase_numbers_present_cond:
                        summary_column_order_cond.append(f"{from_num}-{to_num}")
                    bkg_col_name = f"{from_num}-{BACKGROUND_SUMMARY_ABBR}"
                    if bkg_col_name in cond_summary_df.columns:
                        summary_column_order_cond.append(bkg_col_name)

                # Reindex, fill NaNs/missing with 0.0, and order rows
                cond_summary_df = cond_summary_df.reindex(columns=summary_column_order_cond, fill_value=0.0)
                cond_summary_df = cond_summary_df.reindex(index=processed_files_order)
                cond_summary_df.reset_index(inplace=True)

                # Define the output filename for the conditional summary
                cond_summary_csv_path = os.path.join(output_dir, "all_files_conditional_transition_probabilities_summary_with_bkg.csv")
                cond_summary_df.to_csv(cond_summary_csv_path, index=False, float_format='%.6f')
                print(f"Conditional probability summary table saved to: {cond_summary_csv_path}")

            except Exception as e:
                print(f"Error generating or saving the CONDITIONAL probability summary table: {e}")
                import traceback; traceback.print_exc()
        else:
             print("\nNo data collected for conditional probability summary table.")

    elif not output_dir:
        print("\nSkipping summary table generation as output directory was not specified or couldn't be created.")


if __name__ == "__main__":
    # Basic check for presence of critical libraries
    libs_ok = True
    for lib_name, lib_var in [('Pandas', 'pd'), ('NumPy', 'np')]: # Matplotlib/Seaborn checked earlier
        if lib_var not in globals():
            print(f"Error: Missing critical library: {lib_name}. Please install it.")
            libs_ok = False
    if libs_ok:
         main()
# --- [ END: main function ] ---