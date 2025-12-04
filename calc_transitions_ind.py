# --- [Includes and other functions remain the same] ---
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
    # Optionally try a default backend if Agg fails but matplotlib is installed
    pass # Allow matplotlib to use its default backend

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
import fnmatch # For pattern matching filenames

# --- [compute_transitions_improved function remains the same] ---
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

        try:
            # Ensure the column is numeric-like before converting to int
            if pd.api.types.is_numeric_dtype(df[behavior]):
                 behavior_series = df[behavior].astype(int)
            else:
                 # Attempt conversion, warn if it fails
                 behavior_series = pd.to_numeric(df[behavior], errors='coerce').fillna(0).astype(int)
                 print(f"Warning: Column '{behavior}' was not strictly numeric. Coerced non-numeric values to 0.")

            transitions = np.diff(np.hstack([[0], behavior_series.values, [0]]))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0] - 1

            if len(starts) == 0 or len(ends) == 0:
                continue

            min_len = min(len(starts), len(ends))
            starts = starts[:min_len]
            ends = ends[:min_len]

            for i in range(len(starts)):
                # Check index bounds rigorously
                if 0 <= starts[i] < len(df) and 0 <= ends[i] < len(df) and starts[i] <= ends[i]:
                    start_frame = int(df.iloc[starts[i]][frame_col])
                    end_frame = int(df.iloc[ends[i]][frame_col])
                    behavior_blocks[behavior].append((start_frame, end_frame))
                else:
                    # More informative warning
                    print(f"Warning: Invalid indices start={starts[i]} (valid range 0-{len(df)-1}), end={ends[i]} (valid range 0-{len(df)-1}) for behavior '{behavior}'. Skipping block.")
        except KeyError:
             print(f"Warning: Column '{behavior}' not found during processing. Skipping.")
        except Exception as e:
            print(f"Error processing blocks for behavior '{behavior}': {e}. Skipping behavior.")


    behavior_counts = {behavior: len(blocks) for behavior, blocks in behavior_blocks.items()}
    transitions = defaultdict(lambda: defaultdict(int))
    total_calculated_transitions = 0

    for behavior_from, blocks_from in behavior_blocks.items():
        for start_from, end_from in blocks_from:
            for behavior_to, blocks_to in behavior_blocks.items():
                if behavior_from == behavior_to:
                    continue
                for start_to, end_to in blocks_to:
                    # Ensure frames are comparable (e.g., numeric)
                    try:
                        gap = int(start_to) - int(end_from)
                        if 0 < gap <= max_gap:
                            transitions[behavior_from][behavior_to] += 1
                            total_calculated_transitions += 1
                    except (ValueError, TypeError) as e:
                         print(f"Warning: Could not calculate gap between frames ({end_from}, {start_to}). Error: {e}. Skipping transition check.")


    return {
        'transitions': dict(transitions),
        'behavior_counts': behavior_counts,
        'total_transitions': total_calculated_transitions # Also return total for easy checking
    }


# --- [create_transition_matrices function remains the same] ---
def create_transition_matrices(transition_data, behaviors):
    """
    Create raw count, conditional probability, and joint probability transition matrices.
    Uses original behavior names for indexing.

    Args:
        transition_data (dict): Dictionary with transition counts and behavior counts
        behaviors (list): List of *original* behavior names (column headers) to include

    Returns:
        tuple: (counts_matrix, conditional_probability_matrix, joint_probability_matrix)
               Matrices are indexed/columned by the original behavior names. Returns None if no transitions.
    """
    total_transitions = transition_data.get('total_transitions', 0) # Get total calculated in compute_transitions_improved

    if total_transitions == 0:
        print("Warning: No transitions found in the data. Returning empty matrices.")
        empty_df = pd.DataFrame(0, index=behaviors, columns=behaviors)
        empty_df_float = pd.DataFrame(0.0, index=behaviors, columns=behaviors)
        return empty_df, empty_df_float, empty_df_float

    counts_matrix = pd.DataFrame(0, index=behaviors, columns=behaviors)
    conditional_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)
    joint_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)

    # Fill in the transition counts using original behavior names
    for from_behavior, to_behaviors in transition_data.get('transitions', {}).items():
        if from_behavior in behaviors: # Check if the 'from' behavior is one we are analyzing
            for to_behavior, count in to_behaviors.items():
                if to_behavior in behaviors: # Check if the 'to' behavior is one we are analyzing
                    # Use .loc for safe assignment
                    counts_matrix.loc[from_behavior, to_behavior] = count

    # Calculate conditional probability (rows sum to 1)
    # Use fillna(0) in case a behavior never occurs as 'from'
    row_sums = counts_matrix.sum(axis=1).replace(0, np.nan) # Avoid division by zero, replace 0 with NaN
    conditional_probability_matrix = counts_matrix.div(row_sums, axis=0).fillna(0.0)


    # Calculate joint probability (all cells sum to 1)
    # Recalculate total_transitions from the matrix to be sure it only includes specified behaviors
    total_matrix_transitions = counts_matrix.values.sum()
    if total_matrix_transitions > 0:
        joint_probability_matrix = counts_matrix.astype(float) / total_matrix_transitions
    else:
        # Handle case where transitions occurred but not between the specified behaviors
        joint_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)


    # Return matrices with original behavior names as index/columns
    return counts_matrix, conditional_probability_matrix, joint_probability_matrix


# --- [plot_heatmap function remains the same] ---
def plot_heatmap(matrix, title, matrix_type='conditional_prob', output_path=None, vmin=None, vmax=None, ax=None, add_colorbar=False):
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
         return None, None # Return None for ax and mappable

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        standalone_plot = True
        add_colorbar_in_heatmap = True # Let heatmap add cbar for standalone
    else:
        standalone_plot = False
        fig = ax.figure
        add_colorbar_in_heatmap = add_colorbar # Use flag for existing axes

    fmt = ".2f"
    cmap = "YlGnBu" # Consider "viridis", "plasma", "magma" as alternatives
    cbar_label = "Probability"
    auto_vmin = 0.0
    auto_vmax = 1.0

    # Use np.nanmax to handle potential NaNs if matrix comes from reindexing or empty data
    if matrix.empty:
        print(f"Warning: Matrix for '{title}' is empty. Cannot generate heatmap.")
        if standalone_plot: plt.close(fig)
        return ax, None # Return the axis but no mappable

    finite_values = matrix.values[np.isfinite(matrix.values)] # Get only finite values

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
            cbar=add_colorbar_in_heatmap, # Control based on flag
            cbar_kws={'label': cbar_label} if add_colorbar_in_heatmap else {},
            ax=ax
        )

        ax.set_title(title, wrap=True) # Wrap title if too long
        # These labels are now dynamically set based on matrix index/columns (e.g., 'Phase X')
        ax.set_ylabel(f"From {matrix.index.name if matrix.index.name else 'Behavior'}")
        ax.set_xlabel(f"To {matrix.columns.name if matrix.columns.name else 'Behavior'}")
        # Rotate x-axis labels if they are long (like 'Phase X')
        ax.tick_params(axis='x', rotation=45, labelsize=8) # Adjust label size
        ax.tick_params(axis='y', labelsize=8) # Adjust label size
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
            plt.close(fig) # Close after showing or if not saving

    mappable = ax.collections[0] if ax.collections else None
    return ax, mappable


def main():
    # Define the standard mapping and display order *before* parsing args
    BEHAVIOR_TO_PHASE_MAP = {
        'nose': 'Phase 1',
        'whiskers': 'Phase 2',
        'eyes': 'Phase 3',
        'ears': 'Phase 4',
        'body': 'Phase 5'
    }
    PHASE_DISPLAY_ORDER = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
    DEFAULT_BEHAVIORS = list(BEHAVIOR_TO_PHASE_MAP.keys())

    parser = argparse.ArgumentParser(description='Compute and visualize behavior transition counts and probabilities (conditional and joint) for *each* input CSV file. Uses Phase 1-5 labels for standard behaviors.')
    parser.add_argument('directory', help='Directory containing CSV files (searches recursively)')
    parser.add_argument('--file-pattern', default='*.csv',
                        help='Pattern to match files within the directory (e.g., "*.csv", "*_veh.csv"). Default: *.csv')
    parser.add_argument('--max-gap', type=int, default=60, help='Maximum frame gap for transitions (default: 60)')
    parser.add_argument('--output-dir', help='Directory to save output files (defaults to DIRECTORY/individual_transition_analysis)')
    parser.add_argument('--generate-heatmaps', action='store_true',
                        help='Generate and save individual heatmap PNG files for each matrix type (counts, conditional, joint).')
    parser.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS,
                        help=f'List of original behavior names (column headers) to include in the analysis. Defaults to: {DEFAULT_BEHAVIORS}. Only these defaults will be renamed to Phase 1-5.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output (prints matrices to console).')

    args = parser.parse_args()

    # These are the *original* behavior names requested for analysis from the CSVs
    behaviors_to_process = args.behaviors
    print(f"Analyzing original behaviors: {behaviors_to_process}")
    print(f"Looking for files matching pattern: '{args.file_pattern}' in directory '{args.directory}'")

    # Determine output directory
    output_dir = args.output_dir or os.path.join(args.directory, 'individual_transition_analysis')
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir}. Output will not be saved. Error: {e}")
        output_dir = None # Disable saving if directory fails

    if args.generate_heatmaps and sns is None and output_dir:
         print("Warning: Seaborn is not installed. Heatmaps cannot be generated (--generate-heatmaps ignored).")

    # --- Find and process individual files ---
    files_processed_count = 0
    files_found = []
    all_files_summary_data = [] # <--- Initialize list for summary table data

    for root, _, files in os.walk(args.directory):
        for filename in files:
            if fnmatch.fnmatch(filename, args.file_pattern):
                 files_found.append(os.path.join(root, filename))

    print(f"Found {len(files_found)} files matching pattern '{args.file_pattern}'.")

    # Get phase numbers for summary column headers
    phase_numbers = [p.split(' ')[1] for p in PHASE_DISPLAY_ORDER if p in [BEHAVIOR_TO_PHASE_MAP.get(b, b) for b in behaviors_to_process]]


    for file_path in files_found:
        print(f"\n--- Processing File: {os.path.basename(file_path)} ---")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                 print("Warning: File is empty. Skipping.")
                 continue
            if len(df.columns) < 2:
                print(f"Warning: Skipping {os.path.basename(file_path)} - Less than 2 columns found.")
                continue
            # Basic check for frame column validity
            if not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                 print(f"Warning: First column in {os.path.basename(file_path)} does not appear to be numeric frame numbers. Check file format. Attempting to proceed.")
                 # Could add df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce') here if needed

            # Compute transitions for this file
            transitions_data = compute_transitions_improved(df, args.max_gap)

            if transitions_data['total_transitions'] == 0:
                print("Warning: No transitions found within the max_gap. Skipping matrix generation for this file.")
                # Optionally print behavior counts even if no transitions
                print("Behavior occurrences (number of times behavior block appears):")
                # Define rename_map locally for this case too
                local_rename_map = {orig: phase for orig, phase in BEHAVIOR_TO_PHASE_MAP.items() if orig in behaviors_to_process}
                for behavior in behaviors_to_process:
                     count = transitions_data['behavior_counts'].get(behavior, 0)
                     # Get the display name (Phase X or original name if not mapped)
                     display_name = local_rename_map.get(behavior, behavior)
                     print(f"  {display_name}: {count}")
                continue # Skip to next file

            # Create transition matrices using *original* behavior names
            counts_matrix_orig, cond_prob_matrix_orig, joint_prob_matrix_orig = create_transition_matrices(
                transitions_data, behaviors_to_process
            )

            # --- Rename and Reorder Matrix Index/Columns ---
            rename_map = {orig: phase for orig, phase in BEHAVIOR_TO_PHASE_MAP.items() if orig in behaviors_to_process}
            output_labels_unordered = [rename_map.get(b, b) for b in behaviors_to_process]
            # Ensure ordered labels only contain phases present in the *selected* behaviors
            output_labels_ordered = [p for p in PHASE_DISPLAY_ORDER if p in output_labels_unordered]

            # Rename
            counts_matrix = counts_matrix_orig.rename(index=rename_map, columns=rename_map)
            cond_prob_matrix = cond_prob_matrix_orig.rename(index=rename_map, columns=rename_map)
            joint_prob_matrix = joint_prob_matrix_orig.rename(index=rename_map, columns=rename_map)

            # Reindex to the desired display order, filling missing with 0/0.0
            # Ensure reindexing happens even if some phases are missing from this specific file
            counts_matrix = counts_matrix.reindex(index=output_labels_ordered, columns=output_labels_ordered, fill_value=0)
            cond_prob_matrix = cond_prob_matrix.reindex(index=output_labels_ordered, columns=output_labels_ordered, fill_value=0.0)
            joint_prob_matrix = joint_prob_matrix.reindex(index=output_labels_ordered, columns=output_labels_ordered, fill_value=0.0)

            # --- Save Individual Outputs ---
            if output_dir:
                try:
                    # Save CSVs (now with Phase names and original filename base)
                    counts_path = os.path.join(output_dir, f"{base_filename}_transition_counts.csv")
                    counts_matrix.to_csv(counts_path)

                    joint_prob_path = os.path.join(output_dir, f"{base_filename}_transition_joint_probabilities.csv")
                    joint_prob_matrix.to_csv(joint_prob_path)

                    print(f"Saved counts and joint probability tables for {base_filename}")

                    # Create and save individual heatmaps if requested
                    if args.generate_heatmaps and sns:
                        counts_heatmap_path = os.path.join(output_dir, f"{base_filename}_transition_counts_heatmap.png")
                        plot_heatmap(counts_matrix, f"Transition Counts: {base_filename}",
                                    matrix_type='count', output_path=counts_heatmap_path)

                        cond_prob_heatmap_path = os.path.join(output_dir, f"{base_filename}_transition_conditional_probabilities_heatmap.png")
                        plot_heatmap(cond_prob_matrix, f"Conditional Prob P(To|From): {base_filename}",
                                    matrix_type='conditional_prob', output_path=cond_prob_heatmap_path)

                        joint_prob_heatmap_path = os.path.join(output_dir, f"{base_filename}_transition_joint_probabilities_heatmap.png")
                        plot_heatmap(joint_prob_matrix, f"Joint Prob P(From, To): {base_filename}",
                                    matrix_type='joint_prob', output_path=joint_prob_heatmap_path)
                        print(f"Saved heatmaps for {base_filename}")

                except Exception as e:
                    print(f"Error during saving files/plots for {base_filename}: {e}")

            # --- Collect data for the summary table ---
            file_summary = {'filename': base_filename}
            # Iterate through the *ordered* phase labels to ensure consistent column creation
            for from_phase in output_labels_ordered:
                for to_phase in output_labels_ordered:
                    try:
                        from_num = from_phase.split(' ')[1]
                        to_num = to_phase.split(' ')[1]
                        col_name = f"{from_num}-{to_num}"
                        # Use .loc and handle potential KeyError if reindexing didn't cover all combinations (though it should)
                        prob = joint_prob_matrix.loc[from_phase, to_phase]
                        file_summary[col_name] = prob
                    except (IndexError, KeyError, AttributeError) as e:
                         print(f"Warning: Could not process transition {from_phase} -> {to_phase} for summary table in file {base_filename}. Error: {e}")
                         # Ensure column exists even if data is missing for this file
                         col_name = f"{from_phase.split(' ')[1]}-{to_phase.split(' ')[1]}" # Attempt to create name
                         if col_name not in file_summary:
                             file_summary[col_name] = np.nan # Or 0.0, depending on preference

            all_files_summary_data.append(file_summary)


            # --- Print summary for this file ---
            print(f"Summary for: {base_filename}")
            print("Behavior occurrences (number of times behavior block appears):")
            for behavior in behaviors_to_process:
                count = transitions_data['behavior_counts'].get(behavior, 0)
                display_name = rename_map.get(behavior, behavior) # Use renamed label
                print(f"  {display_name}: {count}")

            print(f"Total Transitions Detected: {transitions_data['total_transitions']}") # Use the calculated total
            print(f"(Sum of joint probabilities in matrix: {joint_prob_matrix.values.sum():.4f})") # Verification

            if args.debug:
                 # Print the RENAMED and REORDERED matrices
                print("\nTransition Counts Matrix:")
                print(counts_matrix)
                # print("\nConditional Transition Probability Matrix P(To|From):")
                # print(cond_prob_matrix.round(3))
                print("\nJoint Transition Probability Matrix P(From, To):")
                print(joint_prob_matrix.round(4))

            files_processed_count += 1

        except pd.errors.EmptyDataError:
             print(f"Warning: Skipping empty file {os.path.basename(file_path)}")
        except FileNotFoundError:
             print(f"Error: File not found {file_path}")
        except Exception as e:
            # Catch broader errors during file processing
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for unexpected errors

    print(f"\n--- Analysis Complete ---")
    print(f"Successfully processed {files_processed_count} out of {len(files_found)} files found.")

    # --- Generate and Save Summary Table ---
    if output_dir and all_files_summary_data:
        print("\n--- Generating Summary Table ---")
        try:
            summary_df = pd.DataFrame(all_files_summary_data)

            # Define the desired order of columns for the summary table
            # Use the phase numbers derived earlier
            summary_column_order = ['filename'] + [f"{f}-{t}" for f in phase_numbers for t in phase_numbers]

            # Reindex to ensure correct column order and presence of all columns, fill missing with 0.0
            # This handles cases where a specific transition (e.g., 5-1) never occurred in any file
            summary_df = summary_df.reindex(columns=summary_column_order, fill_value=0.0)

            # Define output path for the summary CSV
            summary_csv_path = os.path.join(output_dir, "all_files_joint_transition_probabilities_summary.csv")

            # Save the summary DataFrame to CSV
            summary_df.to_csv(summary_csv_path, index=False, float_format='%.6f') # Save with decent precision
            print(f"Summary table saved to: {summary_csv_path}")

        except Exception as e:
            print(f"Error generating or saving the summary transition probability table: {e}")
            import traceback
            traceback.print_exc()
    elif not all_files_summary_data:
        print("\nNo data collected for summary table (perhaps no files had transitions).")


if __name__ == "__main__":
    # --- [ Library availability checks remain the same ] ---
    if 'pd' not in globals() or 'np' not in globals() or 'matplotlib' not in globals() or 'plt' not in globals():
         print("Error: Missing critical libraries (Pandas, NumPy, Matplotlib). Please install them.")
    else:
         main()