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


# --- [ START: NEW compute_transitions_from_sequences function ] ---
def compute_transitions_from_sequences(sequences):
    """
    Computes transitions and behavior counts from a list of behavior sequences.
    Each sequence is a string, e.g., "145", representing observed behaviors.

    Args:
        sequences (list of str): A list of strings, where each string is a sequence of behaviors.
                                 Example: ["145", "123", "1"]

    Returns:
        dict: A dictionary containing:
              'transitions': A dict of dicts (behavior_from -> behavior_to -> count).
              'behavior_counts': A dict (behavior -> count).
    """
    transitions = defaultdict(lambda: defaultdict(int))
    behavior_counts = defaultdict(int)
    total_calculated_transitions = 0

    for seq in sequences:
        if not isinstance(seq, str):
            print(f"Warning: Sequence '{seq}' is not a string. Skipping.")
            continue
        if not seq:  # Skip empty sequences
            continue

        # Count behaviors
        for behavior_char in seq:
            behavior_counts[behavior_char] += 1

        # Count transitions (between adjacent behaviors in the sequence)
        for i in range(len(seq) - 1):
            from_behavior = seq[i]
            to_behavior = seq[i + 1]
            transitions[from_behavior][to_behavior] += 1
            total_calculated_transitions += 1

    # print(f"Debug: Transitions from sequences: {dict(transitions)}")
    # print(f"Debug: Behavior counts from sequences: {dict(behavior_counts)}")

    return {
        'transitions': dict(transitions),
        'behavior_counts': dict(behavior_counts)
    }


# --- [ END: NEW compute_transitions_from_sequences function ] ---


# --- [ create_transition_matrices function is unchanged ] ---
def create_transition_matrices(transition_data, behaviors):
    """
    Create raw count, conditional probability, and joint probability transition matrices
    for transitions *between* the specified behaviors.

    Args:
        transition_data (dict): Dictionary with transition counts (behavior -> behavior)
                                and behavior counts. Keys are original behavior names.
        behaviors (list): List of *original* behavior names (e.g. characters '1', '2')
                          to include as both source (rows) and destination (columns).

    Returns:
        tuple: (counts_matrix, conditional_probability_matrix, joint_probability_matrix)
               Matrices index = original behavior names.
               Matrices columns = original behavior names.
    """
    counts_matrix = pd.DataFrame(0, index=behaviors, columns=behaviors)
    conditional_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)
    joint_probability_matrix = pd.DataFrame(0.0, index=behaviors, columns=behaviors)

    total_transitions = 0
    # Ensure transition_data['transitions'] exists and is a dictionary
    if 'transitions' in transition_data and isinstance(transition_data['transitions'], dict):
        for from_behavior, to_dict in transition_data['transitions'].items():
            if from_behavior in behaviors:  # from_behavior should be a string key
                if isinstance(to_dict, dict):
                    for to_behavior, count in to_dict.items():
                        if to_behavior in behaviors:  # to_behavior should be a string key
                            counts_matrix.loc[str(from_behavior), str(to_behavior)] = count  # Ensure keys are strings
                            total_transitions += count
                else:
                    print(
                        f"Warning: Expected dictionary for transitions from '{from_behavior}', got {type(to_dict)}. Skipping.")
    else:
        print(
            f"Warning: 'transitions' key missing or not a dict in transition_data. Transition data: {transition_data}")

    conditional_probability_matrix = counts_matrix.copy().astype(float)
    for from_behavior in behaviors:
        from_b_str = str(from_behavior)  # Ensure string key
        row_sum = conditional_probability_matrix.loc[from_b_str].sum()
        if row_sum > 0:
            conditional_probability_matrix.loc[from_b_str] = conditional_probability_matrix.loc[from_b_str] / row_sum

    if total_transitions > 0:
        joint_probability_matrix = counts_matrix.astype(float) / total_transitions
    else:
        joint_probability_matrix = pd.DataFrame(0.0, index=behaviors,
                                                columns=behaviors)  # Ensure string keys for index/cols

    # Ensure string type for index and columns if not already
    counts_matrix.index = counts_matrix.index.astype(str)
    counts_matrix.columns = counts_matrix.columns.astype(str)
    conditional_probability_matrix.index = conditional_probability_matrix.index.astype(str)
    conditional_probability_matrix.columns = conditional_probability_matrix.columns.astype(str)
    joint_probability_matrix.index = joint_probability_matrix.index.astype(str)
    joint_probability_matrix.columns = joint_probability_matrix.columns.astype(str)

    return counts_matrix, conditional_probability_matrix, joint_probability_matrix


# --- [ plot_heatmap function is unchanged (using the corrected version provided) ] ---
def plot_heatmap(matrix, title, matrix_type='conditional_prob', output_path=None, vmin=None, vmax=None, ax=None,
                 add_colorbar=False, title_fontsize=14, title_fontweight='bold'):
    if sns is None:
        print("Seaborn not available, skipping heatmap generation.")
        return None, None
    if ax is None:
        base_size = max(6, matrix.shape[0] * 0.8)
        fig, ax = plt.subplots(figsize=(base_size, base_size))
        standalone_plot = True;
        add_colorbar_in_heatmap = True
    else:
        standalone_plot = False;
        fig = ax.figure;
        add_colorbar_in_heatmap = add_colorbar
    fmt = ".2f";
    cmap = "YlGnBu";
    cbar_label = "Probability";
    auto_vmin = 0.0;
    auto_vmax = 1.0
    matrix_values_finite = matrix.values[np.isfinite(matrix.values)] if matrix.size > 0 else np.array([])
    if matrix_type == 'count':
        fmt = ".0f";
        cbar_label = "Transition Count"
        auto_vmax = np.max(matrix_values_finite) if matrix_values_finite.size > 0 else 1.0;
        auto_vmin = 0
    elif matrix_type == 'conditional_prob':
        cbar_label = "Conditional Probability P(To|From)"
        auto_vmax = np.max(matrix_values_finite) if matrix_values_finite.size > 0 else 1.0
        auto_vmax = max(auto_vmax, 1e-9);
        auto_vmin = 0.0
    elif matrix_type == 'joint_prob':
        cbar_label = "Joint Probability P(From, To)";
        fmt = ".3f"
        auto_vmax = np.max(matrix_values_finite) if matrix_values_finite.size > 0 else 1e-9
        auto_vmin = 0.0
    final_vmin = vmin if vmin is not None else auto_vmin
    final_vmax = vmax if vmax is not None else auto_vmax
    if final_vmax <= final_vmin: final_vmax = final_vmin + 1e-6
    g = sns.heatmap(matrix, annot=True, cmap=cmap, vmin=final_vmin, vmax=final_vmax, fmt=fmt, linewidths=.5,
                    cbar=add_colorbar_in_heatmap, cbar_kws={'label': cbar_label} if add_colorbar_in_heatmap else {},
                    ax=ax, annot_kws={"size": 10}, square=True)
    ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight)
    ax.set_ylabel("From Phase");
    ax.set_xlabel("To Phase")
    ax.tick_params(axis='x', rotation=45, labelsize=10);
    ax.tick_params(axis='y', rotation=0, labelsize=10)
    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    if standalone_plot:
        try:
            fig.tight_layout(pad=1.0)
        except ValueError:
            print("Warning: tight_layout failed, plot margins might be suboptimal.")
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight'); plt.close(fig); print(
                f"Heatmap saved to: {output_path}")
        else:
            plt.close(fig)
    mappable = ax.collections[0] if ax.collections else None
    return ax, mappable


# --- [ START: MODIFIED process_group_files function for TXT sequence files ] ---
def process_group_files(directory, group_suffix, output_dir=None):  # max_gap removed
    """
    Process all TXT files (sequence per line) for a specific experimental group.

    Args:
        directory (str): Directory containing TXT files.
        group_suffix (str): Suffix identifying the experimental group (e.g., "_veh.txt").
        output_dir (str, optional): Directory to save output files (currently not used here but kept for consistency).

    Returns:
        dict: Aggregated transition data for the group {'transitions': ..., 'behavior_counts': ...}.
    """
    group_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(group_suffix.lower()):  # Ensure suffix matching is case-insensitive
                full_path = os.path.join(root, file)
                if os.path.isfile(full_path):
                    group_files.append(full_path)

    print(
        f"Found {len(group_files)} files for group suffix: '{group_suffix}' in directory '{directory}' and subdirectories.")

    group_transitions = defaultdict(lambda: defaultdict(int))
    group_behavior_counts = defaultdict(int)
    total_files_processed = 0
    all_sequences_for_group = []

    for file_path in group_files:
        try:
            with open(file_path, 'r') as f:
                sequences_in_file = [line.strip() for line in f if line.strip()]  # Read non-empty lines

            if not sequences_in_file:
                print(f"Warning: Skipping empty or whitespace-only file {file_path}")
                continue

            all_sequences_for_group.extend(sequences_in_file)
            total_files_processed += 1

        except FileNotFoundError:
            print(f"Error: File not found {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")

    if not all_sequences_for_group:
        print(f"No valid sequences found for group {group_suffix} across all files.")
        return {'transitions': {}, 'behavior_counts': {}}

    # Compute transitions for all sequences in the group at once
    # The `max_gap` argument is not relevant for direct sequence transitions
    transitions_data = compute_transitions_from_sequences(all_sequences_for_group)

    for from_behavior, to_behaviors in transitions_data['transitions'].items():
        for to_behavior, count in to_behaviors.items():
            group_transitions[from_behavior][to_behavior] += count
    for behavior, count in transitions_data['behavior_counts'].items():
        group_behavior_counts[behavior] += count

    print(f"Finished processing. Total files successfully processed for group {group_suffix}: {total_files_processed}")
    if total_files_processed > 0:
        total_group_transitions = sum(sum(to_dict.values()) for to_dict in group_transitions.values())
        print(f"Total aggregated transitions for group {group_suffix}: {total_group_transitions}")

    # Output directory creation (if needed for other outputs, though not used directly in this func now)
    if output_dir:
        try:
            if not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}"); output_dir = None

    return {'transitions': dict(group_transitions), 'behavior_counts': dict(group_behavior_counts)}


# --- [ END: MODIFIED process_group_files function ] ---


# --- [ START: MODIFIED main function ] ---
def main():
    # --- Mappings and Display Orders ---
    # IMPORTANT: Keys here should be STRINGS if your sequences use digits, e.g., '1', '2'
    BEHAVIOR_TO_PHASE_MAP = {
        '1': 'Phase 1', '2': 'Phase 2', '3': 'Phase 3',
        '4': 'Phase 4', '5': 'Phase 5'
        # Add more if your sequences use other characters/digits
    }
    # Ensure this order matches the phases you want to display
    PHASE_ORDER = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5']
    GROUP_DISPLAY_NAME_MAP = {'veh': 'Vehicle', 'skf': 'SKF-38393', 'rop': 'Ropinirole'}
    # DEFAULT_BEHAVIORS are now the characters/digits used in your sequence files
    DEFAULT_BEHAVIORS = list(BEHAVIOR_TO_PHASE_MAP.keys())  # e.g., ['1', '2', '3', '4', '5']

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='Compute and visualize behavior transition probabilities from sequence TXT files. Uses Phase 1-5 labels.')
    parser.add_argument('directory', help='Directory containing TXT sequence files (searches recursively)')
    # max-gap is no longer directly used by the transition counting for sequences, but kept for now if other parts might use it.
    # Consider removing if it's truly unused.
    parser.add_argument('--max-gap', type=int, default=60,
                        help='NOT USED for sequence input. (Original: Maximum frame gap for transitions)')
    parser.add_argument('--output-dir',
                        help='Directory to save output files (defaults to DIRECTORY/transition_analysis_sequences)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output')
    parser.add_argument('--groups', nargs='+', default=['_veh.txt', '_skf.txt', '_rop.txt'],
                        help='List of file suffixes identifying the groups (e.g., _veh.txt _drugA.txt)')
    parser.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS,
                        help=f'List of behavior characters (from sequence files) to include in the analysis. Defaults to: {DEFAULT_BEHAVIORS}. These will be mapped to Phases.')
    args = parser.parse_args()

    # --- Setup ---
    groups = args.groups
    behaviors_to_process = [str(b) for b in args.behaviors]  # Ensure behaviors are strings
    print(f"Analyzing transitions BETWEEN behaviors (characters from sequences): {behaviors_to_process}")
    print(f"Looking for group suffixes: {groups}")
    if args.max_gap != 60: print("Note: --max-gap argument is not used for sequence-based input.")

    output_dir = args.output_dir or os.path.join(args.directory, 'transition_analysis_sequences')
    try:
        if not os.path.exists(output_dir): os.makedirs(output_dir); print(f"Created output directory: {output_dir}")
    except OSError as e:
        print(
            f"Error: Could not create output directory {output_dir}. Output will not be saved. Error: {e}"); output_dir = None
    if sns is None and output_dir: print("Warning: Seaborn is not installed. Heatmaps cannot be generated.")

    # --- Group processing loop ---
    all_group_data = {}
    group_matrices_renamed = {}

    for group_suffix in groups:
        print(f"\n--- Processing Group: {group_suffix} ---")
        # Derive group name from suffix (e.g., "_veh.txt" -> "veh")
        group_name_derived = group_suffix.lower().replace('.txt', '').strip('_')
        display_group_name = GROUP_DISPLAY_NAME_MAP.get(group_name_derived, group_name_derived.capitalize())

        # Pass output_dir to process_group_files for potential internal use, though not used for reading here
        group_data = process_group_files(args.directory, group_suffix, output_dir=output_dir)

        # Ensure behaviors_to_process contains string representations for matrix creation
        str_behaviors_to_process = [str(b) for b in behaviors_to_process]
        counts_matrix_orig, cond_prob_matrix_orig, joint_prob_matrix_orig = create_transition_matrices(group_data,
                                                                                                       str_behaviors_to_process)

        total_group_transitions = counts_matrix_orig.values.sum()

        if total_group_transitions == 0:
            print(
                f"Warning: No transitions found *between specified behaviors* {str_behaviors_to_process} for group: {display_group_name}. Skipping plots for this group.")
            all_group_data[display_group_name] = group_data;
            group_matrices_renamed[display_group_name] = None
            continue

        all_group_data[display_group_name] = group_data

        # --- Rename and Reorder (ensure keys in rename_map are strings) ---
        rename_map = {orig_char: phase for orig_char, phase in BEHAVIOR_TO_PHASE_MAP.items() if
                      str(orig_char) in str_behaviors_to_process}

        # Get current labels (they should be strings from matrix creation)
        current_labels_from_matrix = counts_matrix_orig.index.astype(str).tolist()

        # Map them to phase names, or keep original if not in map
        current_phase_labels = [rename_map.get(b, b) for b in current_labels_from_matrix]

        # Desired order of PHASES, plus any other labels not in PHASE_ORDER but present
        desired_order_of_phases = [p for p in PHASE_ORDER if p in current_phase_labels]
        other_labels_in_order = [lbl for lbl in current_phase_labels if lbl not in desired_order_of_phases]
        final_display_order = desired_order_of_phases + other_labels_in_order

        counts_matrix = counts_matrix_orig.rename(index=rename_map, columns=rename_map)
        cond_prob_matrix = cond_prob_matrix_orig.rename(index=rename_map, columns=rename_map)
        joint_prob_matrix = joint_prob_matrix_orig.rename(index=rename_map, columns=rename_map)

        # Reindex using the final_display_order (which contains phase names)
        # Ensure all labels in final_display_order are present in the renamed matrices' indices/columns
        valid_final_order = [lbl for lbl in final_display_order if lbl in counts_matrix.index]

        counts_matrix = counts_matrix.reindex(index=valid_final_order, columns=valid_final_order, fill_value=0)
        cond_prob_matrix = cond_prob_matrix.reindex(index=valid_final_order, columns=valid_final_order, fill_value=0.0)
        joint_prob_matrix = joint_prob_matrix.reindex(index=valid_final_order, columns=valid_final_order,
                                                      fill_value=0.0)
        # --- End Rename/Reorder ---

        group_matrices_renamed[display_group_name] = {'counts': counts_matrix, 'conditional': cond_prob_matrix,
                                                      'joint': joint_prob_matrix}

        # --- Save and Plot Individual ---
        if output_dir and sns:
            try:
                safe_group_name = group_name_derived  # e.g., "veh"
                plot_heatmap(counts_matrix, title=display_group_name, matrix_type='count',
                             output_path=os.path.join(output_dir, f"transition_counts_heatmap_{safe_group_name}.png"))
                counts_matrix.to_csv(os.path.join(output_dir, f"transition_counts_{safe_group_name}.csv"))
                plot_heatmap(cond_prob_matrix, title=display_group_name, matrix_type='conditional_prob',
                             output_path=os.path.join(output_dir,
                                                      f"transition_conditional_probabilities_heatmap_{safe_group_name}.png"))
                cond_prob_matrix.to_csv(
                    os.path.join(output_dir, f"transition_conditional_probabilities_{safe_group_name}.csv"))
                plot_heatmap(joint_prob_matrix, title=display_group_name, matrix_type='joint_prob',
                             output_path=os.path.join(output_dir,
                                                      f"transition_joint_probabilities_heatmap_{safe_group_name}.png"))
                joint_prob_matrix.to_csv(
                    os.path.join(output_dir, f"transition_joint_probabilities_{safe_group_name}.csv"))
                print(f"Saved CSVs and heatmaps for group {display_group_name}")
            except Exception as e:
                print(f"Error during saving files/plots for group {display_group_name}: {e}")

        # --- Print summary ---
        print(f"\n--- Summary for Group: {display_group_name} ---")
        print("Behavior occurrences (from sequences):")
        # Iterate over behaviors_to_process (which are original char labels)
        for behavior_char in str_behaviors_to_process:
            display_label = rename_map.get(behavior_char, behavior_char)  # Get phase name if mapped
            count = group_data['behavior_counts'].get(behavior_char, 0)
            print(f"  {display_label} (raw: {behavior_char}): {count}")

        actual_total_transitions = counts_matrix.values.sum()
        print(f"\nTotal Transitions Counted BETWEEN Analyzed Behaviors: {actual_total_transitions}")
        print("\nTransition Counts Matrix (Phases):");
        print(counts_matrix)
        print("\nConditional Transition Probability Matrix P(To|From) (Phases):");
        print(cond_prob_matrix.round(3))
        print("\nJoint Transition Probability Matrix P(From, To) (Phases):");
        print(joint_prob_matrix.round(4))
        print(f"(Sum of joint probabilities: {joint_prob_matrix.values.sum():.4f})")

    # --- [ Group Comparison and Combined Figures (adjusting for square plots) ] ---
    # This section should largely work as is, assuming group_matrices_renamed is populated correctly
    valid_groups_for_plotting = [name for name, matrices in group_matrices_renamed.items() if
                                 matrices is not None and not matrices['counts'].empty]
    num_valid_groups = len(valid_groups_for_plotting)

    if sns and output_dir and num_valid_groups > 0:
        print(f"\n--- Generating Combined SQUARE Heatmaps with CONSISTENT SCALES for {num_valid_groups} groups ---")
        plot_types = [
            ('counts', 'Transition Counts', 'combined_transition_counts_heatmaps_scaled.png', '.0f', 'count'),
            ('conditional', 'Conditional Probability P(To|From)',
             'combined_transition_conditional_probabilities_heatmaps_scaled.png', '.2f', 'conditional_prob'),
            ('joint', 'Joint Probability P(From, To)', 'combined_transition_joint_probabilities_heatmaps_scaled.png',
             '.3f', 'joint_prob')
        ]
        target_subplot_height_inches = 5.0;
        target_subplot_width_inches = target_subplot_height_inches
        top_margin_inches = 1.0;
        bottom_margin_inches = 1.2;
        colorbar_width_inches = 0.5
        inter_plot_padding = 0.8;
        side_padding = 0.5
        total_fig_width = (target_subplot_width_inches * num_valid_groups) + (
                    inter_plot_padding * (num_valid_groups - 1)) + colorbar_width_inches + side_padding * 2 + 0.5
        total_fig_height = target_subplot_height_inches + top_margin_inches + bottom_margin_inches

        for matrix_key, plot_title_prefix, filename, fmt_str, matrix_plot_type in plot_types:
            global_vmin = 0.0;
            global_vmax = 0.0;
            all_max_values = []
            for group_name in valid_groups_for_plotting:
                if group_matrices_renamed[group_name] and matrix_key in group_matrices_renamed[group_name]:
                    matrix = group_matrices_renamed[group_name][matrix_key]
                    if not matrix.empty:
                        matrix_values_finite = matrix.values[
                            np.isfinite(matrix.values)] if matrix.size > 0 else np.array([])
                        if matrix_values_finite.size > 0: all_max_values.append(np.max(matrix_values_finite))
            if not all_max_values:
                global_vmax = 1.0 if matrix_plot_type != 'count' else 1
            else:
                global_vmax = max(all_max_values) if all_max_values else (
                    1.0 if matrix_plot_type != 'count' else 1)  # Default if no values
            if matrix_plot_type == 'count':
                global_vmax = max(global_vmax, 1); global_vmin = 0.0
            elif matrix_plot_type == 'conditional_prob':
                global_vmax = max(global_vmax, 1e-9); global_vmin = 0.0
            elif matrix_plot_type == 'joint_prob':
                global_vmax = max(global_vmax, 1e-9); global_vmin = 0.0
            if global_vmax <= global_vmin: global_vmax = global_vmin + 1e-6
            print(f"Combined plot for '{matrix_key}': Using scale vmin={global_vmin:.3f}, vmax={global_vmax:.3f}")

            fig, axes = plt.subplots(1, num_valid_groups, figsize=(total_fig_width, total_fig_height), squeeze=False)
            fig.suptitle(f"Combined Heatmaps: {plot_title_prefix}", fontsize=18, fontweight='bold')
            last_mappable = None

            for i, group_name in enumerate(valid_groups_for_plotting):
                ax = axes[0, i];
                matrix = pd.DataFrame()  # Default to empty
                if group_matrices_renamed[group_name] and matrix_key in group_matrices_renamed[group_name]:
                    matrix = group_matrices_renamed[group_name][matrix_key]

                if matrix.empty:
                    # Optionally plot an empty placeholder or skip
                    ax.set_title(f"{group_name}\n(No data)", fontsize=14, fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    print(f"Skipping heatmap for {group_name} - {matrix_key} due to empty matrix.")
                    continue

                _, mappable = plot_heatmap(matrix, title=group_name, matrix_type=matrix_plot_type, vmin=global_vmin,
                                           vmax=global_vmax, ax=ax, add_colorbar=False, title_fontsize=14,
                                           title_fontweight='bold')
                if mappable: last_mappable = mappable
                if i > 0: ax.set_ylabel(""); ax.tick_params(axis='y', which='both', left=False, labelleft=False)

            if last_mappable:
                last_ax = axes[0, num_valid_groups - 1];
                divider = make_axes_locatable(last_ax)
                cax_width = f"{colorbar_width_inches / target_subplot_width_inches * 100:.1f}%"
                cax = divider.append_axes("right", size=cax_width, pad=0.2)
                cbar_label_text = "Count" if matrix_plot_type == 'count' else (
                    "Cond. Prob. P(To|From)" if matrix_plot_type == 'conditional_prob' else "Joint Prob. P(From, To)")
                fig.colorbar(last_mappable, cax=cax, label=cbar_label_text)
            else:
                print(f"Warning: Could not add colorbar for combined {matrix_key} plot (no valid data plotted).")

            try:
                fig.tight_layout(rect=[0.03, bottom_margin_inches / total_fig_height * 0.8, 0.97,
                                       1 - (top_margin_inches / total_fig_height * 0.9)])
                combined_path = os.path.join(output_dir, filename)
                fig.savefig(combined_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"Combined {matrix_key} heatmap saved to: {combined_path}")
            except Exception as e:
                print(f"\nError saving combined {matrix_key} heatmap: {e}")
                plt.close(fig)

    # --- [ Debug comparison section (unchanged logic but check matrix keys) ] ---
    if args.debug and num_valid_groups > 1:
        print("\n---- Debug Group Comparison (Transition Counts) ----")
        common_labels_ph_or_raw = set();
        first_group_processed = True;
        ref_order_ph_or_raw = []

        # Determine common labels based on the *renamed* matrices (phase labels)
        temp_first_matrix = None
        for group_name in valid_groups_for_plotting:
            if group_matrices_renamed[group_name] and 'counts' in group_matrices_renamed[group_name] and not \
            group_matrices_renamed[group_name]['counts'].empty:
                matrix = group_matrices_renamed[group_name]['counts']
                current_labels = set(matrix.index.astype(str))  # Ensure string labels
                if first_group_processed:
                    common_labels_ph_or_raw = current_labels
                    ref_order_ph_or_raw = matrix.index.astype(str).tolist()
                    temp_first_matrix = matrix  # store for ref_order if only one group has data
                    first_group_processed = False
                else:
                    common_labels_ph_or_raw.intersection_update(current_labels)

        if not first_group_processed and not common_labels_ph_or_raw and temp_first_matrix is not None:  # Only one group had data
            common_labels_ph_or_raw = set(temp_first_matrix.index.astype(str))
            ref_order_ph_or_raw = temp_first_matrix.index.astype(str).tolist()

        if not ref_order_ph_or_raw:
            print("No common phases/behaviors found across groups with data to compare.")
        else:
            final_ref_order = [lbl for lbl in ref_order_ph_or_raw if lbl in common_labels_ph_or_raw]
            if not final_ref_order: final_ref_order = ref_order_ph_or_raw  # Fallback if intersection was empty but had one group

            print(f"Comparing common transitions between states: {final_ref_order}")
            for label_from in final_ref_order:
                for label_to in final_ref_order:
                    # Check if this transition exists in *any* group
                    transition_exists_debug = any(
                        group_matrices_renamed[name] is not None and
                        'counts' in group_matrices_renamed[name] and
                        not group_matrices_renamed[name]['counts'].empty and
                        label_from in group_matrices_renamed[name]['counts'].index and
                        label_to in group_matrices_renamed[name]['counts'].columns and
                        group_matrices_renamed[name]['counts'].loc[label_from, label_to] > 0
                        for name in valid_groups_for_plotting
                    )
                    if transition_exists_debug:
                        print(f"\nTransitions {label_from} -> {label_to}:")
                        for group_name in valid_groups_for_plotting:
                            count = 0
                            try:
                                matrix_dict = group_matrices_renamed[group_name]
                                if matrix_dict is not None and 'counts' in matrix_dict and not matrix_dict[
                                    'counts'].empty:
                                    if label_from in matrix_dict['counts'].index and label_to in matrix_dict[
                                        'counts'].columns:
                                        count = matrix_dict['counts'].loc[label_from, label_to]
                            except KeyError:
                                count = 0  # Should be caught by `in` checks
                            except Exception as e_debug:
                                print(
                                    f"  Error getting count for {group_name} {label_from}->{label_to}: {e_debug}"); count = 0
                            print(f"  {group_name}: {int(count)}")

    print("\n--- Analysis Complete ---")


if __name__ == "__main__":
    if 'pd' not in globals() or 'np' not in globals() or 'matplotlib' not in globals() or 'plt' not in globals():
        print("Error: Missing critical libraries (Pandas, NumPy, Matplotlib). Please install them.")
    else:
        main()