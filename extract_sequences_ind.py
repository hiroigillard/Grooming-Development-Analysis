import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict
import traceback

# Define behavior mapping
BEHAVIOR_MAP = {
    'nose': 1,
    'whiskers': 2,
    'eyes': 3,
    'ears': 4,
    'body': 5
}

# Reverse mapping for reference
BEHAVIOR_ID_TO_NAME = {v: k for k, v in BEHAVIOR_MAP.items()}


def extract_behavior_blocks(df, max_gap=60): # max_gap is kept for signature consistency, not used here
    """
    Extract behavior blocks (start and end frames) from the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with frame number and behavior columns
        max_gap (int): (Currently unused in this function but kept for API consistency)

    Returns:
        list: List of (behavior_id, start_frame, end_frame) tuples, sorted by start_frame
    """
    if not df.columns.any():
        print("Warning: DataFrame is empty or has no columns.")
        return []
    frame_col = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[frame_col]):
         print(f"Warning: Frame column '{frame_col}' is not numeric.")

    blocks = []

    for behavior, behavior_id in BEHAVIOR_MAP.items():
        if behavior not in df.columns:
            continue
        if df[behavior].isnull().all():
             continue
        if not pd.api.types.is_numeric_dtype(df[behavior]):
            print(f"Warning: Behavior column '{behavior}' is not numeric. Attempting conversion.")
            try:
                behavior_series = pd.to_numeric(df[behavior], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                print(f"Error converting column '{behavior}' to numeric: {e}. Skipping.")
                continue
        else:
             behavior_series = df[behavior].fillna(0).astype(int)

        if behavior_series.empty:
            continue
        transitions = np.diff(np.hstack([[0], behavior_series.values, [0]]))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0] - 1

        if len(starts) == 0 or len(ends) == 0:
            continue

        for i in range(len(starts)):
            if starts[i] < len(df) and ends[i] < len(df) and starts[i] <= ends[i]:
                try:
                    start_frame = int(df.iloc[starts[i]][frame_col])
                    end_frame = int(df.iloc[ends[i]][frame_col])
                    blocks.append((behavior_id, start_frame, end_frame))
                except IndexError:
                    print(f"Warning: Index out of bounds for behavior {behavior_id} at indices {starts[i]}/{ends[i]}. Skipping.")
                except ValueError:
                    print(f"Warning: Non-integer frame number for behavior {behavior_id} at indices {starts[i]}/{ends[i]}. Skipping.")

    return sorted(blocks, key=lambda x: x[1])


def extract_sequences(blocks, max_gap=60,
                      quiet_period_before=60, quiet_period_after=0,
                      max_sequence_duration=None,
                      start_phase=None, end_phase=None):
    """
    Extract sequences of behaviors.

    Args:
        blocks (list): List of (behavior_id, start_frame, end_frame) tuples, sorted by start_frame.
        max_gap (int): Maximum frame gap *within* a sequence.
        quiet_period_before (int): Minimum frame gap required *before* the sequence starts.
        quiet_period_after (int): Minimum frame gap required *after* the sequence ends.
        max_sequence_duration (int, optional): Maximum allowed duration of the sequence in frames,
                                               from the start of the first behavior to the end of the last.
                                               If None, no maximum duration is enforced.
        start_phase (int, optional): Behavior ID to start sequences with.
        end_phase (int, optional): Behavior ID to end sequences with.

    Returns:
        list: List of sequence strings like "12543".
    """
    sequences = []
    if not blocks:
        return sequences

    num_blocks = len(blocks)

    for start_idx, start_block_tuple in enumerate(blocks):
        current_behavior_id, current_start_frame, current_end_frame = start_block_tuple

        # --- Filter 1: Check start_phase requirement ---
        if start_phase is not None and current_behavior_id != start_phase:
            continue

        # --- Filter 2: Check for quiet period BEFORE this potential start ---
        is_quiet_before = True
        if quiet_period_before > 0: # Only check if a quiet period before is required
            if start_idx > 0:
                prev_block_end_frame = blocks[start_idx - 1][2]
                if current_start_frame - prev_block_end_frame <= quiet_period_before:
                    is_quiet_before = False
            elif current_start_frame <= quiet_period_before: # Check against frame 0 if it's the very first block
                is_quiet_before = False


        if not is_quiet_before:
            continue

        # --- If quiet before, proceed to build the sequence ---
        current_sequence_ids = [current_behavior_id]
        sequence_actual_start_frame = current_start_frame
        last_behavior_end_frame_in_seq = current_end_frame
        last_block_idx_in_sequence = start_idx


        # Look ahead for subsequent behaviors
        current_block_scan_idx = start_idx + 1
        while current_block_scan_idx < num_blocks:
            next_behavior_id, next_start_frame, next_end_frame = blocks[current_block_scan_idx]

            if next_start_frame - last_behavior_end_frame_in_seq <= max_gap:
                current_sequence_ids.append(next_behavior_id)
                last_behavior_end_frame_in_seq = next_end_frame
                last_block_idx_in_sequence = current_block_scan_idx

                if end_phase is not None and next_behavior_id == end_phase:
                    break
            else:
                break

            current_block_scan_idx += 1

        # --- Sequence built, now check filters ---

        if len(current_sequence_ids) < 2:
             continue

        if end_phase is not None and current_sequence_ids[-1] != end_phase:
             continue

        sequence_actual_end_frame = blocks[last_block_idx_in_sequence][2]
        current_sequence_duration = sequence_actual_end_frame - sequence_actual_start_frame
        if max_sequence_duration is not None and current_sequence_duration > max_sequence_duration:
            continue

        is_quiet_after = True
        if quiet_period_after > 0:
            seq_final_block_end_frame = blocks[last_block_idx_in_sequence][2]
            next_block_overall_idx = last_block_idx_in_sequence + 1

            if next_block_overall_idx < num_blocks:
                next_overall_block_start_frame = blocks[next_block_overall_idx][1]
                if next_overall_block_start_frame - seq_final_block_end_frame <= quiet_period_after:
                    is_quiet_after = False

        if is_quiet_after:
            sequences.append(''.join(map(str, current_sequence_ids)))

    return sequences


def process_file(file_path, max_gap=60,
                 quiet_period_before=60, quiet_period_after=0,
                 max_sequence_duration=None,
                 start_phase=None, end_phase=None):
    """
    Process a single CSV file to extract behavioral sequences.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
             print(f"Warning: File {file_path} is empty.")
             return []

        blocks = extract_behavior_blocks(df)
        if not blocks:
            return []

        sequences = extract_sequences(blocks, max_gap,
                                      quiet_period_before, quiet_period_after,
                                      max_sequence_duration,
                                      start_phase, end_phase)
        return sequences

    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Error: File {file_path} is empty or contains no data.")
        return []
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        traceback.print_exc()
        return []


def analyze_sequences(sequences):
    """
    Analyze the frequency of different sequences.
    """
    sequence_counts = defaultdict(int)
    for seq in sequences:
        sequence_counts[seq] += 1
    return dict(sorted(sequence_counts.items(), key=lambda item: item[1], reverse=True))


def save_sequences(sequences, counts, output_file):
    """
    Save sequences and their counts to a CSV file. Also saves raw sequences.
    """
    if not counts:
        print(f"No sequences to save for {output_file}.")
        return

    df = pd.DataFrame({
        'Sequence': list(counts.keys()),
        'Count': list(counts.values())
    })
    df['Behaviors'] = df['Sequence'].apply(
        lambda seq: '-'.join(BEHAVIOR_ID_TO_NAME.get(int(c), '?') for c in seq)
    )
    df = df[['Sequence', 'Behaviors', 'Count']]

    try:
        df.to_csv(output_file, index=False)
        print(f"Saved sequence counts to: {output_file}")
    except Exception as e:
        print(f"Error saving sequence counts to {output_file}: {e}")

    raw_output_file = output_file.replace('.csv', '_raw.txt')
    try:
        with open(raw_output_file, 'w') as f:
            for seq in sequences:
                f.write(seq + '\n')
        print(f"Saved raw sequences list to: {raw_output_file}")
    except Exception as e:
        print(f"Error saving raw sequences to {raw_output_file}: {e}")


def validate_results(sequences, start_phase=None, end_phase=None):
    """
    Validate that all sequences adhere to the start/end phase requirements.
    NOTE: This does NOT validate quiet periods or max duration.
    """
    if not sequences:
        print("No sequences to validate.")
        return

    invalid_start_count = 0
    invalid_end_count = 0

    for seq in sequences:
        if not seq or len(seq) < 1:
             continue
        if start_phase is not None:
            try:
                if int(seq[0]) != start_phase:
                    print(f"Validation Warning: Seq {seq} not starting with {start_phase}")
                    invalid_start_count += 1
            except (ValueError, IndexError):
                 print(f"Validation Warning: Cannot validate start for '{seq}'")
                 invalid_start_count += 1
        if end_phase is not None:
            try:
                if int(seq[-1]) != end_phase:
                    print(f"Validation Warning: Seq {seq} not ending with {end_phase}")
                    invalid_end_count += 1
            except (ValueError, IndexError):
                 print(f"Validation Warning: Cannot validate end for '{seq}'")
                 invalid_end_count += 1

    if start_phase is not None or end_phase is not None:
        if invalid_start_count == 0 and invalid_end_count == 0:
            print("Validation: All sequences conform to start/end phase.")
        else:
            print(f"Validation: {invalid_start_count} start violations, {invalid_end_count} end violations.")


def main():
    parser = argparse.ArgumentParser(
        description='Extract behavioral sequences from individual CSV files with independent quiet periods and max duration.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('directory', help='Directory containing CSV files')
    parser.add_argument('--max-gap', type=int, default=60,
                        help='Maximum frame gap allowed *within* a behavior sequence')
    parser.add_argument('--quiet-before', type=int, default=60,
                        help='Minimum empty frames required *before* the sequence starts')
    parser.add_argument('--quiet-after', type=int, default=0,
                        help='Minimum empty frames required *after* the sequence ends (e.g., 0 for no minimum)')
    parser.add_argument('--max-seq-duration', type=int, default=None,
                        help='Maximum duration of the sequence in frames (from start of first behavior to end of last). No limit if not set.')
    parser.add_argument('--start-phase', type=str,
                        help='Only include sequences starting with this behavior (e.g., nose, 1)')
    parser.add_argument('--end-phase', type=str,
                        help='Only include sequences ending with this behavior (e.g., body, 5)')
    parser.add_argument('--output-dir',
                        help='Directory to save output files (defaults to input directory)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate sequence conformity to start/end phase (not quiet periods/duration)')

    args = parser.parse_args()

    start_phase_id = None
    if args.start_phase:
        try:
            start_phase_id = int(args.start_phase)
            if start_phase_id not in BEHAVIOR_ID_TO_NAME:
                print(f"Error: Invalid start_phase ID: {start_phase_id}. Valid: {list(BEHAVIOR_ID_TO_NAME.keys())}")
                return
        except ValueError:
            if args.start_phase.lower() in BEHAVIOR_MAP:
                start_phase_id = BEHAVIOR_MAP[args.start_phase.lower()]
            else:
                print(f"Error: Unknown start_phase name: '{args.start_phase}'. Valid: {list(BEHAVIOR_MAP.keys())}")
                return

    end_phase_id = None
    if args.end_phase:
        try:
            end_phase_id = int(args.end_phase)
            if end_phase_id not in BEHAVIOR_ID_TO_NAME:
                print(f"Error: Invalid end_phase ID: {end_phase_id}. Valid: {list(BEHAVIOR_ID_TO_NAME.keys())}")
                return
        except ValueError:
            if args.end_phase.lower() in BEHAVIOR_MAP:
                end_phase_id = BEHAVIOR_MAP[args.end_phase.lower()]
            else:
                print(f"Error: Unknown end_phase name: '{args.end_phase}'. Valid: {list(BEHAVIOR_MAP.keys())}")
                return

    print("-" * 30)
    print("Behavioral Sequence Extraction Settings:")
    print(f"Input Directory: {args.directory}")
    print(f"Max gap within sequence: {args.max_gap} frames")
    print(f"Quiet period BEFORE sequence: {args.quiet_before} frames")
    print(f"Quiet period AFTER sequence: {args.quiet_after} frames")
    max_dur_str = f"{args.max_seq_duration} frames" if args.max_seq_duration is not None else "No limit"
    print(f"Max sequence duration: {max_dur_str}")
    if start_phase_id:
        print(f"Start phase filter: {BEHAVIOR_ID_TO_NAME.get(start_phase_id)} (ID: {start_phase_id})")
    else:
        print("Start phase filter: None")
    if end_phase_id:
        print(f"End phase filter: {BEHAVIOR_ID_TO_NAME.get(end_phase_id)} (ID: {end_phase_id})")
    else:
        print("End phase filter: None")
    print("-" * 30)

    output_dir = args.output_dir if args.output_dir else args.directory
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
             print(f"Error creating output directory {output_dir}: {e}")
             return

    # --- MODIFIED SECTION: Find all CSV files and process them individually ---
    all_csv_files = []
    for root, _, files in os.walk(args.directory):
        for file in files:
            if file.lower().endswith('.csv'):
                # To avoid re-processing its own output, check if 'sequences' is in the filename
                if 'sequences' not in file.lower():
                    all_csv_files.append(os.path.join(root, file))

    if not all_csv_files:
        print(f"Warning: No CSV files found in directory {args.directory}")
        return

    print(f"Found {len(all_csv_files)} CSV files to process.")

    for file_path in all_csv_files:
        print(f"\n--- Processing File: {os.path.basename(file_path)} ---")

        sequences = process_file(file_path, args.max_gap,
                                 args.quiet_before, args.quiet_after,
                                 args.max_seq_duration,
                                 start_phase_id, end_phase_id)

        if not sequences:
            print(f"No valid sequences found for: {os.path.basename(file_path)}. Skipping analysis/saving.")
            continue

        if args.validate:
            print("\nValidating start/end phase requirements:")
            validate_results(sequences, start_phase_id, end_phase_id)

        sequence_counts = analyze_sequences(sequences)

        # --- Create a unique output filename for this specific input file ---
        phase_info = ''
        if start_phase_id: phase_info += f"_start{start_phase_id}"
        if end_phase_id: phase_info += f"_end{end_phase_id}"
        max_dur_file_str = f"_maxdur{args.max_seq_duration}" if args.max_seq_duration is not None else "_maxdurnolimit"

        input_file_basename = os.path.splitext(os.path.basename(file_path))[0]
        output_file_base = (f"{input_file_basename}_sequences{phase_info}_gap{args.max_gap}"
                            f"_qb{args.quiet_before}_qa{args.quiet_after}{max_dur_file_str}")
        output_file_csv = os.path.join(output_dir, f"{output_file_base}.csv")

        save_sequences(sequences, sequence_counts, output_file_csv)

        print(f"\n--- Summary for File: {os.path.basename(file_path)} ---")
        print(f"Found {len(sequences)} total sequences meeting all criteria.")
        print(f"Found {len(sequence_counts)} unique sequences.")
        top_n = 10
        print(f"\nTop {top_n} most frequent sequences:")
        if not sequence_counts:
            print("  (No sequences found)")
        else:
            for i, (seq, count) in enumerate(list(sequence_counts.items())[:top_n]):
                try:
                    behaviors = '-'.join(BEHAVIOR_ID_TO_NAME.get(int(c), '?') for c in seq)
                    print(f"  {i+1}. {seq} ({behaviors}): {count} occurrences")
                except ValueError:
                     print(f"  {i+1}. Invalid sequence format '{seq}': {count} occurrences")

    print("\n--- All Files Processed ---")


if __name__ == "__main__":
    main()
