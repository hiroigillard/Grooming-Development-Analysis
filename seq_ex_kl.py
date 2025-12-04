import pandas as pd
import numpy as np
import os
import argparse
from collections import defaultdict
import fnmatch

# --- Mappings ---
BEHAVIOR_TO_PHASE_MAP = {
    'nose': 'Phase 1',
    'whiskers': 'Phase 2',
    'eyes': 'Phase 3',
    'ears': 'Phase 4',
    'body': 'Phase 5'
}
BEHAVIOR_TO_KEY = {
    'nose': '1',
    'whiskers': '2',
    'eyes': '3',
    'ears': '4',
    'body': '5'
}
DEFAULT_BEHAVIORS = list(BEHAVIOR_TO_PHASE_MAP.keys())


def find_bouts(df, max_gap=60, behavior_cols=None, min_bout_events=2):
    """
    Identifies and processes behavioral bouts from a binary DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with frame numbers and binary behavior columns.
        max_gap (int): The maximum frame gap to link events and define isolation.
        behavior_cols (list): List of behavior column names to analyze.
        min_bout_events (int): The minimum number of events required to constitute a valid bout.

    Returns:
        tuple: A tuple containing:
            - list: A list of sequence strings for the .txt file.
            - list: A list of dictionaries, where each dict contains the occupancy
                    of each behavior for a single valid bout.
    """
    if behavior_cols is None:
        behavior_cols = DEFAULT_BEHAVIORS

    frame_col = df.columns[0]
    all_events = []

    # 1. Identify all individual behavior events
    for behavior in behavior_cols:
        if behavior not in df.columns:
            continue
        s = df[behavior].astype(int)
        transitions = np.diff(np.hstack([[0], s.values, [0]]))
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0] - 1
        if len(starts) == 0: continue
        min_len = min(len(starts), len(ends))
        for i in range(min_len):
            if 0 <= starts[i] < len(df) and 0 <= ends[i] < len(df):
                start_frame = int(df.iloc[starts[i]][frame_col])
                end_frame = int(df.iloc[ends[i]][frame_col])
                all_events.append({
                    'behavior': behavior,
                    'start': start_frame,
                    'end': end_frame,
                    'duration': end_frame - start_frame + 1
                })

    if not all_events:
        return [], []

    # 2. Sort events chronologically and group them into bouts
    all_events.sort(key=lambda x: x['start'])
    raw_bouts = []
    if all_events:
        current_bout = [all_events[0]]
        for i in range(1, len(all_events)):
            prev_event, current_event = all_events[i - 1], all_events[i]
            gap = current_event['start'] - prev_event['end']
            if 0 < gap < max_gap:
                current_bout.append(current_event)
            else:
                raw_bouts.append(current_bout)
                current_bout = [current_event]
        raw_bouts.append(current_bout)

    # 3. Filter bouts by minimum length
    valid_bouts = [b for b in raw_bouts if len(b) >= min_bout_events]

    # 4. Process valid bouts for output
    bout_sequences = []
    bout_occupancy_data = []  # Will hold occupancy dict for each bout

    for bout in valid_bouts:
        # Generate sequence string (e.g., "1-2-1")
        seq_str = "".join([BEHAVIOR_TO_KEY.get(e['behavior'], '?') for e in bout])
        bout_sequences.append(seq_str)

        # Calculate occupancy FOR THIS BOUT
        bout_total_duration = sum(e['duration'] for e in bout)
        current_bout_occupancy = {}
        if bout_total_duration > 0:
            for behavior in behavior_cols:
                duration_in_bout = sum(e['duration'] for e in bout if e['behavior'] == behavior)
                current_bout_occupancy[behavior] = duration_in_bout / bout_total_duration

        if current_bout_occupancy:
            bout_occupancy_data.append(current_bout_occupancy)

    return bout_sequences, bout_occupancy_data


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description='Identify behavioral bouts and calculate average phase occupancy per bout.')
    parser.add_argument('directory', help='Directory containing the CSV files to analyze.')
    parser.add_argument('--file-pattern', default='*.csv',
                        help='Pattern to match files within the directory (e.g., "*.csv"). Default: *.csv')
    parser.add_argument('--max-gap', type=int, default=60,
                        help='Maximum frame gap to define bouts and isolation (default: 60).')
    parser.add_argument('--min-bout-len', type=int, default=2,
                        help='Minimum number of events (phases) in a bout to be included (default: 2).')
    parser.add_argument('--output-dir',
                        help='Directory to save output files (defaults to a new "bout_analysis" folder).')
    parser.add_argument('--behaviors', nargs='+', default=DEFAULT_BEHAVIORS,
                        help=f'List of behavior names to analyze. Defaults to: {DEFAULT_BEHAVIORS}.')

    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.directory, 'bout_analysis')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Analysis running with min bout length = {args.min_bout_len}. Output will be in: {output_dir}")

    files_to_process = [
        os.path.join(root, f)
        for root, _, files in os.walk(args.directory)
        for f in fnmatch.filter(files, args.file_pattern)
    ]
    print(f"Found {len(files_to_process)} files to analyze.")

    all_files_avg_occupancy = []

    for file_path in files_to_process:
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\n--- Processing: {base_filename} ---")

        try:
            df = pd.read_csv(file_path)
            if df.empty or len(df.columns) < 2:
                print("  Warning: File is empty or invalid. Skipping.")
                continue

            sequences, bout_occupancy_list = find_bouts(df, args.max_gap, args.behaviors, args.min_bout_len)

            if not sequences:
                print(f"  No valid bouts with at least {args.min_bout_len} events found.")
                continue

            # --- Output 1: Write sequences to .txt file ---
            txt_output_path = os.path.join(output_dir, f"{base_filename}_sequences.txt")
            with open(txt_output_path, 'w') as f:
                for seq in sequences:
                    f.write(f"{seq}\n")
            print(f"  Saved {len(sequences)} bout sequences to {txt_output_path}")

            # --- Prepare data for Occupancy CSV by averaging per-bout results ---
            if bout_occupancy_list:
                # Convert list of bout occupancy dicts to a DataFrame
                bout_df = pd.DataFrame(bout_occupancy_list)

                # Calculate the mean occupancy for each behavior across all bouts
                mean_occupancies = bout_df.mean().to_dict()

                # Store the final averaged results for the file
                file_avg_occupancy = {'filename': base_filename}
                for behavior, phase_name in BEHAVIOR_TO_PHASE_MAP.items():
                    if behavior in args.behaviors:
                        file_avg_occupancy[phase_name] = mean_occupancies.get(behavior, 0)

                all_files_avg_occupancy.append(file_avg_occupancy)

        except Exception as e:
            print(f"  ERROR: Could not process file {base_filename}. Reason: {e}")
            import traceback
            traceback.print_exc()

    # --- Output 2: Write all averaged occupancy data to a single CSV file ---
    if all_files_avg_occupancy:
        print("\n--- Finalizing Average Occupancy Report ---")

        phase_cols = [
            phase for behavior, phase in BEHAVIOR_TO_PHASE_MAP.items()
            if behavior in args.behaviors
        ]
        csv_cols = ['filename'] + sorted(phase_cols)

        summary_df = pd.DataFrame(all_files_avg_occupancy)
        summary_df = summary_df.reindex(columns=csv_cols).fillna(0.0)

        csv_output_path = os.path.join(output_dir, 'bout_avg_occupancy_summary.csv')
        summary_df.to_csv(csv_output_path, index=False, float_format='%.6f')
        print(f"Average bout occupancy summary saved to: {csv_output_path}")
    else:
        print("\nNo valid bouts were found in any files. No summary report generated.")

    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
