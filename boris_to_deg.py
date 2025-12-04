import csv
import os
import math
import argparse
from pathlib import Path

def convert_tsv_to_csv(input_tsv_path, output_csv_path, fps=120):
    """
    Converts a single TSV file to the specified CSV format, handling time gaps
    and calculating frame numbers.

    Args:
        input_tsv_path (str): Path to the input TSV file.
        output_csv_path (str): Path to the output CSV file.
        fps (int): Frames per second to use for time-to-frame conversion.
    """
    print(f"Processing '{input_tsv_path}'...")
    try:
        # Ensure output directory exists
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

        with open(input_tsv_path, 'r', newline='') as infile, \
             open(output_csv_path, 'w', newline='') as outfile:

            tsv_reader = csv.reader(infile, delimiter='\t')
            csv_writer = csv.writer(outfile, delimiter=',')

            # Expected TSV header columns (order matters for reading)
            expected_tsv_header = ['time', 'Body', 'Ears', 'Eyes', 'Nose', 'Whiskers']
            # Required CSV header columns (order matters for writing)
            csv_header = ['Unnamed: 0', 'background', 'nose', 'whiskers', 'eyes', 'ears', 'body']

            # Read and verify TSV header
            try:
                header = next(tsv_reader)
                if header != expected_tsv_header:
                    print(f"Warning: Unexpected header in '{input_tsv_path}'. Expected {expected_tsv_header}, found {header}. Proceeding, but column mapping might be incorrect.")
                    # Attempt to find indices if possible, otherwise assume standard order
                    try:
                        time_idx = header.index('time')
                        body_idx = header.index('Body')
                        ears_idx = header.index('Ears')
                        eyes_idx = header.index('Eyes')
                        nose_idx = header.index('Nose')
                        whiskers_idx = header.index('Whiskers')
                    except ValueError:
                         print(f"Error: Could not find all required columns in header of '{input_tsv_path}'. Skipping file.")
                         return # Skip this file if header is fundamentally wrong
                else:
                    # Standard order indices
                    time_idx, body_idx, ears_idx, eyes_idx, nose_idx, whiskers_idx = 0, 1, 2, 3, 4, 5

            except StopIteration:
                print(f"Warning: Input file '{input_tsv_path}' is empty or has no header. Creating empty CSV.")
                csv_writer.writerow(csv_header)
                return # Nothing more to process

            # Write CSV header
            csv_writer.writerow(csv_header)

            expected_frame = 0
            last_written_frame = -1 # To detect duplicate frames after rounding

            for i, row in enumerate(tsv_reader, start=1):
                if len(row) != len(expected_tsv_header):
                    print(f"Warning: Skipping row {i+1} in '{input_tsv_path}' due to incorrect number of columns: {row}")
                    continue

                try:
                    time_str = row[time_idx]
                    body = int(row[body_idx])
                    ears = int(row[ears_idx])
                    eyes = int(row[eyes_idx])
                    nose = int(row[nose_idx])
                    whiskers = int(row[whiskers_idx])
                    time_val = float(time_str)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid data row {i+1} in '{input_tsv_path}': {row}. Error: {e}")
                    continue

                # Calculate current frame number
                current_frame = round(time_val * fps)

                if current_frame < 0:
                     print(f"Warning: Skipping row {i+1} in '{input_tsv_path}' due to negative calculated frame number ({current_frame}) from time {time_val}.")
                     continue

                # Fill gaps before this frame (initial or intermediate)
                while expected_frame < current_frame:
                    if expected_frame > last_written_frame:
                        # Write gap row (background=1, others=0)
                        gap_row = [expected_frame, 1, 0, 0, 0, 0, 0]
                        csv_writer.writerow(gap_row)
                        last_written_frame = expected_frame
                    expected_frame += 1

                # Process the current row if it corresponds to the next expected frame
                # or if it's the first valid frame encountered
                if current_frame == expected_frame:
                    if current_frame > last_written_frame: # Avoid duplicates from rounding
                         # Calculate background
                        background = 1 if (nose + whiskers + eyes + ears + body) == 0 else 0
                        # Write data row in specified CSV order
                        output_row = [current_frame, background, nose, whiskers, eyes, ears, body]
                        csv_writer.writerow(output_row)
                        last_written_frame = current_frame
                    else:
                        # This frame number was already written (due to rounding collision)
                         print(f"Warning: Multiple time entries round to frame {current_frame} near row {i+1} in '{input_tsv_path}'. Keeping first entry.")
                    expected_frame += 1 # Move expectation to the next frame
                elif current_frame < expected_frame:
                     # This row's time maps to a frame we've already processed/skipped past.
                     # This could happen with slightly out-of-order timestamps or rounding.
                     print(f"Warning: Skipping row {i+1} in '{input_tsv_path}' as its time {time_val} maps to frame {current_frame}, which is earlier than expected frame {expected_frame}.")
                     continue # Skip this row


            print(f"Successfully converted '{input_tsv_path}' to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: Input file not found: '{input_tsv_path}'")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{input_tsv_path}': {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert TSV binary behaviour tables to CSV format with frame numbers and gap filling.")
    parser.add_argument("input_dir", help="Directory containing the input TSV files.")
    parser.add_argument("output_dir", help="Directory where the output CSV files will be saved.")
    parser.add_argument("--fps", type=int, default=120, help="Frames per second used for time-to-frame conversion (default: 120).")

    args = parser.parse_args()

    input_directory = Path(args.input_dir)
    output_directory = Path(args.output_dir)

    if not input_directory.is_dir():
        print(f"Error: Input directory '{args.input_dir}' not found or is not a directory.")
        return

    # Create output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"Starting conversion from '{input_directory}' to '{output_directory}'...")

    file_count = 0
    for item in input_directory.iterdir():
        if item.is_file() and item.suffix.lower() == '.tsv':
            input_tsv_path = item
            output_csv_filename = item.stem + '.csv'
            output_csv_path = output_directory / output_csv_filename
            convert_tsv_to_csv(str(input_tsv_path), str(output_csv_path), args.fps)
            file_count += 1

    if file_count == 0:
        print(f"No '.tsv' files found in '{input_directory}'.")
    else:
        print(f"\nConversion complete. Processed {file_count} TSV file(s).")

if __name__ == "__main__":
    main()