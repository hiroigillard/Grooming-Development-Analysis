import pandas as pd
import os
import argparse

def convert_binary_csv_to_theme(input_folder, output_folder):
    """
    Converts all binary CSV files in an input folder to THEME-compatible
    raw data .txt files in an output folder.

    Each CSV is expected to have a time column (first column) and subsequent
    binary columns for different behaviors. A '1' indicates the active behavior
    for that time point.

    Args:
        input_folder (str): The path to the folder containing the .csv files.
        output_folder (str): The path to the folder where .txt files will be saved.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        except OSError as e:
            print(f"Error creating output folder '{output_folder}': {e}")
            return

    # List all files in the input folder
    try:
        files_to_process = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        if not files_to_process:
            print(f"No .csv files found in '{input_folder}'.")
            return
    except FileNotFoundError:
        print(f"Error: Input folder not found at '{input_folder}'")
        return

    print(f"Found {len(files_to_process)} CSV files to process...")

    # Process each file
    for filename in files_to_process:
        input_filepath = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + '.txt'
        output_filepath = os.path.join(output_folder, output_filename)

        print(f"Processing '{filename}' -> '{output_filename}'")

        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(input_filepath)

            if df.empty:
                print(f"  - Warning: '{filename}' is empty. Skipping.")
                continue

            # Use the first column as the time index, assuming it's the frame number
            time_col = df.columns[0]
            df = df.set_index(time_col)

            # List to store the events (time, event_name)
            events = []
            previous_behavior = None

            # Iterate through each row (time point) of the DataFrame
            for time, row in df.iterrows():
                # Find the column with the '1' in it, which indicates the active behavior
                active_behavior_series = row[row == 1]

                if not active_behavior_series.empty:
                    current_behavior = active_behavior_series.index[0]

                    # We only care about transitions to a new, meaningful behavior
                    # We ignore the 'background' behavior as an event itself
                    if current_behavior != previous_behavior and str(current_behavior).lower() != 'background':
                        events.append((time, current_behavior))

                    previous_behavior = current_behavior
                else:
                    # If no behavior is active (all zeros), treat as background/transition
                    previous_behavior = None

            # If no events were found, skip creating a file
            if not events:
                print(f"  - No valid events found in '{filename}'. Skipping.")
                continue

            # Write the events to the THEME-formatted .txt file
            with open(output_filepath, 'w') as f:
                # Write the header
                f.write("time\tevent\n")

                # Write the start of observation marker ':' at the time of the first event
                first_event_time = events[0][0]
                f.write(f"{first_event_time}\t:\n")

                # Write all the detected events (transitions)
                for time, event_name in events:
                    f.write(f"{time}\t{event_name}\n")

                # Write the end of observation marker '&' at the last time point of the recording
                last_time = df.index[-1]
                f.write(f"{last_time}\t&\n")

            print(f"  - Successfully created '{output_filename}'")

        except Exception as e:
            print(f"  - An error occurred while processing '{filename}': {e}")

    print("\nConversion complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert binary CSV files (DeepEthogram style) to THEME-compatible .txt files."
    )
    parser.add_argument(
        "input_dir",
        help="Path to the folder containing the input .csv files."
    )
    parser.add_argument(
        "output_dir",
        help="Path to the folder where output .txt files will be saved."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    convert_binary_csv_to_theme(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
