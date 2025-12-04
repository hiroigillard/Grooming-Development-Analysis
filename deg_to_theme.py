import pandas as pd
import os

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
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

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

            # Use the first column as the time index, assuming it's the frame number
            time_col = df.columns[0]
            df = df.set_index(time_col)

            # Identify behavior columns (all columns except the original index)
            behavior_columns = df.columns

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
                    if current_behavior != previous_behavior and current_behavior.lower() != 'background':
                        events.append((time, current_behavior))

                    previous_behavior = current_behavior

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


# --- HOW TO USE ---
# 1. Place your .csv files into a single folder.
# 2. Set the `input_data_folder` variable to the path of that folder.
# 3. Set the `output_txt_folder` variable to where you want the .txt files to be saved.
# 4. Set the `fps_rate` to your video's frame rate.
# 5. Run the script.

# Example usage:
# On Windows, a path might look like: 'C:\\Users\\YourUser\\Documents\\GroomingData\\CSV'
# On Mac/Linux, a path might look like: '/Users/youruser/Documents/GroomingData/CSV'

input_data_folder = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/Joint-prob Labels/P18'
output_txt_folder = '/Volumes/Storage/Research Project/Dopamine_Grooming_Hiro+Chloe/THEME_Labels/P18/Point'


convert_binary_csv_to_theme(input_data_folder, output_txt_folder)