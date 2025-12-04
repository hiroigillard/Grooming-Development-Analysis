import os
import pandas as pd


def process_behavioral_data(directory_path):
    """
    Process CSV files in a directory to count 1s in behavioral columns.
    Creates a table with filenames as columns and behaviors as rows.
    Values are converted to seconds (frames/60).

    Args:
        directory_path (str): Path to directory containing CSV files

    Returns:
        pd.DataFrame: Table with behavior durations in seconds for each file
    """
    # Check if directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    # Find all CSV files in the directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return pd.DataFrame()

    # Initialize a dictionary to store results for each file
    results = {}
    processed_files = 0

    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)

        try:
            # Read CSV file
            df = pd.read_csv(file_path)

            # Check if the file has the required columns
            if len(df.columns) >= 7:
                # Extract the behavioral columns (indices 2-6 for columns 3-7)
                # These should be nose, whiskers, eyes, ears, body
                behavioral_columns = df.columns[2:7]

                # Create a dictionary for this file's behaviors
                file_results = {}

                # Sum each behavioral column and convert to seconds (frames/60)
                total_behaviors = 0
                for col in behavioral_columns:
                    count = int(df[col].sum())
                    file_results[col] = round(count / 60, 2)  # Convert to seconds
                    total_behaviors += count

                # Add total of all behaviors (also in seconds)
                file_results['Total'] = round(total_behaviors / 60, 2)

                # Add to results using filename without extension as the key
                file_name = os.path.splitext(csv_file)[0]
                results[file_name] = file_results
                processed_files += 1
                print(f"Processed: {csv_file}")
            else:
                print(f"Warning: {csv_file} does not have enough columns (needs at least 7).")

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    print(f"\nSuccessfully processed {processed_files} out of {len(csv_files)} CSV files.")

    # Convert results to DataFrame with behaviors as rows and filenames as columns
    if results:
        results_df = pd.DataFrame(results)
        return results_df
    else:
        return pd.DataFrame()


def main():
    # Get directory path from user
    directory_path = input("Enter the path to the directory containing CSV files: ")

    try:
        # Process CSV files
        results_df = process_behavioral_data(directory_path)

        if not results_df.empty:
            # Try to use tabulate for prettier console output if available
            try:
                from tabulate import tabulate
                print("\nBehavioral Duration Table (in seconds):")
                print(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt=".2f"))
            except ImportError:
                print("\nBehavioral Duration Table (in seconds):")
                print(results_df)

            # Save table to CSV
            output_path = os.path.join(directory_path, "behavioral_durations_seconds.csv")
            results_df.to_csv(output_path)
            print(f"\nResults saved to: {output_path}")
        else:
            print("No results to display.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()