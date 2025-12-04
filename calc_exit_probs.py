import os
import csv
import argparse
import re
from pathlib import Path


def get_group_from_filename(filename, groups):
    """
    Identifies which group a file belongs to based on its name by checking
    for the group name as a whole word within the filename.

    Args:
        filename (str): The name of the file.
        groups (list): A list of group identifiers (e.g., ['veh', 'rop', 'SKF']).

    Returns:
        str: The identified group name or 'other' if no match is found.
    """
    base_name = Path(filename).stem.lower()
    # Find the best match (longest group name) to avoid ambiguity e.g. 'group' vs 'group_a'
    best_match = ''
    for group in groups:
        # Check if the group name appears as a whole word in the filename
        # A "word" is surrounded by non-alphanumeric characters or is at the start/end
        if re.search(r'\b' + re.escape(group.lower()) + r'\b', base_name):
            if len(group) > len(best_match):
                best_match = group

    if best_match:
        return best_match

    return 'other'


def analyze_file_for_digit_counts(filepath, digits_to_check):
    """
    Analyzes a single file to count total lines and lines ending with specific digits,
    ignoring any lines that are not valid integers.

    Args:
        filepath (Path): The path to the file to analyze.
        digits_to_check (list[int]): A list of digits to check for.

    Returns:
        tuple: A tuple containing (total_numeric_lines, termination_counts dict).
               Returns (0, {}) if the file cannot be read or contains no numbers.
    """
    try:
        # Added errors='ignore' to prevent crashes on UnicodeDecodeError
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = [line.strip() for line in f if line.strip()]
    except IOError as e:
        print(f"Warning: Could not read file {filepath}. Error: {e}")
        return 0, {}

    numeric_lines = []
    for i, line in enumerate(all_lines, 1):
        try:
            # Ensure the line is a valid integer before processing
            int(line)
            numeric_lines.append(line)
        except ValueError:
            # This line is not a valid integer, so we skip it.
            print(f"Warning: Non-numeric value '{line}' found on line {i} in {filepath.name}. Skipping line.")
            continue

    total_numeric_lines = len(numeric_lines)
    termination_counts = {digit: 0 for digit in digits_to_check}

    if total_numeric_lines > 0:
        for line in numeric_lines:
            for digit in digits_to_check:
                if line.endswith(str(digit)):
                    termination_counts[digit] += 1

    return total_numeric_lines, termination_counts


def main():
    """
    Main function to drive the sequence analysis process using command-line arguments.
    """
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description=(
            "Analyzes .txt files to find the proportion of lines ending with specific digits, "
            "and creates a separate CSV report for each group."
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="The path to the folder containing your .txt files."
    )
    parser.add_argument(
        "--numbers",
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help="A space-separated list of digits to check for termination.\n"
             "Default: 1 2 3 4 5"
    )
    parser.add_argument(
        "--groups",
        nargs='+',
        default=['veh', 'rop', 'SKF'],
        help="A space-separated list of group names to categorize files by.\n"
             "Default: veh rop SKF"
    )
    args = parser.parse_args()

    folder_path = Path(args.folder_path)
    digits_to_check = args.numbers
    groups = args.groups

    if not folder_path.is_dir():
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    # --- Initialization ---
    txt_files = sorted(list(folder_path.glob('*.txt')))
    if not txt_files:
        print(f"No .txt files found in '{folder_path}'.")
        return

    # Prepare a dictionary to hold the per-file results for each group
    results_by_group = {group: [] for group in groups}
    results_by_group['other'] = []

    print(f"\nFound {len(txt_files)} .txt files. Analyzing for digits: {digits_to_check}")
    print(f"Grouping by: {groups}")

    # --- File Processing & Data Collection ---
    for filepath in txt_files:
        # Skip macOS metadata files which often cause encoding errors.
        if filepath.name.startswith('._'):
            print(f"  - Skipping macOS metadata file: {filepath.name}")
            continue

        group = get_group_from_filename(filepath.name, groups)
        print(f"  - Processing {filepath.name} (Group: {group})...")
        total_lines, term_counts = analyze_file_for_digit_counts(filepath, digits_to_check)

        if total_lines > 0:
            # Prepare results for this file
            file_proportions = {'Filename': filepath.name}
            for digit, count in term_counts.items():
                proportion = count / total_lines
                file_proportions[f"Proportion_for_{digit}"] = f"{proportion:.4f}"

            # Add the file's results to the correct group
            results_by_group[group].append(file_proportions)
        else:
            # Add a message if a file has no valid data to analyze
            print(f"    -> No valid numeric lines found in {filepath.name}. It will not be included in any report.")

    # --- Writing CSV Reports for each group ---
    print("\nWriting group reports...")
    any_files_written = False
    for group, results in results_by_group.items():
        # Only create a file if there are results for that group
        if not results:
            continue

        output_path = folder_path / f"group_{group}_report.csv"
        headers = ['Filename'] + [f"Proportion_for_{d}" for d in digits_to_check]

        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(results)
            print(f"  - Successfully created report: {output_path}")
            any_files_written = True
        except IOError as e:
            print(f"  - Error writing report for group '{group}': {e}")

    if not any_files_written:
        print("\nNo data was processed, so no report files were created.")


if __name__ == '__main__':
    main()
