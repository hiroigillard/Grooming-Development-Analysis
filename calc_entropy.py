import os
import numpy as np
import pandas as pd


def calculate_properties_for_matrix(matrix: np.ndarray):
    """
    Calculates the stationary distribution (mu), entropy (H), and the second
    largest eigenvalue for a single transition matrix.

    Args:
        matrix (np.ndarray): An individual transition probability matrix.

    Returns:
        tuple: A tuple containing (mu, entropy, second_largest_eigenvalue).
               Returns (None, None, None) if calculation fails.
    """
    try:
        # --- 1. Calculate Eigenvalues and Eigenvectors ---
        P_transpose = matrix.T
        eigenvalues, eigenvectors = np.linalg.eig(P_transpose)

        # --- 2. Find the Second Largest Eigenvalue ---
        # Take the absolute value of the eigenvalues and sort them in descending order
        sorted_magnitudes = np.sort(np.abs(eigenvalues))[::-1]

        # The largest is 1, the second largest is the next one in the sorted list.
        second_largest_eigenvalue = sorted_magnitudes[1]  # <-- ADDED

        # --- 3. Calculate Stationary Distribution (mu) ---
        stationary_index = np.argmin(np.abs(eigenvalues - 1))

        if not np.isclose(eigenvalues[stationary_index], 1):
            print(
                f"Warning: No eigenvalue of 1 found. Closest was {eigenvalues[stationary_index].real:.4f}. Skipping matrix.")
            return None, None, None  # <-- CHANGED

        stationary_vector = eigenvectors[:, stationary_index].real
        mu = stationary_vector / np.sum(stationary_vector)

        # --- 4. Compute Entropy (H) ---
        log_P = np.log2(matrix, where=(matrix > 0), out=np.zeros_like(matrix, dtype=float))
        row_entropies = -np.sum(matrix * log_P, axis=1)
        entropy = np.dot(mu, row_entropies)

        return mu, entropy, second_largest_eigenvalue  # <-- CHANGED

    except np.linalg.LinAlgError:
        print("Warning: Linear algebra error (e.g., matrix not square). Skipping matrix.")
        return None, None, None  # <-- CHANGED


def process_folder(folder_path: str, output_csv_path: str):
    """
    Processes each matrix file in a folder and saves the calculated
    entropy, mu, and second eigenvalue values to a single CSV file.
    """
    print(f"Reading matrices from: {folder_path}")

    matrix_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not matrix_files:
        print("Error: No .csv files found in the specified folder.")
        return

    results_list = []
    num_states = 0

    for file_name in matrix_files:
        file_path = os.path.join(folder_path, file_name)

        try:
            df = pd.read_csv(file_path, index_col=0, encoding='latin1')
            if df.empty:
                print(f"Warning: File {file_name} is empty or unreadable. Skipping.")
                continue
            matrix = df.to_numpy()

        except Exception as e:
            print(f"Could not read file {file_name}. Error: {e}. Skipping.")
            continue

        if not num_states:
            num_states = matrix.shape[0]

        # Unpack all three returned values
        mu, entropy, second_eig = calculate_properties_for_matrix(matrix)  # <-- CHANGED

        if mu is not None:  # Check if calculations were successful
            # Add the new value to the results dictionary
            result_row = {
                'filename': file_name,
                'entropy': entropy,
                'second_largest_eigenvalue': second_eig  # <-- ADDED
            }
            for i, mu_val in enumerate(mu):
                result_row[f'mu_{i + 1}'] = mu_val

            results_list.append(result_row)

    if not results_list:
        print("Processing failed for all files. No output generated.")
        return

    results_df = pd.DataFrame(results_list)

    # Add the new column name to the list for ordering
    cols = ['filename', 'entropy', 'second_largest_eigenvalue'] + [f'mu_{i + 1}' for i in
                                                                   range(num_states)]  # <-- CHANGED

    results_df = results_df[cols]
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nProcessing complete. Results saved to: {output_csv_path}")


if __name__ == "__main__":
    FOLDER_PATH = '/Volumes/Storage/Summer_Studentship/Individual_Cprob'
    OUTPUT_FILE_PATH = '/Volumes/Storage/Summer_Studentship/Individual_Cprob/results.csv'

    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if FOLDER_PATH == 'path/to/your/matrices':
        print("=" * 60)
        print("!!! PLEASE UPDATE THE 'FOLDER_PATH' VARIABLE IN THE SCRIPT !!!")
        print("=" * 60)
    else:
        process_folder(FOLDER_PATH, OUTPUT_FILE_PATH)

