import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
from POMM import *
import matplotlib.pyplot as plt

def analyze_file(input_path, output_dir, args):
    """
    Runs the full POMM analysis for a single input file.
    """
    print(f"--- Starting analysis for {os.path.basename(input_path)} ---")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: {output_dir}")

    # Load sequences from the input file
    try:
        with open(input_path, 'r') as f:
            sequences = [line.strip() for line in f.readlines() if line.strip()]
        if not sequences:
            print(f"Warning: No sequences found in {input_path}. Skipping.")
            return
        print(f"Loaded {len(sequences)} sequences from {os.path.basename(input_path)}")
    except Exception as e:
        print(f"Error loading sequences from {input_path}: {e}")
        return

    # Print some sequences for verification
    print("\nSample of sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"  {seq}")
    if len(sequences) > 5:
        print(f"  ... (and {len(sequences) - 5} more)")

    # Convert sequences to numeric form
    osIn, symbol_to_int, int_to_symbol = convert_sequences(sequences)
    print(f"\nSymbol mapping: {symbol_to_int}")

    # Start POMM inference
    print(f"\nStarting POMM inference (timeout: {args.timeout}s, max n-gram: {args.ngram_max})...")
    try:
        pomm_found = False
        S, P, pv, PBs, PbT = None, None, None, None, None

        # Try fixed n-gram sizes directly
        for ng in range(1, args.ngram_max + 1):
            print(f"\nTrying n-gram size = {ng}")
            start_time = time.time()
            try:
                temp_S, temp_P, _ = constructNGramPOMMC(osIn, ng)
                print(f"Model construction took {time.time() - start_time:.2f} seconds")

                # Test statistical significance
                print("Testing model significance...")
                temp_pv, temp_PBs, temp_PbT = getPVSampledSeqsPOMM(temp_S, temp_P, osIn, nSample=args.samples, nProc=args.nproc)
                print(f"P-value: {temp_pv:.3f}, Pbeta: {temp_PbT:.3f}")

                if temp_pv > 0.05:  # Acceptable model found
                    print(f"Acceptable model found with n-gram size {ng}")
                    S, P, pv, PBs, PbT = temp_S, temp_P, temp_pv, temp_PBs, temp_PbT
                    pomm_found = True
                    break
            except Exception as e:
                print(f"Error during n-gram {ng} processing: {e}")
                continue
        
        if not pomm_found:
             print("\nNo suitable model found within the n-gram range. Try increasing --ngram-max.")
             return

        # Simplify model if one was found
        print("\nSimplifying model...")
        S, P, pv, PBs, PbT = MinPOMMSimp(S, osIn, minP=0.001, nProc=args.nproc,
                                         nRerun=args.rerun, factors=[0.5])
        print(f"After simplification p-value={pv:.3f}")

        # Create readable state vector for visualization
        S_readable = [0, -1]  # Start and end states
        for s_val in S[2:]:
            S_readable.append(int_to_symbol.get(s_val, '?'))

        # --- Generate outputs ---
        print("\n--- Generating Output Files ---")
        
        # Calculate and save Mean Normalized Transition Entropy
        mean_entropy = calculate_mean_transition_entropy(S, P)
        print(f"\nMean Normalized Transition Entropy: {mean_entropy:.4f}")
        entropy_file = os.path.join(output_dir, "mean_transition_entropy.txt")
        try:
            with open(entropy_file, 'w') as f:
                f.write(f"Mean Normalized Transition Entropy: {mean_entropy:.4f}\n")
            print(f"Mean transition entropy saved to {entropy_file}")
        except IOError as e:
            print(f"Error writing to {entropy_file}: {e}")

        # Plot the transition diagram
        print("\nPlotting transition diagram...")
        plot_file = os.path.join(output_dir, "inferred_pomm.pdf")
        plotTransitionDiagram(S_readable, P, Pcut=0.01, filenamePDF=plot_file,
                              removeUnreachable=False, markedStates=[])
        print(f"Transition diagram saved to {plot_file}")

        # Extract and save transition probabilities
        print("\nExtracting transition probabilities...")
        transitions_base_name = args.export_transitions if args.export_transitions else "transition_probabilities"
        transitions_txt_file = os.path.join(output_dir, f"{transitions_base_name}.txt")
        
        transitions, _ = print_transition_probabilities(
            S, P, int_to_symbol, threshold=args.prob_threshold, output_file=transitions_txt_file)

        # Save transitions to CSV
        df_trans = pd.DataFrame(transitions, columns=['source_idx', 'dest_idx', 'probability'])
        df_trans['source_state'] = df_trans['source_idx'].apply(
            lambda i: "Start" if i == 0 else ("End" if i == 1 else f"{int_to_symbol.get(S[i], '?')}_{i}"))
        df_trans['dest_state'] = df_trans['dest_idx'].apply(
            lambda i: "Start" if i == 0 else ("End" if i == 1 else f"{int_to_symbol.get(S[i], '?')}_{i}"))
        df_trans['dest_symbol'] = df_trans['dest_idx'].apply(
            lambda i: "End" if i == 1 else int_to_symbol.get(S[i], '?'))
        
        csv_file = os.path.join(output_dir, f"{transitions_base_name}.csv")
        df_trans.to_csv(csv_file, index=False)
        print(f"Transition probabilities also saved as CSV to {csv_file}")

        # Plot the P-beta distribution
        plt.figure(figsize=(8, 5))
        plt.hist(PBs, bins=30)
        plt.axvline(x=PbT, color='r', linestyle='-', label=f'Observed Pbeta={PbT:.3f}')
        plt.title(f'P-beta distribution, p-value={pv:.3f}')
        plt.xlabel('P-beta')
        plt.ylabel('Frequency')
        plt.legend()
        distr_file = os.path.join(output_dir, "pbeta_distribution.pdf")
        plt.savefig(distr_file)
        plt.close() # Close the figure to free up memory
        print(f"P-beta distribution plot saved to {distr_file}")

        # Generate sequences from the model
        print("\nGenerating example sequences from the inferred model:")
        gen_seqs = generateSequencePOMM(S, P, 5)
        for seq in gen_seqs:
            readable_seq = ''.join([int_to_symbol.get(s, '?') for s in seq])
            print(f"  {readable_seq}")

    except Exception as e:
        print(f"\nAn unexpected error occurred during the analysis of {os.path.basename(input_path)}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"--- Finished analysis for {os.path.basename(input_path)} ---\n")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze sequences from all .txt files in a directory using POMM.')
    parser.add_argument('--input-dir', '-d', type=str, required=True, help='Input directory containing .txt files with sequences.')
    parser.add_argument('--nproc', '-n', type=int, default=2, help='Number of processes to use.')
    parser.add_argument('--timeout', '-t', type=int, default=300, help='Timeout in seconds for NGramPOMMSearch.')
    parser.add_argument('--ngram-max', '-g', type=int, default=3, help='Maximum n-gram size to try.')
    parser.add_argument('--rerun', '-r', type=int, default=50, help='Number of BW re-runs.')
    parser.add_argument('--samples', '-s', type=int, default=5000, help='Number of samples for Pbeta.')
    parser.add_argument('--export-transitions', '-e', type=str, default="transition_probabilities",
                        help='Base name for the exported transition probabilities files (without extension).')
    parser.add_argument('--prob-threshold', '-p', type=float, default=0.01,
                        help='Probability threshold for reporting transitions (default: 0.01).')
    args = parser.parse_args()

    # Configure POMM parameters
    global nProc
    nProc = args.nproc

    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        sys.exit(1)

    # Find all .txt files in the directory
    try:
        txt_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.txt')])
    except OSError as e:
        print(f"Error reading directory {args.input_dir}: {e}")
        sys.exit(1)

    if not txt_files:
        print(f"No .txt files found in '{args.input_dir}'.")
        sys.exit(0)

    print(f"Found {len(txt_files)} .txt files to process: {', '.join(txt_files)}")

    # Loop through each file and analyze it
    for filename in txt_files:
        input_file_path = os.path.join(args.input_dir, filename)
        
        # Create a corresponding output directory
        base_name = os.path.splitext(filename)[0]
        output_dir_path = os.path.join(args.input_dir, f"{base_name}_output")

        # Run the analysis
        analyze_file(input_file_path, output_dir_path, args)


def print_transition_probabilities(S, P, int_to_symbol, threshold=0.01, output_file=None):
    """
    Print all transition probabilities from the POMM model and save them to files.
    """
    N = len(S)
    state_names = []
    for i in range(N):
        if i == 0:
            state_names.append("Start")
        elif i == 1:
            state_names.append("End")
        else:
            state_names.append(f"{int_to_symbol.get(S[i], '?')}_{i}")
    
    transitions = []
    for i in range(N):
        for j in range(N):
            if P[i,j] >= threshold:
                transitions.append((i, j, P[i,j]))
    
    transitions.sort(key=lambda x: (x[0], -x[2]))
    
    # Format detailed transitions
    output_lines = ["Transition Probabilities (sorted by source state):\n",
                    f"{'From':<15} {'To':<15} {'Symbol':<10} {'Probability':<10}\n",
                    "-" * 55 + "\n"]
    current_source = -1
    for i, j, prob in transitions:
        if i != current_source:
            if current_source != -1: output_lines.append("\n")
            current_source = i
        symbol = "End" if j == 1 else int_to_symbol.get(S[j], '?')
        output_lines.append(f"{state_names[i]:<15} {state_names[j]:<15} {symbol:<10} {prob:.4f}\n")
    
    result = ''.join(output_lines)
    print("\n" + result)
    
    # Save detailed transitions to file
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(result)
            print(f"Detailed transition probabilities saved to {output_file}")
        except IOError as e:
            print(f"Error writing to {output_file}: {e}")

    # Group transitions by source and destination symbols
    sym_transitions = {}
    for i in range(2, N):
        for j in range(1, N):
            if P[i,j] >= threshold:
                source_sym = int_to_symbol.get(S[i], '?')
                dest_sym = "End" if j == 1 else int_to_symbol.get(S[j], '?')
                key = (source_sym, dest_sym)
                sym_transitions[key] = sym_transitions.get(key, 0) + P[i,j]
    
    sorted_sym_transitions = sorted(sym_transitions.items(), key=lambda x: (x[0][0], -x[1]))
    
    # Format symbol-level transitions
    sym_output_lines = ["\nSequence-level transition probabilities:\n",
                        f"{'From Symbol':<15} {'To Symbol':<15} {'Probability':<10}\n",
                        "-" * 55 + "\n"]
    current_source_sym = None
    for (source, dest), prob in sorted_sym_transitions:
        if source != current_source_sym:
            if current_source_sym is not None: sym_output_lines.append("\n")
            current_source_sym = source
        sym_output_lines.append(f"{source:<15} {dest:<15} {prob:.4f}\n")
    
    sym_result = ''.join(sym_output_lines)
    print(sym_result)
    
    # Save symbol-level transitions to a separate file
    if output_file:
        base, _ = os.path.splitext(output_file)
        sym_file = f"{base}_symbol_transitions.txt"
        try:
            with open(sym_file, 'w') as f:
                f.write(sym_result)
            print(f"Symbol transition probabilities saved to {sym_file}")
        except IOError as e:
            print(f"Error writing to {sym_file}: {e}")
            
    return transitions, sym_transitions

def convert_sequences(sequences):
    """Convert symbolic sequences to numeric form required by POMM library."""
    all_symbols = sorted(list(set(symbol for seq in sequences for symbol in seq)))
    symbol_to_int = {sym: i + 1 for i, sym in enumerate(all_symbols)}
    int_to_symbol = {i: sym for sym, i in symbol_to_int.items()}
    numeric_sequences = [[symbol_to_int[sym] for sym in seq] for seq in sequences]
    return numeric_sequences, symbol_to_int, int_to_symbol

def calculate_mean_transition_entropy(S, P):
    """
    Calculates the mean normalized transition entropy for a given POMM.

    Args:
        S: State vector.
        P: Transition probability matrix.

    Returns:
        The mean normalized transition entropy for the model.
    """
    entropies = []
    
    # Iterate over each state in the model
    for i in range(len(S)):
        # The end state (i=1) has no outgoing transitions, so we skip it.
        if i == 1:
            continue

        # Get the probabilities of all outgoing transitions from state i
        # Use a small epsilon to handle floating point inaccuracies
        probabilities = P[i, P[i, :] > 1e-10]
        
        # M is the number of possible branches (outgoing transitions)
        M = len(probabilities)

        if M <= 1:
            # If there's 0 or 1 branch, entropy is 0 (completely predictable)
            entropies.append(0.0)
            continue

        # Calculate Shannon entropy for the transitions from state i
        # H(X) = -Σ p(x) * log2(p(x))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize the entropy by the maximum possible entropy for M branches (log2(M))
        normalized_entropy = entropy / np.log2(M)
        entropies.append(normalized_entropy)

    # Return the mean of all the normalized entropies
    return np.mean(entropies)
    
if __name__ == "__main__":
    main()
