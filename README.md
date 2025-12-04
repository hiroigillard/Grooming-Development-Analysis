# Behavioral Analysis Pipeline for Mouse Grooming Ontogeny

## Overview

This repository contains a suite of Python tools developed to analyze the ontogeny of stereotyped grooming behavior in mice. The pipeline bridges the gap between manual scoring (BORIS) and automated deep learning analysis (**DeepEthogram**), providing tools for data transformation, Markov modeling, and sequence validation.

This codebase was developed as part of my MSci Pharmacology research project at UCL (Koch Lab) to investigate striatal circuit development.

## Pipeline Workflow

### 1. Data Preprocessing & Transformation

Tools to convert raw data between different ethological software formats.

* **`boris_to_deg.py`**: Converts BORIS observation files (`.tsv`) into the specific frame-by-frame CSV format required to train DeepEthogram (https://github.com/jbohnslav/deepethogram). Handles gap filling and frame-rate conversion.
* **`deg_to_theme.py`**:  Transforms DeepEthogram inference outputs (binary CSVs) into event files compatible with **THEME (PatternVision)** for T-pattern analysis.

### 2. Sequence Extraction

Scripts to convert raw frame-by-frame probability maps into discrete behavioral sequence strings (e.g., "1-2-1-5").

* **`extract_sequences.py`**: **Master Extractor.** Offers granular control with independent parameters for pre-sequence quiet periods, post-sequence quiet periods, and maximum sequence duration limits.
* **`extract_sequences_ind.py`**: Optimized pipeline for processing individual files and generating subject-specific sequence logs for validation.

### 3. Transition Analysis (Raw Data)

Analyzing transitions directly from the time-series CSVs produced by DeepEthogram.

* **`calc_transitions.py`**: Calculates **Joint**, **Conditional**, and **Raw Count** transition matrices from group data. Includes heatmap generation.
* **`calc_transitions_bg.py`**: Variation that includes a **"Background"** state, tracking transitions into and out of inactivity.
* **`calc_transitions_ind.py`** & **`calc_transitions_ind_bg.py`**: High-throughput versions that calculate transition matrices for *every individual file* in a dataset (with/without background) and generate aggregate summary tables.
* **`prob_start_end.py`**: Specialized script for calculating conditional transition probabilities, explicitly modeling **Start** and **End** probabilities for behavioral bouts.

### 4. Transition Analysis (Sequence Data)

Analyzing transitions based on the extracted text sequences (from Step 2).

* **`calc_seq_transitions.py`**: Computes transition probabilities directly from `.txt` sequence files rather than raw time-series data.
* **`model_group_matrix.py`**: **Core Modeling Script.** Calculates individual transition matrices from sequences, normalizes them, and computes the **Group Median Matrix**. This represents the "canonical" grooming model for a specific experimental group.

### 5. Ethological Metrics

Calculations for specific features of behavioral rigidity, complexity, and state occupancy.

* **`calc_kl_divergence.py`**: Calculates **Kullback-Leibler (KL) Divergence** to quantify how much an individual animal's behavioral distribution diverges from the group mean.
* **`calc_entropy.py`**: Computes **Shannon Entropy**, stationary distributions ($\mu$), and the **second largest eigenvalue** from transition matrices to quantify system stability.
* **`calc_durations.py`**: Calculates the total duration (in seconds) of specific behavioral phases across multiple files.
* **`calc_exit_probs.py`**: Analyzes sequences to calculate the probability of "exiting" (terminating) a specific behavioral state.

### 6. Model Validation (Generative)

Validating the Markov models by comparing synthetic data against real-world observations.

* **`generate_seqs.py`**: Generates synthetic behavioral sequences based on the calculated transition matrices and compares them against real animal data using **Kolmogorov-Smirnov (K-S) tests** and KDE plots.
* **`generate_seqs_individual.py`**: Performs model validation at the *individual animal level*, matching specific subject matrices to their own behavioral output to test model fit personalized to each subject.

### 7. Partially Observable Markov Model Analysis & Significance Testing
Advanced statistical validation using the **P-beta** metric to test if observed sequences are significantly more likely under a given POMM than expected by chance. (Requires the POMM-Python repository https://github.com/djinpsu/POMM-Python)

* **`pomm_inference.py`**: A wrapper for the POMM library that infers an optimal POMM structure (n-gram search) from raw sequences, simplifies the model, and exports transition diagrams and entropy metrics.
* **`run_pbeta_analysis.py`**: Performs **Group-Level** P-beta analysis. It tests if the sequences from a specific group (e.g., Veh) are a good fit for that group's median model.
* **`run_individual_pbeta_analysis.py`**: Performs **Paired Individual** analysis. It tests if a specific animal's sequences fit its *own* individual model significantly better than chance.
* **`run_longitudinal_analysis.py`**: Tracks model fit over time. It takes a reference model (e.g., an animal's P21 matrix) and tests how well that model explains sequences from earlier ages (P18, etc.) to quantify developmental drift.
## Dependencies

The analysis pipeline requires the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scipy
```
Note: The P-Beta analysis scripts (`run_pbeta_analysis.py`, etc.) require the POMM library and its compiled C dependencies (`libPOMM.so`) to be present in the working directory.
## Usage Examples

**Generating a Group Model:**
To calculate the median transition matrix for a group from extracted sequences:

```bash
python seq_to_avgmatrix.py
```

**Running KL Divergence Analysis:** 
First, calculate occupancy, then run the KL analysis:
```bash
python seq-ex_KL.py /path/to/csvs --output-dir ./occupancy_data
python KL.py ./occupancy_data/bout_avg_occupancy_summary.csv
```
## Author
Sean Hiroyuki Isomura-Gillard

* MSci Pharmacology, University College London
* Contact: hiroigillard@gmail.com

