#!/usr/bin/env python3
"""
Similarity Analysis Calculator for DL Mutants Realism Analysis

This script calculates both Detectability Overlap (IoU) and Coupling Strength (CS) 
between mutants and real faults based on the killing probability matrix.
Developed to process individual bug files for realism study.
"""

import argparse
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

# =============================================================================
# CONFIGURATION SECTION - ADJUST THESE PARAMETERS
# =============================================================================

# Import bug configuration from execution matrix script
try:
    from execution_matrix import BUG_CONFIG
    BUG_LIST = list(BUG_CONFIG.keys())
except ImportError:
    # Fallback: manual configuration
    BUG_LIST = ['cleanml001', 'cleanml002']

# Root directories
RESULTS_ROOT = 'results/'

# Mutant types to process - should match execution matrix runs
MUTANT_TYPES = [
    #'pre-training',
    #'post-training/scenario1_mutants', 
    'post-training/scenario2_mutants'
]

# Statistical parameters
MIN_TEST_INPUTS = 5
CONFIDENCE_THRESHOLD = 0.50 #strictness: low 0.50, medium 0.20, high 0.10

# =============================================================================

def load_killing_probabilities(file_path):
    """Load the killing probability matrix and metadata from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return (data['killing_prob_matrix'], data['mutant_types'], 
            data['confidence_intervals'], data.get('mutant_type_info', {}),
            data.get('bug_id'), data.get('mutant_type'))

def calculate_detectability_overlap(killing_prob_matrix, mutant_types, confidence_intervals, 
                                   min_test_inputs=5, confidence_threshold=0.10):
    """
    Calculate detectability overlap (IoU) between mutants and faulty models.
    
    IoU(M_k, F_j, T) = sum(min(KP_M_k(t), KP_F_j(t))) / sum(max(KP_M_k(t), KP_F_j(t)))
    
    Parameters:
    - killing_prob_matrix: Matrix of killing probabilities
    - mutant_types: List of mutant type identifiers
    - confidence_intervals: Confidence intervals for probabilities
    - min_test_inputs: Minimum number of test inputs for reliable results
    - confidence_threshold: Maximum allowed error in confidence intervals
    
    Returns:
    - iou_matrix: Matrix of IoU values
    - mm_indices: Indices of mutant models
    - fm_indices: Indices of faulty models
    - reliable_ious: Boolean matrix indicating reliable IoU measurements
    """
    # Separate mutant models and faulty models
    mm_indices = [i for i, mt in enumerate(mutant_types) if mt.startswith('MM_')]
    fm_indices = [i for i, mt in enumerate(mutant_types) if mt.startswith('FM_')]
    
    num_mm = len(mm_indices)
    num_fm = len(fm_indices)
    
    # Initialize IoU matrix and reliability matrix
    iou_matrix = np.zeros((num_mm, num_fm))
    reliable_ious = np.zeros((num_mm, num_fm), dtype=bool)
    
    # For each pair of mutant and faulty model, calculate IoU
    for i, mm_idx in enumerate(mm_indices):
        for j, fm_idx in enumerate(fm_indices):
            # Get killing probabilities for this pair
            mm_probs = killing_prob_matrix[:, mm_idx]
            fm_probs = killing_prob_matrix[:, fm_idx]
            
            # Get confidence intervals
            mm_ci = confidence_intervals[:, mm_idx, :]
            fm_ci = confidence_intervals[:, fm_idx, :]
            
            # Identify test inputs with non-overlapping confidence intervals
            definitive_inputs = []
            for k in range(len(mm_probs)):
                mm_lb, mm_ub = mm_ci[k]
                fm_lb, fm_ub = fm_ci[k]
                
                # Check if confidence intervals don't overlap
                if mm_ub < fm_lb or mm_lb > fm_ub:
                    definitive_inputs.append(k)
            
            # Calculate IoU using all test inputs
            min_probs = np.minimum(mm_probs, fm_probs)
            max_probs = np.maximum(mm_probs, fm_probs)
            sum_min = np.sum(min_probs)
            sum_max = np.sum(max_probs)
            
            if sum_max > 0:
                iou = sum_min / sum_max
                iou_matrix[i, j] = iou
                
                # Check reliability
                if len(definitive_inputs) >= min_test_inputs:
                    # Also mark as reliable if we pass the confidence check
                    avg_ci_width = np.mean([mm_ub - mm_lb for mm_lb, mm_ub in mm_ci])
                    if avg_ci_width <= confidence_threshold:
                        reliable_ious[i, j] = True
    
    return iou_matrix, mm_indices, fm_indices, reliable_ious

def calculate_coupling_strength(killing_prob_matrix, mutant_types, confidence_intervals, 
                               min_test_inputs=5, confidence_threshold=0.10):
    """
    Calculate coupling strength between mutants and faulty models.
    
    CS(M_k, F_j, T) = sum(min(KP_M_k(t), KP_F_j(t))) / sum(KP_M_k(t))
    
    Parameters:
    - killing_prob_matrix: Matrix of killing probabilities
    - mutant_types: List of mutant type identifiers
    - confidence_intervals: Confidence intervals for probabilities
    - min_test_inputs: Minimum number of test inputs for reliable results
    - confidence_threshold: Maximum allowed error in confidence intervals
    
    Returns:
    - coupling_matrix: Matrix of coupling strengths
    - mm_indices: Indices of mutant models
    - fm_indices: Indices of faulty models
    - reliable_couplings: Boolean matrix indicating reliable coupling measurements
    """
    # Separate mutant models and faulty models
    mm_indices = [i for i, mt in enumerate(mutant_types) if mt.startswith('MM_')]
    fm_indices = [i for i, mt in enumerate(mutant_types) if mt.startswith('FM_')]
    
    num_mm = len(mm_indices)
    num_fm = len(fm_indices)
    
    # Initialize coupling matrix and reliability matrix
    coupling_matrix = np.zeros((num_mm, num_fm))
    reliable_couplings = np.zeros((num_mm, num_fm), dtype=bool)
    
    # For each pair of mutant and faulty model, calculate coupling strength
    for i, mm_idx in enumerate(mm_indices):
        for j, fm_idx in enumerate(fm_indices):
            # Get killing probabilities for this pair
            mm_probs = killing_prob_matrix[:, mm_idx]
            fm_probs = killing_prob_matrix[:, fm_idx]
            
            # Get confidence intervals
            mm_ci = confidence_intervals[:, mm_idx, :]
            fm_ci = confidence_intervals[:, fm_idx, :]
            
            # Identify test inputs with non-overlapping confidence intervals
            definitive_inputs = []
            for k in range(len(mm_probs)):
                mm_lb, mm_ub = mm_ci[k]
                fm_lb, fm_ub = fm_ci[k]
                
                # Check if confidence intervals don't overlap
                if mm_ub < fm_lb or mm_lb > fm_ub:
                    definitive_inputs.append(k)
            
            # Calculate coupling strength using all test inputs
            min_probs = np.minimum(mm_probs, fm_probs)
            sum_min = np.sum(min_probs)
            sum_mm = np.sum(mm_probs)
            
            if sum_mm > 0:
                coupling = sum_min / sum_mm
                coupling_matrix[i, j] = coupling
                
                # Check reliability
                if len(definitive_inputs) >= min_test_inputs:
                    # Also mark as reliable if we pass the confidence check
                    avg_ci_width = np.mean([mm_ub - mm_lb for mm_lb, mm_ub in mm_ci])
                    if avg_ci_width <= confidence_threshold:
                        reliable_couplings[i, j] = True
    
    return coupling_matrix, mm_indices, fm_indices, reliable_couplings

def process_single_bug_mutant_type(bug_id, mutant_type, results_root):
    """Process similarity analysis for a single bug and mutant type."""
    print(f"\n=== Processing Bug: {bug_id}, Mutant Type: {mutant_type} ===")
    
    # Generate mutant type suffix for filename
    mutant_suffix = mutant_type.replace('/', '_').replace('-', '_')
    
    # Load killing probabilities for this bug and mutant type
    killing_prob_file = os.path.join(results_root, "killing_probability", 
                                   f"killing_probability_{bug_id}_{mutant_suffix}.pkl")
    
    if not os.path.exists(killing_prob_file):
        print(f"Warning: Killing probability file not found: {killing_prob_file}")
        return None
        
    killing_prob_matrix, mutant_types, confidence_intervals, mutant_type_info, loaded_bug_id, loaded_mutant_type = load_killing_probabilities(killing_prob_file)
    print(f"Loaded killing probability matrix with shape {killing_prob_matrix.shape}")
    
    # Verify we loaded the correct file
    if loaded_bug_id != bug_id:
        print(f"Warning: Bug ID mismatch. Expected {bug_id}, got {loaded_bug_id}")
    if loaded_mutant_type != mutant_type:
        print(f"Warning: Mutant type mismatch. Expected {mutant_type}, got {loaded_mutant_type}")
    
    # Calculate detectability overlap (IoU)
    print("Calculating detectability overlap (IoU)...")
    iou_matrix, mm_indices, fm_indices, reliable_ious = calculate_detectability_overlap(
        killing_prob_matrix, mutant_types, confidence_intervals,
        MIN_TEST_INPUTS, CONFIDENCE_THRESHOLD
    )
    
    # Calculate coupling strength
    print("Calculating coupling strength...")
    coupling_matrix, mm_indices_cs, fm_indices_cs, reliable_couplings = calculate_coupling_strength(
        killing_prob_matrix, mutant_types, confidence_intervals,
        MIN_TEST_INPUTS, CONFIDENCE_THRESHOLD
    )
    
    # Verify indices are the same (they should be)
    assert mm_indices == mm_indices_cs and fm_indices == fm_indices_cs, "Index mismatch between IoU and Coupling calculations"
    
    return (iou_matrix, coupling_matrix, mutant_types, mm_indices, fm_indices, 
            reliable_ious, reliable_couplings)

def save_bug_results(iou_matrix, coupling_matrix, mutant_types, mm_indices, fm_indices, 
                    reliable_ious, reliable_couplings, bug_id, mutant_type, results_root):
    """Save the similarity analysis results for a single bug and mutant type."""
    # Create separate subdirectories for each metric
    iou_output_dir = os.path.join(results_root, "detectability_overlap")
    coupling_output_dir = os.path.join(results_root, "coupling_strength")
    os.makedirs(iou_output_dir, exist_ok=True)
    os.makedirs(coupling_output_dir, exist_ok=True)
    
    # Generate mutant type suffix for filename
    mutant_suffix = mutant_type.replace('/', '_').replace('-', '_')
    
    # Generate output filenames with bug ID and mutant type
    iou_output_file = os.path.join(iou_output_dir, f"detectability_overlap_{bug_id}_{mutant_suffix}.pkl")
    coupling_output_file = os.path.join(coupling_output_dir, f"coupling_strength_{bug_id}_{mutant_suffix}.pkl")
    
    # Save IoU results as pickle
    with open(iou_output_file, 'wb') as f:
        pickle.dump({
            'bug_id': bug_id,
            'mutant_type': mutant_type,
            'iou_matrix': iou_matrix,
            'mutant_types': mutant_types,
            'mm_indices': mm_indices,
            'fm_indices': fm_indices,
            'reliable_ious': reliable_ious
        }, f)
    
    # Save Coupling Strength results as pickle
    with open(coupling_output_file, 'wb') as f:
        pickle.dump({
            'bug_id': bug_id,
            'mutant_type': mutant_type,
            'coupling_matrix': coupling_matrix,
            'mutant_types': mutant_types,
            'mm_indices': mm_indices,
            'fm_indices': fm_indices,
            'reliable_couplings': reliable_couplings
        }, f)
    
    # Create readable column and row names
    mm_names = [mutant_types[i] for i in mm_indices]
    fm_names = [mutant_types[i] for i in fm_indices]
    
    # Convert to DataFrames
    iou_df = pd.DataFrame(iou_matrix, index=mm_names, columns=fm_names)
    coupling_df = pd.DataFrame(coupling_matrix, index=mm_names, columns=fm_names)
    
    # Mark reliable measurements
    reliable_iou_df = pd.DataFrame(reliable_ious, index=mm_names, columns=fm_names)
    reliable_coupling_df = pd.DataFrame(reliable_couplings, index=mm_names, columns=fm_names)
    
    # Create formatted DataFrames where each cell contains "value (reliable/unreliable)"
    formatted_iou_df = iou_df.copy()
    formatted_coupling_df = coupling_df.copy()
    
    for i, mm in enumerate(mm_names):
        for j, fm in enumerate(fm_names):
            # IoU formatting
            iou_value = iou_df.loc[mm, fm]
            iou_reliable = reliable_iou_df.loc[mm, fm]
            formatted_iou_df.loc[mm, fm] = f"{iou_value:.4f} ({'reliable' if iou_reliable else 'unreliable'})"
            
            # Coupling formatting
            coupling_value = coupling_df.loc[mm, fm]
            coupling_reliable = reliable_coupling_df.loc[mm, fm]
            formatted_coupling_df.loc[mm, fm] = f"{coupling_value:.4f} ({'reliable' if coupling_reliable else 'unreliable'})"
    
    # Save CSVs for IoU in detectability_overlap folder
    iou_csv_file = os.path.join(iou_output_dir, f"detectability_overlap_{bug_id}_{mutant_suffix}.csv")
    formatted_iou_df.to_csv(iou_csv_file)
    
    iou_clean_csv = os.path.join(iou_output_dir, f"detectability_overlap_{bug_id}_{mutant_suffix}_clean.csv")
    iou_df.to_csv(iou_clean_csv)
    
    # Save CSVs for Coupling Strength in coupling_strength folder
    coupling_csv_file = os.path.join(coupling_output_dir, f"coupling_strength_{bug_id}_{mutant_suffix}.csv")
    formatted_coupling_df.to_csv(coupling_csv_file)
    
    coupling_clean_csv = os.path.join(coupling_output_dir, f"coupling_strength_{bug_id}_{mutant_suffix}_clean.csv")
    coupling_df.to_csv(coupling_clean_csv)
    
    print(f"Bug {bug_id} ({mutant_type}) detectability overlap saved to {iou_output_file}")
    print(f"Bug {bug_id} ({mutant_type}) coupling strength saved to {coupling_output_file}")
    print(f"Bug {bug_id} ({mutant_type}) IoU CSV saved to {iou_csv_file}")
    print(f"Bug {bug_id} ({mutant_type}) Coupling CSV saved to {coupling_csv_file}")
    
    return iou_output_file, coupling_output_file

def main():
    print(f"Configuration:")
    print(f"  Bug list: {BUG_LIST}")
    print(f"  Results root: {RESULTS_ROOT}")
    print(f"  Mutant types: {MUTANT_TYPES}")
    print(f"  Min test inputs: {MIN_TEST_INPUTS}")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    
    print(f"\nProcessing similarity analysis for all bugs and mutant types...")
    
    # Process each combination of bug and mutant type
    processed_count = 0
    for bug_id in BUG_LIST:
        for mutant_type in MUTANT_TYPES:
            result = process_single_bug_mutant_type(bug_id, mutant_type, RESULTS_ROOT)
            
            if result is not None:
                (iou_matrix, coupling_matrix, mutant_types, mm_indices, fm_indices, 
                 reliable_ious, reliable_couplings) = result
                
                # Save individual bug results
                save_bug_results(iou_matrix, coupling_matrix, mutant_types, mm_indices, fm_indices,
                               reliable_ious, reliable_couplings, bug_id, mutant_type, RESULTS_ROOT)
                
                # Print summary for this combination
                #print(f"Bug {bug_id} ({mutant_type}) - IoU Matrix Shape: {iou_matrix.shape}")
                #print(f"Bug {bug_id} ({mutant_type}) - Coupling Matrix Shape: {coupling_matrix.shape}")
                print(f"Bug {bug_id} ({mutant_type}) - Mutant models: {len(mm_indices)}")
                #print(f"Bug {bug_id} ({mutant_type}) - Faulty models: {len(fm_indices)}")
                print(f"Bug {bug_id} ({mutant_type}) - Mean IoU: {np.mean(iou_matrix):.4f}")
                print(f"Bug {bug_id} ({mutant_type}) - Max IoU: {np.max(iou_matrix):.4f}")
                print(f"Bug {bug_id} ({mutant_type}) - Mean Coupling Strength: {np.mean(coupling_matrix):.4f}")
                print(f"Bug {bug_id} ({mutant_type}) - Max Coupling Strength: {np.max(coupling_matrix):.4f}")
                #print(f"Bug {bug_id} ({mutant_type}) - Reliable IoU measurements: {np.sum(reliable_ious)} out of {iou_matrix.size}")
                #print(f"Bug {bug_id} ({mutant_type}) - Reliable Coupling measurements: {np.sum(reliable_couplings)} out of {coupling_matrix.size}")
                
                processed_count += 1
            else:
                print(f"Skipping bug {bug_id}, mutant type {mutant_type} due to errors")
    
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Successfully processed {processed_count} bug-mutant type combinations")
    print(f"Detectability overlap results saved in: {os.path.join(RESULTS_ROOT, 'detectability_overlap')}")
    print(f"Coupling strength results saved in: {os.path.join(RESULTS_ROOT, 'coupling_strength')}")

if __name__ == "__main__":
    main()