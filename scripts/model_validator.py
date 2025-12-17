#!/usr/bin/env python3
"""
Comprehensive Model Validator for DL Mutants Realism Study

Performs two types of validation:
1. Instance Count Validation: Quarantines mutant folders with insufficient .h5 files
2. Model Functionality Validation: Quarantines models that produce NaN/Inf outputs

All problematic models are moved to organized quarantine folders.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import shutil
tf.get_logger().setLevel('ERROR')

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Bug configuration for functionality validation

#============ CleanML Dataset ============
'''
BUG_CONFIG = {
    'cleanml001': 'classification',
    'cleanml002': 'classification',
    'cleanml003': 'classification',
    'cleanml004': 'classification', 
    'cleanml005': 'classification', 
    'cleanml006': 'classification', 
    'cleanml007': 'classification',
    'cleanml008': 'classification', 
    'cleanml009': 'classification',
    'cleanml010': 'classification', 
    'cleanml011': 'classification',
    'cleanml012': 'classification',
    'cleanml013': 'classification', 
    'cleanml014': 'classification', 
    'cleanml015': 'classification',
    'cleanml016': 'classification',
    'cleanml017': 'classification', 
    'cleanml018': 'classification',
    'cleanml019': 'classification'
}
'''

#============ DeepFD Dataset ============
'''
BUG_CONFIG = {
    'deepfd002': 'classification',
    'deepfd006': 'multi-label',
    'deepfd008': 'classification',
    'deepfd010': 'classification',
    'deepfd011': 'classification',
    'deepfd014': 'classification',
    'deepfd015': 'classification',
    'deepfd016': 'classification',
    'deepfd021': 'classification',
    'deepfd022': 'classification',
    'deepfd023': 'classification',
    'deepfd024': 'classification',
    'deepfd026': 'classification',
    'deepfd027': 'classification',
    'deepfd028': 'classification',
    'deepfd029': 'classification',
    'deepfd030': 'classification',
    'deepfd031': 'multi-label',
    'deepfd034': 'classification',
    'deepfd035': 'classification',
    'deepfd036': 'classification',
    'deepfd037': 'classification',
    'deepfd047': 'classification',
    'deepfd048': 'classification',
    'deepfd050': 'classification'
}
'''
#============ DeepLocalize Dataset ============
'''
BUG_CONFIG = {
    'deeplocalize002': 'classification',
    'deeplocalize006': 'multi-label',
    'deeplocalize008': 'classification', 
    'deeplocalize010': 'classification', 
    'deeplocalize011': 'classification',
    'deeplocalize015': 'classification',
    'deeplocalize021': 'classification',
    'deeplocalize024': 'classification',
    'deeplocalize026': 'classification',
    'deeplocalize027': 'classification',
    'deeplocalize028': 'classification',
    'deeplocalize030': 'classification',
    'deeplocalize031': 'classification', 
    'deeplocalize034': 'classification',
    'deeplocalize035': 'classification',
    'deeplocalize038': 'classification',
    'deeplocalize039': 'classification',
    'deeplocalize040': 'classification'
}
'''

#============ defect4ML Dataset ============
'''
BUG_CONFIG = {
    'defect4ml010': 'classification',
    'defect4ml011': 'classification',
    'defect4ml026': 'classification',
    'defect4ml038': 'classification',
    'defect4ml041': 'classification',
    'defect4ml044': 'classification',
    'defect4ml047': 'classification', 
    'defect4ml048': 'classification',
    'defect4ml050': 'classification',
    'defect4ml051': 'classification', 
    'defect4ml052': 'classification',
    'defect4ml053': 'classification',
    'defect4ml054': 'classification',
    'defect4ml061': 'classification',
    'defect4ml067': 'multi-label',
    'defect4ml068': 'classification',
    'defect4ml069': 'classification',
    'defect4ml074': 'classification',
    'defect4ml075': 'classification',
    'defect4ml088': 'classification',
    'defect4ml091': 'classification',
    'defect4ml095': 'classification',
    'defect4ml098': 'classification',
    'defect4ml100': 'classification' 
}
'''

BUG_CONFIG = { 
    'deepfd035': 'classification'
}

# Root directory containing bug subdirectories
MODELS_ROOT = 'models/'

# Mutant type to validate - choose one:
MUTANT_TYPE = 'pre-training'
#MUTANT_TYPE = 'post-training/scenario1_mutants'
#MUTANT_TYPE = 'post-training/scenario2_mutants'

# Instance count validation parameters
INSTANCE_THRESHOLD = 5  # Quarantine folders with fewer than this many .h5 files accross all pre- and post- mutants

# Functionality validation parameters
VALIDATION_SAMPLE_SIZE = 32  # Number of test inputs to use for validation
BATCH_SIZE = 32

# Quarantine structure
QUARANTINE_ROOT = os.path.join(MODELS_ROOT, "_quarantine")
INSUFFICIENT_INSTANCES_DIR = os.path.join(QUARANTINE_ROOT, "insufficient_instances")
PROBLEMATIC_OUTPUTS_DIR = os.path.join(QUARANTINE_ROOT, "problematic_outputs")

# Execution mode
DRY_RUN = False  # True -> preview only, False -> actually move files

# =============================================================================

# Custom objects for loading models with deprecated functions
def cosine_proximity(y_true, y_pred):
    """Deprecated cosine_proximity loss function for compatibility."""
    return -tf.keras.utils.cosine_similarity(y_true, y_pred, axis=-1)

CUSTOM_OBJECTS = {
    'cosine_proximity': cosine_proximity
}

def count_h5_recursive(folder: str) -> int:
    """Count all .h5 files recursively under folder."""
    total = 0
    for root, _, files in os.walk(folder):
        total += sum(1 for f in files if f.lower().endswith(".h5"))
    return total

def unique_dest_dir(base_dir: str, name: str) -> str:
    """Return a unique destination path under base_dir."""
    candidate = os.path.join(base_dir, name)
    if not os.path.exists(candidate):
        return candidate
    n = 1
    while True:
        candidate_n = os.path.join(base_dir, f"{name}_{n}")
        if not os.path.exists(candidate_n):
            return candidate_n
        n += 1

def load_test_data(test_data_path):
    """Load test inputs for validation."""
    if test_data_path.endswith('.csv'):
        df = pd.read_csv(test_data_path)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if feature_cols:
            X = df[feature_cols].values
        else:
            X = df.iloc[:, :-1].values
    elif test_data_path.endswith('.npz'):
        data = np.load(test_data_path)
        X = data['X']
    else:
        raise ValueError(f"Unsupported test data format: {test_data_path}")
    
    return X

def validate_instance_counts(bug_id):
    """
    Validate instance counts for mutants and quarantine insufficient ones.
    Returns list of quarantined mutant paths.
    """
    quarantined_mutants = []
    mutants_dir = os.path.join(MODELS_ROOT, bug_id, "mutants")

    if not os.path.isdir(mutants_dir):
        print(f"[SKIP] {bug_id}: mutants folder not found")
        return quarantined_mutants

    # Collect all nested subfolders under 'mutants'
    candidate_dirs = []
    for dirpath, dirnames, _ in os.walk(mutants_dir):
        for d in dirnames:
            candidate_dirs.append(os.path.join(dirpath, d))

    # Sort deepest first so moving parents won't break the scan
    candidate_dirs.sort(key=lambda p: len(p.split(os.sep)), reverse=True)

    print(f"\n=== Instance Count Validation: {bug_id} ===")
    if not candidate_dirs:
        print("No mutant subfolders found.")
        return quarantined_mutants

    for cand in candidate_dirs:
        h5_count = count_h5_recursive(cand)

        if h5_count < INSTANCE_THRESHOLD:
            # Build quarantine path
            rel_path = os.path.relpath(cand, MODELS_ROOT)
            quarantine_target = unique_dest_dir(INSUFFICIENT_INSTANCES_DIR, rel_path.replace(os.sep, "_"))

            print(f"[QUARANTINE] {cand} -> {quarantine_target} (h5={h5_count})")
            if not DRY_RUN:
                os.makedirs(os.path.dirname(quarantine_target), exist_ok=True)
                shutil.move(cand, quarantine_target)
            
            quarantined_mutants.append(cand)
        else:
            print(f"[KEEP] {cand} (h5={h5_count})")

    return quarantined_mutants

def validate_model_group(model_dir, group_name, X_test_sample):
    """
    Validate ALL model instances in a group for NaN/Inf outputs.
    """
    if not os.path.exists(model_dir):
        return {
            'functional': False,
            'reason': 'Directory does not exist',
            'group_name': group_name
        }
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not model_files:
        return {
            'functional': False,
            'reason': 'No .h5 model files found',
            'group_name': group_name
        }
    
    model_files = sorted(model_files)
    problematic_models = []
    
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(model_dir, model_file)
        
        try:
            # Load model
            model = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
            
            # Test predictions
            predictions = model.predict(X_test_sample, batch_size=BATCH_SIZE, verbose=0)
            
            # Zero tolerance check for NaN/Inf
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                nan_count = np.sum(np.isnan(predictions))
                inf_count = np.sum(np.isinf(predictions))
                total_outputs = predictions.size
                
                problematic_models.append({
                    'filename': model_file,
                    'instance': i,
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'total_outputs': total_outputs
                })
            
            # Clean up memory
            del model
            
        except Exception as e:
            problematic_models.append({
                'filename': model_file,
                'instance': i,
                'error': str(e)
            })
    
    # If any model instance failed, the entire group is non-functional
    if problematic_models:
        if len(problematic_models) == 1:
            prob = problematic_models[0]
            if 'error' in prob:
                reason = f"Model {prob['filename']} (instance {prob['instance']}) loading/prediction error: {prob['error']}"
            else:
                reason = f"Model {prob['filename']} (instance {prob['instance']}) produces {prob['nan_count']} NaN and {prob['inf_count']} Inf outputs"
        else:
            reason = f"{len(problematic_models)} out of {len(model_files)} instances produce NaN/Inf or have errors"
        
        return {
            'functional': False,
            'reason': reason,
            'group_name': group_name,
            'problematic_models': problematic_models,
            'total_instances': len(model_files)
        }
    
    # All model instances passed
    return {
        'functional': True,
        'group_name': group_name,
        'total_instances': len(model_files),
        'tested_instances': len(model_files)
    }

def validate_functionality(bug_id, quarantined_mutants):
    """
    Validate model functionality (NaN/Inf detection) and quarantine problematic ones.
    Skip mutants that were already quarantined for instance counts.
    """
    print(f"\n=== Functionality Validation: {bug_id} ===")
    
    bug_path = os.path.join(MODELS_ROOT, bug_id)
    if not os.path.exists(bug_path):
        print(f"Bug directory does not exist: {bug_path}")
        return
    
    # Load test data
    test_data_path = os.path.join(bug_path, f"test_data_{bug_id}.npz")
    if not os.path.exists(test_data_path):
        print(f"Test data not found: {test_data_path}")
        return
    
    try:
        X_test = load_test_data(test_data_path)
        X_sample = X_test[:min(VALIDATION_SAMPLE_SIZE, len(X_test))]
        print(f"Using {len(X_sample)} test inputs for validation")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    quarantined_count = 0

    # Validate Original models
    original_path = os.path.join(bug_path, 'original')
    result = validate_model_group(original_path, 'Original', X_sample)
    
    if result['functional']:
        print(f"✅ Original models: OK ({result['total_instances']} instances)")
    else:
        print(f"❌ Original models: {result['reason']}")
        if not DRY_RUN:
            quarantine_target = unique_dest_dir(PROBLEMATIC_OUTPUTS_DIR, f"{bug_id}_original")
            shutil.move(original_path, quarantine_target)
            print(f"[QUARANTINED] {original_path} -> {quarantine_target}")
        quarantined_count += 1

    # Validate Faulty models
    faulty_path = os.path.join(bug_path, 'faulty')
    result = validate_model_group(faulty_path, 'Faulty', X_sample)
    
    if result['functional']:
        print(f"✅ Faulty models: OK ({result['total_instances']} instances)")
    else:
        print(f"❌ Faulty models: {result['reason']}")
        if not DRY_RUN:
            quarantine_target = unique_dest_dir(PROBLEMATIC_OUTPUTS_DIR, f"{bug_id}_faulty")
            shutil.move(faulty_path, quarantine_target)
            print(f"[QUARANTINED] {faulty_path} -> {quarantine_target}")
        quarantined_count += 1

    # Validate Mutant models (skip those already quarantined for instance counts)
    mutants_path = os.path.join(bug_path, 'mutants', MUTANT_TYPE)
    if os.path.exists(mutants_path):
        mutant_dirs = [d for d in os.listdir(mutants_path) 
                      if os.path.isdir(os.path.join(mutants_path, d))]
        mutant_dirs = sorted(mutant_dirs)
        
        print(f"Found {len(mutant_dirs)} mutant groups to validate")
        
        for mutant_dir in mutant_dirs:
            mutant_path = os.path.join(mutants_path, mutant_dir)
            
            # Skip if already quarantined for instance count
            if mutant_path in quarantined_mutants:
                print(f"⏭️  Mutant {mutant_dir}: Skipped (already quarantined for insufficient instances)")
                continue
                
            result = validate_model_group(mutant_path, f'Mutant_{mutant_dir}', X_sample)
            
            if result['functional']:
                print(f"✅ Mutant {mutant_dir}: OK ({result['total_instances']} instances)")
            else:
                print(f"❌ Mutant {mutant_dir}: {result['reason']}")
                if not DRY_RUN:
                    quarantine_target = unique_dest_dir(PROBLEMATIC_OUTPUTS_DIR, f"{bug_id}_mutants_{MUTANT_TYPE.replace('/', '_')}_{mutant_dir}")
                    shutil.move(mutant_path, quarantine_target)
                    print(f"[QUARANTINED] {mutant_path} -> {quarantine_target}")
                quarantined_count += 1
    else:
        print(f"⚠️  Mutants directory not found: {mutants_path}")

    return quarantined_count

def main():
    """Main validation function."""
    # Setup quarantine directories
    os.makedirs(INSUFFICIENT_INSTANCES_DIR, exist_ok=True)
    os.makedirs(PROBLEMATIC_OUTPUTS_DIR, exist_ok=True)
    
    bug_list = list(BUG_CONFIG.keys())
    
    print("Comprehensive Model Validation Configuration:")
    print(f"  Bugs to validate: {bug_list}")
    print(f"  Models root: {MODELS_ROOT}")
    print(f"  Mutant type: {MUTANT_TYPE}")
    print(f"  Instance threshold: {INSTANCE_THRESHOLD}")
    print(f"  Validation sample size: {VALIDATION_SAMPLE_SIZE}")
    print(f"  Quarantine root: {QUARANTINE_ROOT}")
    print(f"  Dry run mode: {DRY_RUN}")
    
    total_instance_quarantined = 0
    total_functionality_quarantined = 0
    
    # Process each bug
    for bug_id in bug_list:
        print(f"\n{'='*80}")
        print(f"PROCESSING BUG: {bug_id}")
        print(f"{'='*80}")
        
        # Step 1: Instance count validation
        quarantined_mutants = validate_instance_counts(bug_id)
        total_instance_quarantined += len(quarantined_mutants)
        
        # Step 2: Functionality validation (skip already quarantined mutants)
        functionality_quarantined = validate_functionality(bug_id, quarantined_mutants)
        total_functionality_quarantined += functionality_quarantined if functionality_quarantined else 0
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total bugs processed: {len(bug_list)}")
    print(f"Quarantined for insufficient instances: {total_instance_quarantined}")
    print(f"Quarantined for problematic outputs: {total_functionality_quarantined}")
    print(f"Total quarantined model groups: {total_instance_quarantined + total_functionality_quarantined}")
    
    if total_instance_quarantined + total_functionality_quarantined > 0:
        print(f"\n🚨 QUARANTINE LOCATIONS:")
        print(f"  Instance issues: {INSUFFICIENT_INSTANCES_DIR}")
        print(f"  Output issues: {PROBLEMATIC_OUTPUTS_DIR}")
        print(f"After reviewing quarantined models, you can proceed with execution matrix generation.")
    else:
        print(f"\n✅ ALL MODEL GROUPS PASSED VALIDATION!")
        print(f"You can proceed with execution matrix generation.")

if __name__ == "__main__":
    main()