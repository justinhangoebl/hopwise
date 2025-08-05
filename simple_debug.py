#!/usr/bin/env python3
"""
Simple debug script with minimal imports.
"""

import os
import sys

# Add the hopwise directory to Python path
sys.path.insert(0, '/mnt/c/Users/Justin/Desktop/hopwise')

try:
    print("🔍 Starting simple KGGLM debug...")
    
    # Step 1: Basic imports
    print("\n📋 Step 1: Testing imports...")
    from hopwise.config import Config
    print("✅ Config imported")
    
    from hopwise.data.utils import create_dataset, data_preparation
    print("✅ Data utils imported")
    
    from hopwise.utils import init_seed, init_logger
    print("✅ Utils imported")
    
    # Step 2: Create config
    print("\n📋 Step 2: Creating config...")
    config = Config(model="KGGLM", dataset="ml-100k")
    try:
        train_stage = config['train_stage']
        print(f"✅ Config created - train_stage: {train_stage}")
    except:
        print("✅ Config created - train_stage: Not set")
    
    # Step 3: Initialize
    print("\n📋 Step 3: Initializing...")
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    print("✅ Initialization complete")
    
    # Step 4: Create dataset
    print("\n📊 Step 4: Creating dataset...")
    dataset_obj = create_dataset(config)
    print(f"✅ Dataset created")
    print(f"   - Users: {dataset_obj.user_num}")
    print(f"   - Items: {dataset_obj.item_num}")
    if hasattr(dataset_obj, 'entity_num'):
        print(f"   - Entities: {dataset_obj.entity_num}")
    
    # Step 5: Data preparation (this is where path sampling happens)
    print("\n🛤️  Step 5: Data preparation (PATH SAMPLING)...")
    print("   This step includes path sampling - watch for any errors...")
    
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)
    print("✅ Data preparation completed!")
    
    # Step 6: Check what we got
    print("\n🔍 Step 6: Checking results...")
    print(f"   - Train data type: {type(train_data)}")
    print(f"   - Train dataset type: {type(train_data.dataset)}")
    
    # Check if it's a path dataset
    if hasattr(train_data.dataset, '_path_dataset'):
        if train_data.dataset._path_dataset is not None:
            paths = train_data.dataset._path_dataset.split('\n')
            non_empty_paths = [p for p in paths if p.strip()]
            print(f"✅ Path dataset found: {len(non_empty_paths)} paths")
            
            if len(non_empty_paths) > 0:
                print(f"   Sample path: {non_empty_paths[0][:100]}...")
            else:
                print("❌ No valid paths found!")
        else:
            print("❌ Path dataset is None!")
    else:
        print("❌ No path dataset attribute!")
    
    # Check tokenized dataset
    if hasattr(train_data.dataset, '_tokenized_dataset'):
        if train_data.dataset._tokenized_dataset is not None:
            tokenized_size = len(train_data.dataset._tokenized_dataset)
            print(f"✅ Tokenized dataset: {tokenized_size} samples")
            
            if tokenized_size == 0:
                print("❌ CRITICAL: Tokenized dataset is EMPTY!")
                print("   This is likely why training stops silently!")
            else:
                print("✅ Tokenized dataset has samples")
        else:
            print("❌ Tokenized dataset is None!")
    else:
        print("❌ No tokenized dataset attribute!")
    
    # Check data loader
    print(f"\n📦 Data loader info:")
    print(f"   - Train batches: {len(train_data)}")
    
    # Try to get a sample
    try:
        sample = next(iter(train_data))
        print(f"   - Sample keys: {list(sample.keys())}")
        if 'input_ids' in sample:
            print(f"   - Input shape: {sample['input_ids'].shape}")
    except Exception as e:
        print(f"❌ Error getting sample: {e}")
    
    print("\n🎉 Debug completed successfully!")
    print("If you see 'Tokenized dataset is EMPTY', that's your problem!")
    
except Exception as e:
    print(f"\n❌ Error during debug: {e}")
    import traceback
    traceback.print_exc()
