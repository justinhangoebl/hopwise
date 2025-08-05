#!/usr/bin/env python3
"""
Fast debug script that limits the dataset to just a few users.
"""

import os
import sys
import pandas as pd

# Add the hopwise directory to Python path
sys.path.insert(0, '/mnt/c/Users/Justin/Desktop/hopwise')

def create_tiny_dataset():
    """Create a tiny version of ml-100k for debugging."""
    print("🔧 Creating tiny dataset for debugging...")
    
    # Read the original ml-100k data
    data_path = "/mnt/c/Users/Justin/Desktop/hopwise/dataset/ml-100k"
    
    # Read interactions
    inter_file = os.path.join(data_path, "ml-100k.inter")
    if os.path.exists(inter_file):
        df = pd.read_csv(inter_file, sep='\t')
        print(f"Original dataset: {len(df)} interactions, {df['user_id'].nunique()} users")
        
        # Keep only first 3 users
        tiny_users = df['user_id'].unique()[:3]
        tiny_df = df[df['user_id'].isin(tiny_users)]
        
        # Save tiny version
        tiny_file = os.path.join(data_path, "ml-100k-tiny.inter")
        tiny_df.to_csv(tiny_file, sep='\t', index=False)
        print(f"Tiny dataset: {len(tiny_df)} interactions, {tiny_df['user_id'].nunique()} users")
        
        return True
    else:
        print(f"❌ Could not find {inter_file}")
        return False

def debug_with_tiny_data():
    """Debug KGGLM with tiny dataset."""
    try:
        print("🔍 Starting fast KGGLM debug with tiny dataset...")
        
        # Create tiny dataset first
        if not create_tiny_dataset():
            return False
        
        # Import hopwise components
        from hopwise.config import Config
        from hopwise.data.utils import create_dataset, data_preparation
        from hopwise.utils import init_seed, init_logger
        
        print("\n📋 Creating config...")
        # Use our debug dataset config
        config = Config(model="KGGLM", dataset="ml-100k-debug")
        
        try:
            train_stage = config['train_stage']
            print(f"✅ Config created - train_stage: {train_stage}")
        except:
            print("✅ Config created - train_stage: Not set")
        
        print("\n📋 Initializing...")
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        
        print("\n📊 Creating dataset...")
        dataset_obj = create_dataset(config)
        print(f"✅ Dataset created")
        print(f"   - Users: {dataset_obj.user_num}")
        print(f"   - Items: {dataset_obj.item_num}")
        if hasattr(dataset_obj, 'entity_num'):
            print(f"   - Entities: {dataset_obj.entity_num}")
        
        print("\n🛤️  Starting path sampling (should be fast now)...")
        train_data, valid_data, test_data = data_preparation(config, dataset_obj)
        print("✅ Path sampling completed!")
        
        # Check results
        print("\n🔍 Checking results...")
        
        # Check path dataset
        if hasattr(train_data.dataset, '_path_dataset'):
            if train_data.dataset._path_dataset is not None:
                paths = train_data.dataset._path_dataset.split('\n')
                non_empty_paths = [p for p in paths if p.strip()]
                print(f"✅ Path dataset: {len(non_empty_paths)} paths")
                
                if len(non_empty_paths) > 0:
                    print(f"   Sample paths:")
                    for i, path in enumerate(non_empty_paths[:3]):
                        print(f"     {i+1}: {path}")
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
                    print("   This is why training stops silently!")
                    
                    # Let's check why tokenization failed
                    print("\n🔍 Debugging tokenization...")
                    if hasattr(train_data.dataset, 'tokenizer'):
                        tokenizer = train_data.dataset.tokenizer
                        print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")
                        print(f"   - Special tokens: {tokenizer.special_tokens_map}")
                        
                        # Try tokenizing a sample path manually
                        if len(non_empty_paths) > 0:
                            sample_path = non_empty_paths[0]
                            print(f"   - Sample path: {sample_path}")
                            try:
                                tokenized = tokenizer(sample_path, return_tensors="pt")
                                print(f"   - Tokenized shape: {tokenized['input_ids'].shape}")
                                print(f"   - Tokenized IDs: {tokenized['input_ids'][0][:10]}...")
                            except Exception as e:
                                print(f"   - Tokenization error: {e}")
                else:
                    print("✅ Tokenized dataset has samples - this is good!")
            else:
                print("❌ Tokenized dataset is None!")
        else:
            print("❌ No tokenized dataset attribute!")
        
        # Check data loader
        print(f"\n📦 Data loader info:")
        print(f"   - Train batches: {len(train_data)}")
        
        try:
            sample = next(iter(train_data))
            print(f"   - Sample keys: {list(sample.keys())}")
            if 'input_ids' in sample:
                print(f"   - Input shape: {sample['input_ids'].shape}")
                print(f"   - Sample input_ids: {sample['input_ids'][0][:10]}...")
        except Exception as e:
            print(f"❌ Error getting sample: {e}")
        
        print("\n🎉 Fast debug completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_with_tiny_data()
