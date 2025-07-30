#!/usr/bin/env python3
"""
Debug script for KGGLM training issues.
This script helps identify where the training process fails after path sampling.
"""

import sys
import traceback
import logging
from hopwise.quick_start import run
from hopwise.config import Config
from hopwise.data.utils import create_dataset, data_preparation
from hopwise.utils import init_seed, init_logger
from hopwise.utils.logger import getLogger

def debug_kgglm_training(model="KGGLM", dataset="ml-100k"):
    """Debug KGGLM training step by step."""
    
    print("🔍 Starting KGGLM Training Debug...")
    
    try:
        # Step 1: Initialize configuration
        print("\n📋 Step 1: Loading configuration...")
        config = Config(model=model, dataset=dataset)
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        logger = getLogger()
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Model: {config['model']}")
        print(f"   - Dataset: {config['dataset']}")
        print(f"   - Train stage: {config.get('train_stage', 'Not set')}")
        
        # Step 2: Create dataset
        print("\n📊 Step 2: Creating dataset...")
        dataset_obj = create_dataset(config)
        print(f"✅ Dataset created successfully")
        print(f"   - User count: {dataset_obj.user_num}")
        print(f"   - Item count: {dataset_obj.item_num}")
        print(f"   - Entity count: {getattr(dataset_obj, 'entity_num', 'N/A')}")
        
        # Step 3: Data preparation (this includes path sampling)
        print("\n🛤️  Step 3: Data preparation (includes path sampling)...")
        train_data, valid_data, test_data = data_preparation(config, dataset_obj)
        print(f"✅ Data preparation completed")
        
        # Step 4: Check if path dataset was generated
        print("\n🔍 Step 4: Checking path dataset...")
        if hasattr(train_data.dataset, '_path_dataset'):
            if train_data.dataset._path_dataset is not None:
                path_count = len(train_data.dataset._path_dataset.split('\n'))
                print(f"✅ Path dataset generated: {path_count} paths")
                
                # Show sample paths
                sample_paths = train_data.dataset._path_dataset.split('\n')[:5]
                print("   Sample paths:")
                for i, path in enumerate(sample_paths):
                    if path.strip():
                        print(f"     {i+1}: {path[:100]}...")
            else:
                print("❌ Path dataset is None!")
                return False
        else:
            print("❌ No path dataset attribute found!")
            return False
        
        # Step 5: Check tokenized dataset
        print("\n🔤 Step 5: Checking tokenized dataset...")
        if hasattr(train_data.dataset, '_tokenized_dataset'):
            if train_data.dataset._tokenized_dataset is not None:
                tokenized_count = len(train_data.dataset._tokenized_dataset)
                print(f"✅ Tokenized dataset: {tokenized_count} samples")
                
                if tokenized_count == 0:
                    print("❌ CRITICAL: Tokenized dataset is empty!")
                    print("   This is likely why training fails silently.")
                    return False
                    
                # Check tokenizer
                tokenizer = train_data.dataset.tokenizer
                print(f"   - Tokenizer vocab size: {tokenizer.vocab_size}")
                print(f"   - Special tokens: {tokenizer.special_tokens_map}")
                
            else:
                print("❌ Tokenized dataset is None!")
                return False
        else:
            print("❌ No tokenized dataset attribute found!")
            return False
        
        # Step 6: Check data loader
        print("\n📦 Step 6: Checking data loader...")
        print(f"✅ Train data loader: {len(train_data)} batches")
        
        # Try to get first batch
        try:
            first_batch = next(iter(train_data))
            print(f"   - First batch keys: {list(first_batch.keys())}")
            if 'input_ids' in first_batch:
                print(f"   - Input IDs shape: {first_batch['input_ids'].shape}")
        except Exception as e:
            print(f"❌ Error getting first batch: {e}")
            return False
        
        # Step 7: Try model initialization
        print("\n🤖 Step 7: Initializing model...")
        from hopwise.model import get_model
        model_class = get_model(config["model"])
        model = model_class(config, train_data.dataset).to(config["device"])
        print(f"✅ Model initialized successfully")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Device: {config['device']}")
        
        # Step 8: Try trainer initialization
        print("\n🏃 Step 8: Initializing trainer...")
        from hopwise.trainer import get_trainer
        trainer_class = get_trainer(config["MODEL_TYPE"], config["model"])
        trainer = trainer_class(config, model)
        print(f"✅ Trainer initialized successfully")
        print(f"   - Trainer type: {type(trainer).__name__}")
        
        # Step 9: Try HF trainer initialization (this is where it might fail)
        print("\n🤗 Step 9: Initializing HuggingFace trainer...")
        try:
            trainer.init_hf_trainer(
                train_data,
                valid_data=valid_data,
                verbose=True,
                saved=True,
                show_progress=True
            )
            print(f"✅ HuggingFace trainer initialized successfully")
        except Exception as e:
            print(f"❌ CRITICAL: HuggingFace trainer initialization failed!")
            print(f"   Error: {e}")
            traceback.print_exc()
            return False
        
        print("\n🎉 All initialization steps completed successfully!")
        print("   The issue might be in the actual training loop.")
        print("   Try running with more verbose logging or smaller batch sizes.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in step: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "KGGLM"
    dataset = sys.argv[2] if len(sys.argv) > 2 else "ml-100k"
    
    success = debug_kgglm_training(model, dataset)
    
    if success:
        print("\n✅ Debug completed successfully - no obvious issues found")
        sys.exit(0)
    else:
        print("\n❌ Debug found critical issues - check the output above")
        sys.exit(1)
