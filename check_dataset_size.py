#!/usr/bin/env python3
"""
Quick check for dataset size after path sampling and tokenization.
"""

from hopwise.config import Config
from hopwise.data.utils import create_dataset, data_preparation
from hopwise.utils import init_seed, init_logger

def check_dataset_size():
    config = Config(model="KGGLM", dataset="ml-100k")
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    
    print("Creating dataset...")
    dataset_obj = create_dataset(config)
    
    print("Preparing data (path sampling)...")
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)
    
    # Check path dataset
    if hasattr(train_data.dataset, '_path_dataset') and train_data.dataset._path_dataset:
        path_count = len(train_data.dataset._path_dataset.split('\n'))
        print(f"Path dataset size: {path_count} paths")
    else:
        print("❌ No path dataset found!")
        return
    
    # Check tokenized dataset
    if hasattr(train_data.dataset, '_tokenized_dataset') and train_data.dataset._tokenized_dataset:
        tokenized_count = len(train_data.dataset._tokenized_dataset)
        print(f"Tokenized dataset size: {tokenized_count} samples")
        
        if tokenized_count == 0:
            print("❌ CRITICAL: Tokenized dataset is EMPTY!")
            print("This is why training fails silently.")
        else:
            print("✅ Tokenized dataset has samples")
    else:
        print("❌ No tokenized dataset found!")

if __name__ == "__main__":
    check_dataset_size()
