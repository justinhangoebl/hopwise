# @Time   : 2025/01
# @Author : Integration Team
# @Email  : team@hopwise.ai

r"""KIGER Dataset
##################################################
Extended KG Path Dataset that integrates RQ-VAE semantic IDs
for item representation in knowledge graph paths.
"""

import re
from typing import List, Dict, Optional
from hopwise.data.dataset.kg_path_dataset import KnowledgePathDataset
from hopwise.utils import PathLanguageModelingTokenType, set_color


class KIGERDataset(KnowledgePathDataset):
    """Extended KG Path Dataset with RQ-VAE semantic item representations."""
    
    def __init__(self, config, dataset=None):
        # Store KIGER-specific configuration
        self.kiger_config = config["kiger"]
        self.use_semantic_ids = self.kiger_config["use_semantic_ids"]
        self.semantic_token_prefix = self.kiger_config["semantic_token_prefix"]
        self.n_quantization_layers = self.kiger_config["n_quantization_layers"]
        
        # Initialize parent dataset
        super().__init__(config)
        
        # Load semantic mapping if available
        self.semantic_mapping = None
        if self.use_semantic_ids:
            self._load_semantic_mapping()
            # Disable semantic IDs if mapping failed to load
            if self.semantic_mapping is None:
                self.logger.warning("Disabling semantic IDs due to mapping load failure")
                self.use_semantic_ids = False
    
    def _load_semantic_mapping(self):
        """Load semantic ID mapping for items."""
        import torch
        import os
        
        mapping_path = self.kiger_config["semantic_mapping_path"]
        self.logger.info(f"Attempting to load semantic mapping from: {mapping_path}")
        
        if not mapping_path:
            self.logger.error("No semantic mapping path provided in config")
            self.semantic_mapping = None
            return
            
        if not os.path.exists(mapping_path):
            self.logger.error(f"Semantic mapping file does not exist: {mapping_path}")
            self.semantic_mapping = None
            return
            
        try:
            self.logger.info(f"Loading semantic mapping from {mapping_path}")
            mapping_data = torch.load(mapping_path, map_location="cpu", weights_only=False)
            self.logger.info(f"Raw mapping data type: {type(mapping_data)}")
            
            # Handle the structured format from generate_semids.py
            if isinstance(mapping_data, dict) and 'semantic_ids' in mapping_data:
                semantic_ids_tensor = mapping_data['semantic_ids']
                num_items = mapping_data.get('num_items', len(semantic_ids_tensor))
                
                # Convert tensor to dictionary mapping: {item_id: [semantic_codes]}
                self.semantic_mapping = {}
                for item_id in range(len(semantic_ids_tensor)):
                    self.semantic_mapping[item_id] = semantic_ids_tensor[item_id].tolist()
                self.logger.info(f"Loaded semantic mapping for {len(self.semantic_mapping)} items of {num_items} items")
            else:
                # Handle direct dictionary format
                self.semantic_mapping = mapping_data
                self.logger.info(f"Loaded semantic mapping for {len(self.semantic_mapping)} items")
                
        except Exception as e:
            self.logger.error(f"Failed to load semantic mapping: {e}")
            self.logger.error(f"Exception type: {type(e)}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.semantic_mapping = None
    def _setup_tokenizer(self):
        """Setup tokenizer with semantic tokens."""
        # Call parent tokenizer setup
        super()._setup_tokenizer()
        
        # Extend vocabulary with semantic tokens
        if self.use_semantic_ids:
            self._extend_tokenizer_vocabulary()
    
    def _extend_tokenizer_vocabulary(self):
        """Extend tokenizer vocabulary with semantic ID tokens."""
        if not self.semantic_mapping:
            return
        
        # Generate actual semantic tokens from the mapping
        semantic_tokens = set()
        for item_id, semantic_codes in self.semantic_mapping.items():
            semantic_token = self.semantic_token_prefix + ''.join(f"{code:03d}" for code in semantic_codes)
            semantic_tokens.add(semantic_token)
    
        # Add semantic tokens to tokenizer
        if hasattr(self.tokenizer, 'add_tokens'):
            added_count = self.tokenizer.add_tokens(list(semantic_tokens))
            self.logger.info(f"Extended tokenizer vocabulary with {added_count} semantic tokens")
    
        # Update token count
        self.n_tokens = len(self.tokenizer) if hasattr(self.tokenizer, '__len__') else self.n_tokens
    
    def _format_path(self, path):
        """Format path with semantic item tokens."""
        if not self.use_semantic_ids or self.semantic_mapping is None:
            return super()._format_path(path)
        
        # Convert path to string format first
        path_str = super()._format_path(path)
        
        # Replace item tokens with semantic sequences
        path_str = self._replace_item_tokens_with_semantics(path_str)
        
        return path_str
    
    def _replace_item_tokens_with_semantics(self, path_str: str) -> str:
        """Replace item tokens in path string with semantic ID sequences."""
        if not self.semantic_mapping:
            return path_str
        
        #print(path_str)
        
        # Pattern to match item tokens (adjust based on your token format)
        item_token_prefix = PathLanguageModelingTokenType.ITEM.token
        item_pattern = rf'{re.escape(item_token_prefix)}(\d+)'
        
        def replace_item_token(match):
            item_id = int(match.group(1))
            if item_id in self.semantic_mapping:
                semantic_codes = self.semantic_mapping[item_id]
                # Create a single compact semantic token: S073051122
                semantic_token = " ".join(f"SEM{code}" for code in semantic_codes)
                return semantic_token
            else:
                # Keep original token if no semantic mapping
                return match.group(0)
        
        # Replace all item tokens with semantic sequences
        self.modified_path = re.sub(item_pattern, replace_item_token, path_str)
        #print(modified_path)
        return self.modified_path 
        return path_str
    
    def build(self):
        """Build dataset with semantic integration."""
        self.logger.info(set_color("Building KIGER dataset with semantic item representations", "green"))
        
        # Call parent build method
        datasets = super().build()
        
        
        # Log semantic integration status
        if self.use_semantic_ids and self.semantic_mapping:
            self.logger.info(f"KIGER dataset built with semantic IDs for {len(self.semantic_mapping)} items")
            # Debug: Check item ID range in dataset vs semantic mapping
            if hasattr(self, 'item_num'):
                self.logger.info(f"Dataset has {self.item_num} items, semantic mapping has {len(self.semantic_mapping)} items")
        else:
            self.logger.warning("KIGER dataset built without semantic IDs")
        
        return datasets


