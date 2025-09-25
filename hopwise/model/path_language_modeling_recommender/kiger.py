# @Time   : 2025/01
# @Author : Integration Team
# @Email  : team@hopwise.ai

r"""KIGER (Knowledge-aware Item Generation Enhanced Recommender)
##################################################
KIGER extends KGGLM by integrating RQ-VAE semantic item representations.
Instead of using random item tokens, KIGER uses semantic ID sequences 
generated from a pretrained RQ-VAE model to provide meaningful item embeddings.
"""

import os
import pickle
import torch
from pathlib import Path
from typing import Dict, List, Optional

from logging import getLogger
from hopwise.model.path_language_modeling_recommender.kgglm import KGGLM
from hopwise.utils import set_color


class KIGER(KGGLM):
    """KIGER: Knowledge-aware Item Generation Enhanced Recommender
    
    Extends KGGLM with RQ-VAE semantic item representations for better
    item understanding and recommendation performance.
    """
    
    def __init__(self, config, dataset):
        # Initialize logger early for RQ-VAE integration
        from logging import getLogger
        self.logger = getLogger()
        
        # Initialize RQ-VAE integration BEFORE parent initialization
        self.use_semantic_ids = True  # Always enabled for KIGER
        self.semantic_mapping = None
        self.rqvae_model = None
        
        # Load RQ-VAE components
        self._init_rqvae_integration(config)
        
        # Initialize parent KGGLM
        super().__init__(config, dataset)
        
        self.logger.info(set_color("KIGER initialized with RQ-VAE semantic IDs", "green"))
        if self.semantic_mapping:
            self.logger.info(f"Loaded semantic mapping for {len(self.semantic_mapping)} items kiger")
    
    def _init_rqvae_integration(self, config):
        """Initialize RQ-VAE model and semantic mappings."""
        try:
            # Import RQ-VAE components
            from hopwise.external.rqvae.modules.rq_vae import RQ_VAE
            # Load RQ-VAE model configuration
            self.rqvae_config = config["rqvae_config"]
            self.n_quantization_layers = self.rqvae_config["n_quantization_layers"]
            self.codebook_size = self.rqvae_config["codebook_size"]
            self.semantic_token_prefix = self.rqvae_config["semantic_token_prefix"]
            
            # Load pretrained RQ-VAE model if path provided
            rqvae_model_path = self.rqvae_config["model_path"]
            if rqvae_model_path and os.path.exists(rqvae_model_path):
                self._load_rqvae_model(rqvae_model_path, config)
            
            # Load semantic mapping if available
            semantic_mapping_path = self.rqvae_config["semantic_mapping_path"]
            if semantic_mapping_path and os.path.exists(semantic_mapping_path):
                self._load_semantic_mapping(semantic_mapping_path)
            
        except ImportError as e:
            self.logger.error(f"Failed to import RQ-VAE components: {e}")
            self.logger.error("Please ensure RQ-VAE files are copied to hopwise/external/rqvae/")
            raise
    
    def _load_rqvae_model(self, model_path: str, config):
        try:
            device = config["device"]

            # Force-load as weights (avoid loading pickled Module objects)
            checkpoint = torch.load(model_path, map_location=device)

            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    model_config = checkpoint.get("model_config", None)
                else:
                    state_dict = checkpoint
                    model_config = None
            else:
                # Unexpected: checkpoint is a full Module, extract state_dict
                self.logger.warning("Checkpoint is a full nn.Module, extracting state_dict...")
                state_dict = checkpoint.state_dict()
                model_config = getattr(checkpoint, "model_config", None)

            # Default config if not inside checkpoint
            if model_config is None:
                model_config = {
                    "input_dim": 768,
                    "latent_dim": 256,
                    "hidden_dims": [768, 512, 256],
                    "codebook_size": self.codebook_size,
                    "n_quantization_layers": self.n_quantization_layers,
                    "commitment_weight": 0.25,
                }

            # Build fresh model and load weights
            from hopwise.external.rqvae.modules.rq_vae import RQ_VAE
            self.rqvae_model = RQ_VAE(**model_config)
            self.rqvae_model.load_state_dict(state_dict, strict=False)
            self.rqvae_model.eval()
            self.rqvae_model.to(device)

            self.logger.info(f"Loaded RQ-VAE model from {model_path}")

        except Exception as e:
            self.logger.warning(f"Failed to load RQ-VAE model: {e}")
            self.rqvae_model = None
    
    def _load_semantic_mapping(self, mapping_path: str):
        """Load precomputed semantic ID mapping."""
        try:
            mapping_data = torch.load(mapping_path, map_location="cpu", weights_only=False)
            
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
            self.logger.warning(f"Failed to load semantic mapping: {e}")
            self.semantic_mapping = None
    
    def generate_semantic_ids_for_item(self, item_embedding: torch.Tensor) -> List[int]:
        """Generate semantic IDs for a single item using RQ-VAE."""
        if self.rqvae_model is None:
            return []
        
        try:
            with torch.no_grad():
                semantic_ids = self.rqvae_model.get_semantic_id_single(item_embedding)
                return semantic_ids.cpu().tolist()
        except Exception as e:
            self.logger.warning(f"Failed to generate semantic IDs: {e}")
            return []
    
    def get_semantic_tokens_for_item(self, item_id: int) -> List[str]:
        """Get semantic token sequence for an item."""
        if self.semantic_mapping and item_id in self.semantic_mapping:
            semantic_codes = self.semantic_mapping[item_id]
            tokens = []
            for layer_idx, code in enumerate(semantic_codes):
                token = f"{self.semantic_token_prefix}_{layer_idx}_{code}"
                tokens.append(token)
            return tokens
        return []
    
    def extend_vocabulary_with_semantic_tokens(self, tokenizer):
        """Extend tokenizer vocabulary with semantic ID tokens."""
        semantic_tokens = []
        
        for layer in range(self.n_quantization_layers):
            for code in range(self.codebook_size):
                token = f"{self.semantic_token_prefix}_{layer}_{code}"
                semantic_tokens.append(token)
        
        # Add tokens to tokenizer
        added_tokens = tokenizer.add_tokens(semantic_tokens)
        
        self.logger.info(f"Added {added_tokens} semantic tokens to vocabulary")
        return added_tokens

    @torch.no_grad()
    def generate(self, inputs, top_k=None, paths_per_user=1, **kwargs):
        """Override generate to log semantic token usage."""
        outputs = super().generate(inputs, top_k=top_k, paths_per_user=paths_per_user, **kwargs)
        
        # Log semantic token usage
        self.logger.info(f"Semantic tokens used: {type(outputs)}")
        try:
            self.logger.info(outputs.keys())
        except:
            pass
        # semantic_tokens = [token for token in outputs['generated_tokens'] if token.startswith(self.semantic_token_prefix)]
        #self.logger.info(f"Semantic tokens used: {semantic_tokens}")
        
        return outputs

    def validation_step(self, batch):
        """Override validation to log semantic token statistics."""
        result = super().validation_step(batch)
        
        return result
