"""
RQ-VAE Semantic ID Generator for KIGER Integration
Generates semantic IDs for items using pretrained RQ-VAE models.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from hopwise.utils import set_color, getLogger


class SemanticIDGenerator:
    """Generator for creating semantic ID mappings from RQ-VAE models."""
    
    def __init__(self, rqvae_model_path: str, dataset_name: str, device: str = "cpu"):
        self.logger = getLogger()
        self.device = device
        self.dataset_name = dataset_name
        self.rqvae_model = None
        
        # Load RQ-VAE model
        if rqvae_model_path:
            self._load_rqvae_model(rqvae_model_path)
    
    def _load_rqvae_model(self, model_path: str):
        """Load pretrained RQ-VAE model."""
        try:
            from hopwise.external.rqvae.modules.rq_vae import RQ_VAE
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            if 'model_config' in checkpoint:
                model_config = checkpoint['model_config']
            else:
                # Default configuration for MovieLens
                model_config = {
                    'input_dim': 768,
                    'latent_dim': 256,
                    'hidden_dims': [512, 256],
                    'codebook_size': 256,
                    'n_quantization_layers': 3,
                    'commitment_weight': 0.25,
                }
            
            # Initialize and load model
            self.rqvae_model = RQ_VAE(**model_config)
            
            if 'model_state_dict' in checkpoint:
                self.rqvae_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.rqvae_model.load_state_dict(checkpoint)
            
            self.rqvae_model.eval()
            self.rqvae_model.to(self.device)
            
            self.logger.info(set_color(f"Loaded RQ-VAE model from {model_path}", "green"))
            
        except Exception as e:
            self.logger.error(f"Failed to load RQ-VAE model: {e}")
            self.rqvae_model = None
    
    def _load_item_embeddings(self) -> torch.Tensor:
        """Load item embeddings for the dataset."""
        try:
            from hopwise.external.rqvae.data.loader import load_movie_lens, load_amazon
            
            if "ml-" in self.dataset_name.lower():
                # MovieLens dataset
                category = "100k" if "100k" in self.dataset_name else "1m"
                data = load_movie_lens(category=category, dimension="item", train=True, raw=False)
            elif "amazon" in self.dataset_name.lower():
                # Amazon dataset
                data = load_amazon(dimension="item", train=True, raw=False)
            else:
                raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
            self.logger.info(f"Loaded item embeddings for {data.shape[0]} items")
            return torch.tensor(data, dtype=torch.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to load item embeddings: {e}")
            return None
    
    def generate_item_semantic_mapping(self) -> Dict[int, List[int]]:
        """Generate mapping from item_id to semantic_codes."""
        if self.rqvae_model is None:
            self.logger.error("RQ-VAE model not loaded")
            return {}
        
        # Load item embeddings
        item_embeddings = self._load_item_embeddings()
        if item_embeddings is None:
            return {}
        
        semantic_mapping = {}
        
        self.logger.info("Generating semantic IDs for items...")
        
        with torch.no_grad():
            for item_id in range(item_embeddings.shape[0]):
                try:
                    item_embedding = item_embeddings[item_id].to(self.device)
                    semantic_ids = self.rqvae_model.get_semantic_id_single(item_embedding)
                    semantic_mapping[item_id] = semantic_ids.cpu().tolist()
                    
                    if (item_id + 1) % 100 == 0:
                        self.logger.info(f"Processed {item_id + 1}/{item_embeddings.shape[0]} items")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to generate semantic ID for item {item_id}: {e}")
                    continue
        
        self.logger.info(f"Generated semantic mappings for {len(semantic_mapping)} items")
        return semantic_mapping
    
    def save_mapping(self, mapping: Dict[int, List[int]], output_path: str):
        """Save semantic ID mapping to file."""
        try:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(set_color(f"Saved semantic mapping to {output_path}", "green"))
            
        except Exception as e:
            self.logger.error(f"Failed to save semantic mapping: {e}")
    
    def load_mapping(self, mapping_path: str) -> Optional[Dict[int, List[int]]]:
        """Load semantic ID mapping from file."""
        try:
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f)
            
            self.logger.info(f"Loaded semantic mapping with {len(mapping)} items rqvae")
            return mapping
            
        except Exception as e:
            self.logger.error(f"Failed to load semantic mapping: {e}")
            return None
