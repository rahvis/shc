"""
SHC Transformer

Complete transformer model with SHC residual connections.

Features:
- Configurable scales (500M, 3B, 7B)
- SHC blocks with sparse orthogonal routing
- torch.compile compatibility
- Generation support with KV cache
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shc.blocks.shc_block import SHCBlock, SHCBlockConfig, SHCBlockWithGradientCheckpoint
from shc.models.embeddings import TokenEmbedding, RMSNorm


@dataclass
class SHCTransformerConfig:
    """Configuration for SHC Transformer model."""
    
    # Architecture
    vocab_size: int = 32000
    hidden_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_dim: Optional[int] = None  # Defaults to 8/3 * hidden_dim
    max_seq_len: int = 4096
    
    # SHC-specific
    n_streams: int = 4
    k_mixture: int = 2
    factorization_rank: int = 1
    use_adaptive_rank: bool = True
    
    # Options
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_flash_attention: bool = True
    use_rope: bool = True
    tie_embeddings: bool = True
    use_gradient_checkpointing: bool = False
    
    # torch.compile
    compile_model: bool = False
    compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    
    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = int(8 / 3 * self.hidden_dim)
    
    def to_block_config(self) -> SHCBlockConfig:
        """Convert to SHCBlockConfig."""
        return SHCBlockConfig(
            hidden_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            n_heads=self.n_heads,
            n_streams=self.n_streams,
            k_mixture=self.k_mixture,
            factorization_rank=self.factorization_rank,
            use_adaptive_rank=self.use_adaptive_rank,
            dropout=self.dropout,
            use_flash_attention=self.use_flash_attention,
            use_rope=self.use_rope,
        )


# Predefined configurations
CONFIGS = {
    '500m': SHCTransformerConfig(
        vocab_size=32000,
        hidden_dim=1024,
        n_layers=24,
        n_heads=16,
        max_seq_len=4096,
    ),
    '1b': SHCTransformerConfig(
        vocab_size=32000,
        hidden_dim=2048,
        n_layers=24,
        n_heads=16,
        max_seq_len=4096,
    ),
    '3b': SHCTransformerConfig(
        vocab_size=32000,
        hidden_dim=2560,
        n_layers=32,
        n_heads=32,
        max_seq_len=4096,
    ),
    '7b': SHCTransformerConfig(
        vocab_size=32000,
        hidden_dim=4096,
        n_layers=32,
        n_heads=32,
        max_seq_len=4096,
        use_gradient_checkpointing=True,
    ),
}


def get_config(model_size: str) -> SHCTransformerConfig:
    """Get predefined configuration by model size."""
    if model_size not in CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Options: {list(CONFIGS.keys())}")
    return CONFIGS[model_size]


class SHCTransformer(nn.Module):
    """
    Complete SHC Transformer Language Model.
    
    Architecture:
        1. Token embedding (+ optional positional)
        2. N × SHC blocks (attention + FFN with orthogonal routing)
        3. Final LayerNorm
        4. LM head (output projection)
    
    Args:
        config: SHCTransformerConfig with all hyperparameters
        
    Example:
        >>> config = SHCTransformerConfig(hidden_dim=768, n_layers=12)
        >>> model = SHCTransformer(config)
        >>> input_ids = torch.randint(0, 32000, (2, 128))
        >>> logits = model(input_ids)  # (2, 128, 32000)
    """
    
    def __init__(self, config: SHCTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embed = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            padding_idx=0,
            scale=False,  # RoPE models don't scale embeddings
        )
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # SHC blocks
        block_config = config.to_block_config()
        BlockClass = SHCBlockWithGradientCheckpoint if config.use_gradient_checkpointing else SHCBlock
        
        self.layers = nn.ModuleList([
            BlockClass(block_config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # Final normalization (RMSNorm for efficiency)
        self.final_norm = RMSNorm(config.hidden_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie embeddings with LM head
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embed.embedding.weight
        
        # Initialize weights
        self._init_weights()
        
        # Compile if requested
        if config.compile_model:
            self._compile_model()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _compile_model(self):
        """Apply torch.compile for faster execution."""
        if hasattr(torch, 'compile'):
            # Compile the forward method
            self.forward = torch.compile(
                self.forward,
                mode=self.config.compile_mode,
                fullgraph=False,
            )
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Dict[str, Any]]]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            use_cache: Return KV cache for generation
            past_key_values: Cached KV states from previous forward
            
        Returns:
            If labels provided: (loss, logits)
            If use_cache: (logits, new_cache)
            Otherwise: logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.token_embed(input_ids)
        hidden_states = self.embed_dropout(hidden_states)
        
        # Initialize cache list
        new_cache = [] if use_cache else None
        
        # Apply SHC blocks
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            hidden_states, layer_cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                kv_cache=past_kv,
                use_cache=use_cache,
            )
            
            if use_cache and layer_cache is not None:
                new_cache.append(layer_cache)
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits
        
        if use_cache:
            return logits, new_cache
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Prompt token IDs of shape (batch, prompt_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k logits
            top_p: Nucleus sampling threshold
            do_sample: Sample vs greedy decoding
            eos_token_id: Stop generation on this token
            pad_token_id: Padding token ID
            
        Returns:
            Generated token IDs of shape (batch, prompt_len + generated)
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize with prompt
        generated = input_ids
        past_key_values = None
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_new_tokens):
            # Forward pass with caching
            if past_key_values is None:
                # First pass: process entire prompt
                logits, past_key_values = self.forward(
                    generated, use_cache=True
                )
                next_token_logits = logits[:, -1, :]
            else:
                # Subsequent passes: only last token
                logits, past_key_values = self.forward(
                    generated[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = next_token_logits.argmax(dim=-1)
            
            # Update finished sequences
            if eos_token_id is not None:
                unfinished = unfinished & (next_tokens != eos_token_id)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Stop if all sequences finished
            if not unfinished.any():
                break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get number of parameters.
        
        Args:
            non_embedding: Exclude embedding parameters
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embed.embedding.weight.numel()
        return n_params
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Estimate memory usage in GB.
        
        Returns:
            Dict with memory breakdown
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'params_gb': param_size / 1e9,
            'buffers_gb': buffer_size / 1e9,
            'total_gb': (param_size + buffer_size) / 1e9,
        }
    
    def get_routing_stats(self, input_ids: Tensor) -> Dict[str, Any]:
        """
        Get routing statistics across all layers.
        
        Useful for analyzing SHC behavior.
        
        Args:
            input_ids: Sample input
            
        Returns:
            Dict with routing statistics per layer
        """
        self.eval()
        stats = {}
        
        hidden_states = self.token_embed(input_ids)
        
        for i, layer in enumerate(self.layers):
            layer_stats = layer.get_routing_stats(hidden_states)
            stats[f'layer_{i}'] = {
                k: v.item() if isinstance(v, Tensor) else v
                for k, v in layer_stats.items()
            }
            hidden_states, _ = layer(hidden_states)
        
        return stats
    
    @classmethod
    def from_pretrained(cls, path: str) -> 'SHCTransformer':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint directory or file
            
        Returns:
            Loaded model
        """
        from pathlib import Path
        
        path = Path(path)
        if path.is_dir():
            checkpoint_path = path / 'checkpoint.pt'
            config_path = path / 'config.json'
        else:
            checkpoint_path = path
            config_path = path.parent / 'config.json'
        
        # Load config
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = SHCTransformerConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def save_pretrained(self, path: str):
        """
        Save model to checkpoint.
        
        Args:
            path: Path to save directory
        """
        from pathlib import Path
        import json
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith('_')
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), path / 'model.pt')
    
    def extra_repr(self) -> str:
        return (
            f'vocab_size={self.config.vocab_size}, '
            f'hidden_dim={self.config.hidden_dim}, '
            f'n_layers={self.config.n_layers}, '
            f'n_heads={self.config.n_heads}, '
            f'params={self.get_num_params() / 1e6:.1f}M'
        )
