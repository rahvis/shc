"""
SHC Block - Complete Sparse Selective Hyper-Connections Block

Implements Algorithm 1 from the paper: the full SHC residual block with
- Adaptive rank selection (n_eff)
- Stream expansion
- Sparse orthogonal routing (H^pre, H^res, H^post)
- Attention + FFN
- Factorized KV cache compression

Reference: Algorithm 1, Section 4 of the SHC paper
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from shc.layers.cayley import CayleyTransform, BatchedCayleyTransform
from shc.layers.sparse_mixture import SparseOrthogonalMixture, TripleRoutingMatrices
from shc.layers.factorized_cache import FactorizedKVCache
from shc.layers.adaptive_rank import AdaptiveRankSelector, StreamExpander
from shc.blocks.attention import MultiHeadAttention
from shc.blocks.feedforward import FeedForward


@dataclass
class SHCBlockConfig:
    """Configuration for SHC Block."""
    
    # Dimensions
    hidden_dim: int = 768
    ffn_dim: int = 3072
    n_heads: int = 12
    
    # SHC-specific
    n_streams: int = 4
    k_mixture: int = 2
    factorization_rank: int = 1
    
    # Adaptive rank
    use_adaptive_rank: bool = True
    adaptive_temperature: float = 1.0
    
    # Options
    dropout: float = 0.1
    use_flash_attention: bool = True
    use_rope: bool = True
    pre_norm: bool = True  # Pre-LayerNorm (LLaMA style)
    
    # KV cache
    use_factorized_cache: bool = True


class SHCBlock(nn.Module):
    """
    Complete SHC Block implementing Algorithm 1.
    
    The forward pass:
        1. Adaptive rank selection: n_eff = AdaptiveRank(x, l)
        2. Stream expansion: x̄ = StreamExpand(x, n_eff)
        3. Compute routing: H^pre, H^res, H^post = SparseOrthogonal(x̄)
        4. Attention: x_attn = Attention(H^pre @ x̄)
        5. FFN: x_ffn = FFN(x_attn)
        6. Residual: x̄_out = H^res @ x̄ + H^post @ x_ffn
        7. Compress: x_out = Compress(x̄_out) for KV cache
    
    Args:
        config: SHCBlockConfig containing all hyperparameters
        layer_idx: Layer index (for layer-wise adaptation)
        
    Example:
        >>> config = SHCBlockConfig(hidden_dim=768, n_heads=12)
        >>> block = SHCBlock(config, layer_idx=0)
        >>> x = torch.randn(32, 128, 768)
        >>> output, cache = block(x)
    """
    
    def __init__(self, config: SHCBlockConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer norms (pre-norm architecture)
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        
        # Stream norm (for multi-stream hidden state)
        self.ln_stream = nn.LayerNorm(config.hidden_dim)
        
        # Attention
        self.attention = MultiHeadAttention(
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_flash=config.use_flash_attention,
            use_rope=config.use_rope,
        )
        
        # Feed-forward
        self.ffn = FeedForward(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
        
        # SHC Components
        # Adaptive rank selector
        if config.use_adaptive_rank:
            self.rank_selector = AdaptiveRankSelector(
                hidden_dim=config.hidden_dim,
                max_n=config.n_streams,
                temperature=config.adaptive_temperature,
            )
        else:
            self.rank_selector = None
            
        # Stream expander/compressor
        self.stream_expander = StreamExpander(
            hidden_dim=config.hidden_dim,
            max_n=config.n_streams,
        )
        
        # Triple routing matrices (H^pre, H^res, H^post)
        self.routing = TripleRoutingMatrices(
            n=config.n_streams,
            k=config.k_mixture,
            hidden_dim=config.hidden_dim,
        )
        
        # Factorized KV cache
        if config.use_factorized_cache:
            self.kv_compressor = FactorizedKVCache(
                n=config.n_streams,
                d=config.hidden_dim,
                r=config.factorization_rank,
            )
        else:
            self.kv_compressor = None
            
        # Stream-to-single projector (for merging streams back)
        self.stream_merge = nn.Linear(
            config.n_streams * config.hidden_dim, 
            config.hidden_dim
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def _get_effective_rank(self, x: Tensor) -> int:
        """Get effective stream count for this input."""
        if self.rank_selector is not None:
            n_eff = self.rank_selector(x)
            return max(1, int(n_eff.mean().item()))
        return self.config.n_streams
    
    def _expand_to_streams(self, x: Tensor, n_eff: int) -> Tensor:
        """
        Expand input to multi-stream representation.
        
        Args:
            x: Input of shape (batch, seq, hidden)
            n_eff: Effective number of streams
            
        Returns:
            x_bar: Multi-stream of shape (batch, seq, n_eff, hidden)
        """
        return self.stream_expander(x, n_eff)
    
    def _merge_streams(self, x_bar: Tensor) -> Tensor:
        """
        Merge multi-stream representation back to single stream.
        
        Args:
            x_bar: Multi-stream of shape (batch, seq, n, hidden)
            
        Returns:
            x: Single stream of shape (batch, seq, hidden)
        """
        batch, seq, n, hidden = x_bar.shape
        
        # Pad to max streams if needed
        if n < self.config.n_streams:
            padding = torch.zeros(
                batch, seq, self.config.n_streams - n, hidden,
                device=x_bar.device, dtype=x_bar.dtype
            )
            x_bar = torch.cat([x_bar, padding], dim=2)
        
        # Flatten and project
        x_flat = x_bar.view(batch, seq, -1)  # (batch, seq, n * hidden)
        return self.stream_merge(x_flat)  # (batch, seq, hidden)
    
    def _apply_routing(
        self, 
        x_bar: Tensor, 
        H: Tensor,
    ) -> Tensor:
        """
        Apply routing matrix to multi-stream hidden state.
        
        Args:
            x_bar: (batch, seq, n, hidden)
            H: (batch, n, n) routing matrices
            
        Returns:
            Routed x_bar of same shape
        """
        batch, seq, n, hidden = x_bar.shape
        
        # Reshape for batched matmul
        # x_bar: (batch, seq, n, hidden) → (batch * seq, n, hidden)
        x_flat = x_bar.view(batch * seq, n, hidden)
        
        # Expand H for sequence: (batch, n, n) → (batch * seq, n, n)
        H_expanded = H.unsqueeze(1).expand(-1, seq, -1, -1).reshape(batch * seq, n, n)
        
        # Apply routing: (batch * seq, n, n) @ (batch * seq, n, hidden)
        routed = torch.bmm(H_expanded, x_flat)
        
        # Reshape back
        return routed.view(batch, seq, n, hidden)
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass implementing Algorithm 1.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            kv_cache: Optional KV cache for generation
            use_cache: Whether to return updated cache
            
        Returns:
            output: Output tensor of shape (batch, seq_len, hidden_dim)
            new_cache: Updated cache (if use_cache=True)
        """
        batch, seq_len, hidden = x.shape
        residual = x
        
        # Step 1: Adaptive rank selection
        n_eff = self._get_effective_rank(x)
        
        # Step 2: Stream expansion
        if n_eff > 1:
            # Normalize before expansion
            x_normed = self.ln_1(x)
            
            # Expand to multi-stream: (batch, seq, hidden) → (batch, seq, n, hidden)
            x_bar = self._expand_to_streams(x_normed, n_eff)
            
            # Normalize streams
            x_bar = self.ln_stream(x_bar)
            
            # Step 3: Compute routing matrices
            # Use mean across streams for routing computation
            x_for_routing = x_bar.mean(dim=2)  # (batch, seq, hidden)
            H_pre, H_res, H_post = self.routing(x_for_routing)
            
            # Step 4: Apply pre-routing for attention
            x_routed = self._apply_routing(x_bar, H_pre)
            
            # Merge for attention (attention expects single stream)
            x_for_attn = self._merge_streams(x_routed)
            
            # Attention
            attn_out, new_attn_cache = self.attention(
                x_for_attn,
                kv_cache=kv_cache.get('attn') if kv_cache else None,
                attention_mask=attention_mask,
            )
            
            # Apply dropout
            attn_out = self.dropout(attn_out)
            
            # Step 5: FFN
            x_ffn_in = self.ln_2(attn_out)
            ffn_out = self.ffn(x_ffn_in)
            ffn_out = self.dropout(ffn_out)
            
            # Expand FFN output back to streams
            ffn_bar = self._expand_to_streams(ffn_out, n_eff)
            
            # Step 6: Residual with routing
            # x̄_out = H^res @ x̄ + H^post @ ffn_bar
            x_bar_residual = self._apply_routing(x_bar, H_res)
            x_bar_ffn = self._apply_routing(ffn_bar, H_post)
            x_bar_out = x_bar_residual + x_bar_ffn
            
            # Merge back to single stream
            output = self._merge_streams(x_bar_out) + residual
            
            # Step 7: Compress for KV cache
            if use_cache and self.kv_compressor is not None:
                # Compress multi-stream state
                compressed = self.kv_compressor.compress(
                    x_bar_out.view(batch * seq_len, n_eff, hidden)
                )
                compressed = compressed.view(batch, seq_len, -1)
            else:
                compressed = None
                
        else:
            # Bypass: standard transformer block (no multi-stream)
            x_normed = self.ln_1(x)
            
            attn_out, new_attn_cache = self.attention(
                x_normed,
                kv_cache=kv_cache.get('attn') if kv_cache else None,
                attention_mask=attention_mask,
            )
            attn_out = self.dropout(attn_out)
            x = residual + attn_out
            
            x_normed = self.ln_2(x)
            ffn_out = self.ffn(x_normed)
            ffn_out = self.dropout(ffn_out)
            output = x + ffn_out
            
            compressed = None
            new_attn_cache = None
        
        # Build new cache
        new_cache = None
        if use_cache:
            new_cache = {
                'attn': new_attn_cache,
                'compressed': compressed,
                'n_eff': n_eff,
            }
        
        return output, new_cache
    
    def get_routing_stats(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Get routing statistics for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Dict with spectral norms, mixing weights, etc.
        """
        n_eff = self._get_effective_rank(x)
        x_bar = self._expand_to_streams(x, n_eff)
        x_for_routing = x_bar.mean(dim=2)
        
        return {
            'effective_rank': torch.tensor(n_eff),
            **self.routing.get_combined_spectral_norm(x_for_routing),
        }
    
    def extra_repr(self) -> str:
        return (
            f'hidden_dim={self.config.hidden_dim}, '
            f'n_streams={self.config.n_streams}, '
            f'k_mixture={self.config.k_mixture}, '
            f'layer_idx={self.layer_idx}'
        )


class SHCBlockWithGradientCheckpoint(SHCBlock):
    """
    SHC Block with gradient checkpointing for memory efficiency.
    
    Trades compute for memory by recomputing activations during backward pass.
    Essential for training large models on limited GPU memory.
    """
    
    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """Forward with gradient checkpointing during training."""
        if self.training and not use_cache:
            # Use checkpointing during training
            from torch.utils.checkpoint import checkpoint
            
            def forward_fn(x, attention_mask):
                return super(SHCBlockWithGradientCheckpoint, self).forward(
                    x, attention_mask, None, False
                )
            
            output, _ = checkpoint(
                forward_fn, 
                x, 
                attention_mask,
                use_reentrant=False,
            )
            return output, None
        else:
            # Standard forward during inference
            return super().forward(x, attention_mask, kv_cache, use_cache)
