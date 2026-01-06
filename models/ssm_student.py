"""
SSM Student Model

Mamba-style State Space Model for efficient O(L) inference.
Used to distill SHC Transformer for deployment scenarios
requiring minimal memory (no KV cache).

Reference: Section 3.4 of the SHC paper
"""

from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass 
class SSMConfig:
    """Configuration for SSM student model."""
    
    # Dimensions matching teacher
    vocab_size: int = 32000
    hidden_dim: int = 768
    n_layers: int = 12
    
    # SSM-specific
    state_dim: int = 16  # SSM state dimension
    expansion_factor: int = 2  # Inner dimension expansion
    dt_rank: str = 'auto'  # Rank of dt projection
    
    # Options
    use_selective: bool = True  # Input-dependent A
    bidirectional: bool = False
    dropout: float = 0.1


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model layer.
    
    Implements input-dependent state transitions (selective scan):
        h_t = A(x_t) h_{t-1} + B x_t
        y_t = C h_t
    
    Key innovation: A is input-dependent, allowing selective information flow.
    
    Args:
        d_model: Input dimension
        d_state: SSM state dimension
        expansion_factor: Inner dimension expansion
        dt_rank: Rank of delta projection
        
    Example:
        >>> ssm = SelectiveSSM(d_model=768, d_state=16)
        >>> x = torch.randn(32, 128, 768)  # [batch, seq, dim]
        >>> y = ssm(x)  # [batch, seq, dim]
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expansion_factor: int = 2,
        dt_rank: Optional[int] = None,
        use_selective: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expansion_factor
        self.use_selective = use_selective
        
        # Delta (timestep) projection rank
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        
        # Input projection: d_model -> d_inner * 2 (for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner,
        )
        
        # SSM parameters
        # A: state transition matrix (parameterized as log for stability)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        
        # D: skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Input-dependent projections for B, C, dt
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Initialize
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for stable training."""
        # Initialize A_log for stable dynamics
        with torch.no_grad():
            # HiPPO-style initialization
            A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
            self.A_log.copy_(torch.log(A))
        
        # dt initialization
        nn.init.uniform_(self.dt_proj.weight, -0.1, 0.1)
        
        # Small initialization for projections
        for proj in [self.in_proj, self.x_proj, self.out_proj]:
            nn.init.normal_(proj.weight, std=0.02)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with selective scan.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            
        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Input projection with gating
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_proj, z = xz.chunk(2, dim=-1)  # Each (batch, seq_len, d_inner)
        
        # Convolution for local context
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Compute input-dependent B, C, dt
        x_dbl = self.x_proj(x_conv)  # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Project dt to full dimension
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)  # Ensure positive
        
        # Get A from log parameterization
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Selective scan
        y = self._selective_scan(x_conv, dt, A, B, C)
        
        # Skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        return self.out_proj(y)
    
    def _selective_scan(
        self,
        x: Tensor,  # (batch, seq_len, d_inner)
        dt: Tensor,  # (batch, seq_len, d_inner)
        A: Tensor,  # (d_inner, d_state)
        B: Tensor,  # (batch, seq_len, d_state)
        C: Tensor,  # (batch, seq_len, d_state)
    ) -> Tensor:
        """
        Perform selective scan operation.
        
        This is the core SSM computation with input-dependent discretization.
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A: A_bar = exp(dt * A)
        # For numerical stability, compute element-wise
        # dt: (batch, seq_len, d_inner) -> (batch, seq_len, d_inner, 1)
        # A: (d_inner, d_state) -> (1, 1, d_inner, d_state)
        dt_expanded = dt.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        A_bar = torch.exp(dt_expanded * A_expanded)  # (batch, seq_len, d_inner, d_state)
        
        # Discretize B: B_bar = dt * B
        # B: (batch, seq_len, d_state) -> (batch, seq_len, 1, d_state)
        B_bar = dt_expanded * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)
        
        # Sequential scan (can be parallelized with associative scan)
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for t in range(seq_len):
            # State update: h = A_bar * h + B_bar * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :, None]
            # Output: y = C * h
            y_t = (h * C[:, t, None, :]).sum(dim=-1)  # (batch, d_inner)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
    
    def step(
        self,
        x: Tensor,  # (batch, d_model)
        h: Tensor,  # (batch, d_inner, d_state)
        conv_state: Tensor,  # (batch, d_inner, conv_len)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Single step for autoregressive generation.
        
        O(1) time and memory per step (no KV cache needed).
        
        Args:
            x: Single token input (batch, d_model)
            h: SSM hidden state
            conv_state: Convolution state buffer
            
        Returns:
            output, new_h, new_conv_state
        """
        # Input projection
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Update conv state
        conv_state = torch.cat([conv_state[:, :, 1:], x_proj.unsqueeze(-1)], dim=-1)
        x_conv = (self.conv1d.weight.squeeze(1) * conv_state).sum(dim=-1)
        x_conv = x_conv + self.conv1d.bias
        x_conv = F.silu(x_conv)
        
        # Compute B, C, dt
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        # Single step update
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0)
        A_bar = torch.exp(dt_A)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(1)
        
        h = A_bar * h + B_bar * x_conv.unsqueeze(-1)
        y = (h * C.unsqueeze(1)).sum(dim=-1)
        
        # Skip and gate
        y = y + x_conv * self.D
        y = y * F.silu(z)
        
        return self.out_proj(y), h, conv_state


class SSMBlock(nn.Module):
    """
    Complete SSM block with norm and residual.
    
    Args:
        d_model: Model dimension
        d_state: SSM state dimension
        expansion_factor: Inner dimension expansion
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expansion_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, expansion_factor)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward with pre-norm and residual."""
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return residual + x


class SSMStudent(nn.Module):
    """
    Complete SSM student model for distillation.
    
    Replaces SHC Transformer attention with selective SSM
    for O(L) inference complexity and no KV cache.
    
    Args:
        config: SSMConfig with model hyperparameters
        
    Example:
        >>> config = SSMConfig(hidden_dim=768, n_layers=12)
        >>> student = SSMStudent(config)
        >>> logits = student(input_ids)
    """
    
    def __init__(self, config: SSMConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # SSM blocks
        self.layers = nn.ModuleList([
            SSMBlock(
                d_model=config.hidden_dim,
                d_state=config.state_dim,
                expansion_factor=config.expansion_factor,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed.weight
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embed.weight, std=0.02)
    
    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            labels: Optional labels for loss
            
        Returns:
            logits (and loss if labels provided)
        """
        x = self.embed(input_ids)
        x = self.embed_dropout(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Generate with O(1) per-step complexity.
        
        No KV cache needed - uses constant-size SSM state.
        """
        self.eval()
        batch = input_ids.size(0)
        device = input_ids.device
        
        # Initialize states for each layer
        states = []
        for layer in self.layers:
            h = torch.zeros(batch, layer.ssm.d_inner, layer.ssm.d_state, device=device)
            conv_state = torch.zeros(batch, layer.ssm.d_inner, 4, device=device)
            states.append((h, conv_state))
        
        # Process prompt
        generated = input_ids.clone()
        for t in range(input_ids.size(1)):
            token = input_ids[:, t]
            x = self.embed(token)
            
            for i, layer in enumerate(self.layers):
                h, conv_state = states[i]
                x_norm = layer.norm(x)
                x_ssm, h, conv_state = layer.ssm.step(x_norm, h, conv_state)
                x = x + layer.dropout(x_ssm)
                states[i] = (h, conv_state)
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            x = self.final_norm(x)
            logits = self.lm_head(x)
            
            if temperature != 1.0:
                logits = logits / temperature
            
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Step through layers
            x = self.embed(next_token.squeeze(-1))
            for i, layer in enumerate(self.layers):
                h, conv_state = states[i]
                x_norm = layer.norm(x)
                x_ssm, h, conv_state = layer.ssm.step(x_norm, h, conv_state)
                x = x + layer.dropout(x_ssm)
                states[i] = (h, conv_state)
        
        return generated
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage info.
        
        SSM has constant memory regardless of sequence length.
        """
        batch_size = 1
        d_inner = self.config.hidden_dim * self.config.expansion_factor
        state_size = batch_size * d_inner * self.config.state_dim * self.config.n_layers
        
        return {
            'state_elements': state_size,
            'state_bytes': state_size * 4,  # float32
            'kv_cache_equivalent': 'None - O(1) per step',
        }
    
    @classmethod
    def from_teacher_config(cls, teacher_config) -> 'SSMStudent':
        """Create SSM student matching teacher dimensions."""
        config = SSMConfig(
            vocab_size=teacher_config.vocab_size,
            hidden_dim=teacher_config.hidden_dim,
            n_layers=teacher_config.n_layers,
        )
        return cls(config)
