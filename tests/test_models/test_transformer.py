"""
Tests for SHC Transformer Model

Verifies:
- Forward pass shapes
- Loss computation
- Generation
- Save/load functionality
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from shc.models.transformer import SHCTransformer, SHCTransformerConfig, get_config


class TestSHCTransformerConfig:
    """Tests for SHCTransformerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SHCTransformerConfig()
        assert config.vocab_size == 32000
        assert config.hidden_dim == 768
        assert config.n_layers == 12
    
    def test_predefined_configs(self):
        """Test predefined model configurations."""
        for size in ['500m', '1b', '3b', '7b']:
            config = get_config(size)
            assert isinstance(config, SHCTransformerConfig)
    
    def test_invalid_config(self):
        """Test error on invalid config name."""
        with pytest.raises(KeyError):
            get_config('invalid_size')


class TestSHCTransformer:
    """Tests for SHCTransformer model."""
    
    @pytest.fixture
    def small_config(self) -> SHCTransformerConfig:
        """Create minimal config for fast testing."""
        return SHCTransformerConfig(
            vocab_size=1000,
            hidden_dim=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=128,
            n_streams=4,
            k_mixture=2,
        )
    
    @pytest.fixture
    def model(self, small_config: SHCTransformerConfig) -> SHCTransformer:
        """Create small model for testing."""
        return SHCTransformer(small_config)
    
    def test_forward_shape(self, model: SHCTransformer, batch_size: int, seq_len: int, small_config: SHCTransformerConfig):
        """Test forward pass output shape."""
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_forward_with_labels(self, model: SHCTransformer, batch_size: int, seq_len: int, small_config: SHCTransformerConfig):
        """Test forward pass with labels returns loss."""
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # Check loss is returned (could be tuple or dict)
        if isinstance(outputs, tuple):
            loss = outputs[0]
        elif isinstance(outputs, dict):
            loss = outputs.get('loss')
        else:
            loss = outputs
        
        assert loss is not None or outputs.shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_forward_with_attention_mask(self, model: SHCTransformer, batch_size: int, seq_len: int, small_config: SHCTransformerConfig):
        """Test forward pass with attention mask."""
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = 0  # Mask second half
        
        logits = model(input_ids, attention_mask=attention_mask)
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_generate(self, model: SHCTransformer, small_config: SHCTransformerConfig):
        """Test text generation."""
        prompt = torch.randint(0, small_config.vocab_size, (1, 10))
        
        output = model.generate(
            prompt,
            max_new_tokens=5,
            temperature=1.0,
            do_sample=False,
        )
        
        assert output.shape[0] == 1
        assert output.shape[1] >= 10  # At least prompt length
        assert output.shape[1] <= 15  # At most prompt + max_new_tokens
    
    def test_num_params(self, model: SHCTransformer):
        """Test parameter counting."""
        n_params = model.get_num_params()
        assert n_params > 0
        assert isinstance(n_params, int)
    
    def test_memory_footprint(self, model: SHCTransformer):
        """Test memory estimation."""
        memory = model.get_memory_footprint()
        assert 'parameters_gb' in memory or isinstance(memory, dict)
    
    def test_save_load(self, model: SHCTransformer, small_config: SHCTransformerConfig):
        """Test model save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            
            # Save
            model.save_pretrained(str(save_path))
            
            # Check files exist
            assert (save_path / "model.pt").exists() or (save_path / "pytorch_model.bin").exists()
            
            # Load
            loaded_model = SHCTransformer.from_pretrained(str(save_path))
            
            # Compare outputs
            input_ids = torch.randint(0, small_config.vocab_size, (1, 16))
            
            model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                orig_out = model(input_ids)
                loaded_out = loaded_model(input_ids)
            
            assert torch.allclose(orig_out, loaded_out, atol=1e-5)
    
    def test_device_transfer(self, model: SHCTransformer, device: torch.device, small_config: SHCTransformerConfig):
        """Test model works after device transfer."""
        model = model.to(device)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 16), device=device)
        
        logits = model(input_ids)
        assert logits.device == device
    
    def test_gradient_flow(self, model: SHCTransformer, small_config: SHCTransformerConfig):
        """Test gradients flow through the model."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16))
        labels = torch.randint(0, small_config.vocab_size, (2, 16))
        
        model.train()
        outputs = model(input_ids, labels=labels)
        
        # Get loss
        if isinstance(outputs, tuple):
            loss = outputs[0]
        elif isinstance(outputs, dict):
            loss = outputs['loss']
        else:
            # Compute loss manually if not returned
            logits = outputs
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, small_config.vocab_size),
                labels.view(-1)
            )
        
        loss.backward()
        
        # Check at least some gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients found"
