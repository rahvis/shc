"""
Tests for Sparse Orthogonal Mixture

Verifies:
- Spectral norm bounded by 1 (Proposition 1)
- Mixing weights sum to 1
- Routing matrix properties
"""

import pytest
import torch
import torch.nn as nn

from shc.layers.sparse_mixture import SparseOrthogonalMixture, TripleRoutingMatrices


class TestSparseOrthogonalMixture:
    """Tests for SparseOrthogonalMixture layer."""
    
    @pytest.fixture
    def mixture(self) -> SparseOrthogonalMixture:
        """Create SparseOrthogonalMixture instance."""
        return SparseOrthogonalMixture(n=4, k=2, hidden_dim=64)
    
    def test_output_shape(self, mixture: SparseOrthogonalMixture, batch_size: int):
        """Test H^res has correct shape (batch, n, n)."""
        x = torch.randn(batch_size, 64)
        H_res = mixture(x)
        assert H_res.shape == (batch_size, 4, 4)
    
    def test_output_shape_with_seq(self, mixture: SparseOrthogonalMixture, batch_size: int, seq_len: int):
        """Test with sequence input (batch, seq, hidden_dim)."""
        x = torch.randn(batch_size, seq_len, 64)
        H_res = mixture(x)
        assert H_res.shape == (batch_size, 4, 4)
    
    def test_spectral_norm_bounded(self, mixture: SparseOrthogonalMixture, batch_size: int):
        """Test ρ(H^res) ≤ 1 (Proposition 1 from paper)."""
        x = torch.randn(batch_size, 64)
        norms = mixture.get_spectral_norm(x)
        assert norms.shape == (batch_size,)
        assert (norms <= 1.0 + 1e-4).all(), f"Spectral norms exceed 1: {norms}"
    
    def test_mixing_weights_sum_to_one(self, mixture: SparseOrthogonalMixture, batch_size: int):
        """Test α_i(x) sum to 1."""
        x = torch.randn(batch_size, 64)
        alpha = mixture.compute_mixing_weights(x)
        assert alpha.shape == (batch_size, 2)  # k=2
        sums = alpha.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    def test_mixing_weights_non_negative(self, mixture: SparseOrthogonalMixture, batch_size: int):
        """Test α_i(x) ≥ 0."""
        x = torch.randn(batch_size, 64)
        alpha = mixture.compute_mixing_weights(x)
        assert (alpha >= 0).all()
    
    def test_temperature_effect(self, mixture: SparseOrthogonalMixture):
        """Test temperature affects mixing weight distribution."""
        x = torch.randn(1, 64)
        
        # Low temperature = more peaked
        alpha_low = mixture.compute_mixing_weights(x, temperature=0.1)
        # High temperature = more uniform
        alpha_high = mixture.compute_mixing_weights(x, temperature=10.0)
        
        # Low temp should have higher max value (more peaked)
        assert alpha_low.max() > alpha_high.max()
    
    def test_return_components(self, mixture: SparseOrthogonalMixture, batch_size: int):
        """Test return_components flag."""
        x = torch.randn(batch_size, 64)
        H_res, Q, alpha = mixture(x, return_components=True)
        
        assert H_res.shape == (batch_size, 4, 4)
        assert Q.shape == (2, 4, 4)  # k orthogonal matrices
        assert alpha.shape == (batch_size, 2)
    
    def test_apply_routing(self, mixture: SparseOrthogonalMixture, batch_size: int):
        """Test routing matrix application."""
        x = torch.randn(batch_size, 64)
        x_bar = torch.randn(batch_size, 4, 64)  # Multi-stream input
        
        H_res = mixture(x)
        routed = mixture.apply_routing(x_bar, H_res)
        
        assert routed.shape == (batch_size, 4, 64)
    
    def test_gradient_flow(self, mixture: SparseOrthogonalMixture, batch_size: int):
        """Test gradients flow through the mixture."""
        x = torch.randn(batch_size, 64, requires_grad=True)
        H_res = mixture(x)
        loss = H_res.sum()
        loss.backward()
        
        assert x.grad is not None


class TestTripleRoutingMatrices:
    """Tests for TripleRoutingMatrices (H^pre, H^res, H^post)."""
    
    @pytest.fixture
    def triple(self) -> TripleRoutingMatrices:
        """Create TripleRoutingMatrices instance."""
        return TripleRoutingMatrices(n=4, k=2, hidden_dim=64)
    
    def test_output_shapes(self, triple: TripleRoutingMatrices, batch_size: int):
        """Test all three matrices have correct shapes."""
        x = torch.randn(batch_size, 64)
        H_pre, H_res, H_post = triple(x)
        
        assert H_pre.shape == (batch_size, 4, 4)
        assert H_res.shape == (batch_size, 4, 4)
        assert H_post.shape == (batch_size, 4, 4)
    
    def test_all_spectral_norms_bounded(self, triple: TripleRoutingMatrices, batch_size: int):
        """Test all three matrices have ρ ≤ 1."""
        x = torch.randn(batch_size, 64)
        norms = triple.get_combined_spectral_norm(x)
        
        assert 'pre' in norms
        assert 'res' in norms
        assert 'post' in norms
        
        for key, norm in norms.items():
            assert (norm <= 1.0 + 1e-4).all(), f"{key} spectral norm exceeds 1"
