"""
Tests for Cayley Transform

Verifies:
- Orthogonality of generated matrices (Q^T Q = I)
- Spectral norm equals 1
- Batched operations
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn

from shc.layers.cayley import CayleyTransform, BatchedCayleyTransform


class TestCayleyTransform:
    """Tests for CayleyTransform layer."""
    
    @pytest.fixture
    def cayley(self) -> CayleyTransform:
        """Create CayleyTransform instance."""
        return CayleyTransform(n=4, init_scale=0.01)
    
    def test_output_shape(self, cayley: CayleyTransform):
        """Test output has correct shape."""
        Q = cayley()
        assert Q.shape == (4, 4)
    
    def test_orthogonality(self, cayley: CayleyTransform):
        """Test Q^T @ Q ≈ I (orthogonal matrix property)."""
        Q = cayley()
        I = torch.eye(4)
        QtQ = Q.T @ Q
        assert torch.allclose(QtQ, I, atol=1e-5), f"Q^T @ Q differs from I: {QtQ}"
    
    def test_orthogonality_reverse(self, cayley: CayleyTransform):
        """Test Q @ Q^T ≈ I."""
        Q = cayley()
        I = torch.eye(4)
        QQt = Q @ Q.T
        assert torch.allclose(QQt, I, atol=1e-5)
    
    def test_spectral_norm(self, cayley: CayleyTransform):
        """Test spectral norm equals 1 for orthogonal matrices."""
        norm = cayley.get_spectral_norm()
        assert torch.allclose(norm, torch.tensor(1.0), atol=1e-4)
    
    def test_determinant(self, cayley: CayleyTransform):
        """Test |det(Q)| = 1 for orthogonal matrices."""
        Q = cayley()
        det = torch.linalg.det(Q).abs()
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-4)
    
    def test_verify_orthogonality_method(self, cayley: CayleyTransform):
        """Test the verify_orthogonality helper method."""
        assert cayley.verify_orthogonality(atol=1e-5)
    
    def test_gradient_flow(self, cayley: CayleyTransform):
        """Test gradients flow through Cayley transform."""
        Q = cayley()
        loss = Q.sum()
        loss.backward()
        
        assert cayley.params.grad is not None
        assert not torch.isnan(cayley.params.grad).any()
    
    def test_different_sizes(self):
        """Test with different matrix sizes."""
        for n in [2, 4, 8, 16]:
            cayley = CayleyTransform(n=n)
            Q = cayley()
            assert Q.shape == (n, n)
            assert cayley.verify_orthogonality()
    
    def test_device_transfer(self, device: torch.device):
        """Test layer works after device transfer."""
        cayley = CayleyTransform(n=4).to(device)
        Q = cayley()
        assert Q.device == device
        assert cayley.verify_orthogonality()


class TestBatchedCayleyTransform:
    """Tests for BatchedCayleyTransform layer."""
    
    @pytest.fixture
    def batched_cayley(self) -> BatchedCayleyTransform:
        """Create BatchedCayleyTransform instance."""
        return BatchedCayleyTransform(n=4, k=3, init_scale=0.01)
    
    def test_output_shape(self, batched_cayley: BatchedCayleyTransform):
        """Test output has correct shape (k, n, n)."""
        Q = batched_cayley()
        assert Q.shape == (3, 4, 4)
    
    def test_all_orthogonal(self, batched_cayley: BatchedCayleyTransform):
        """Test all k matrices are orthogonal."""
        Q = batched_cayley()  # (k, n, n)
        I = torch.eye(4)
        
        for i in range(3):
            QtQ = Q[i].T @ Q[i]
            assert torch.allclose(QtQ, I, atol=1e-5), f"Matrix {i} not orthogonal"
    
    def test_spectral_norms(self, batched_cayley: BatchedCayleyTransform):
        """Test all spectral norms are 1."""
        norms = batched_cayley.get_spectral_norms()
        assert norms.shape == (3,)
        expected = torch.ones(3)
        assert torch.allclose(norms, expected, atol=1e-4)
    
    def test_matrices_distinct(self, batched_cayley: BatchedCayleyTransform):
        """Test that k matrices are distinct (not identical)."""
        Q = batched_cayley()
        # Check Q[0] != Q[1]
        diff = (Q[0] - Q[1]).abs().sum()
        assert diff > 0.01, "Matrices should be distinct"
    
    def test_gradient_flow(self, batched_cayley: BatchedCayleyTransform):
        """Test gradients flow through all k matrices."""
        Q = batched_cayley()
        loss = Q.sum()
        loss.backward()
        
        assert batched_cayley.params.grad is not None
        assert batched_cayley.params.grad.shape == (3, 6)  # k matrices, n(n-1)/2 params each
