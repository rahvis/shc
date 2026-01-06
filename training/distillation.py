"""
SSM Distillation

Training utilities for distilling SHC Transformer teacher
into SSM student for efficient inference.

Distillation methods:
1. KL divergence on output logits
2. MSE on intermediate hidden states
3. Attention transfer (optional)

Reference: Section 3.4 of the SHC paper
"""

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from shc.models.transformer import SHCTransformer
from shc.models.ssm_student import SSMStudent, SSMConfig
from shc.training.optimizer import create_optimizer, create_scheduler, GradientClipper
from shc.training.distributed import is_main_process, all_reduce_mean


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""
    
    # Loss weights
    kl_weight: float = 1.0
    mse_weight: float = 0.5
    ce_weight: float = 0.1  # Cross-entropy on labels
    
    # Temperature for KL divergence
    temperature: float = 2.0
    
    # Training
    learning_rate: float = 1e-4
    max_steps: int = 10000
    warmup_steps: int = 500
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    # Layer mapping (for hidden state matching)
    layer_mapping: str = 'uniform'  # 'uniform', 'first_last', 'all'


class DistillationLoss(nn.Module):
    """
    Combined distillation loss.
    
    Loss = α * KL(student || teacher) + β * MSE(hiddens) + γ * CE(labels)
    
    Args:
        config: DistillationConfig with loss weights
    """
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.kl_weight = config.kl_weight
        self.mse_weight = config.mse_weight
        self.ce_weight = config.ce_weight
    
    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_hidden: Optional[List[Tensor]] = None,
        teacher_hidden: Optional[List[Tensor]] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Student output logits
            teacher_logits: Teacher output logits (no grad)
            student_hidden: Optional student hidden states
            teacher_hidden: Optional teacher hidden states
            labels: Optional ground truth labels
            
        Returns:
            total_loss, loss_dict with individual components
        """
        losses = {}
        total_loss = 0.0
        
        # KL divergence loss (scaled by T^2 for gradient scaling)
        if self.kl_weight > 0:
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean',
            ) * (self.temperature ** 2)
            
            total_loss = total_loss + self.kl_weight * kl_loss
            losses['kl_loss'] = kl_loss.item()
        
        # Hidden state MSE loss
        if self.mse_weight > 0 and student_hidden is not None and teacher_hidden is not None:
            mse_loss = 0.0
            n_layers = min(len(student_hidden), len(teacher_hidden))
            
            for s_h, t_h in zip(student_hidden, teacher_hidden):
                mse_loss = mse_loss + F.mse_loss(s_h, t_h.detach())
            
            mse_loss = mse_loss / n_layers
            total_loss = total_loss + self.mse_weight * mse_loss
            losses['mse_loss'] = mse_loss.item()
        
        # Cross-entropy on labels (regularization)
        if self.ce_weight > 0 and labels is not None:
            ce_loss = F.cross_entropy(
                student_logits[:, :-1].contiguous().view(-1, student_logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + self.ce_weight * ce_loss
            losses['ce_loss'] = ce_loss.item()
        
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses


class DistillationTrainer:
    """
    Trainer for distilling SHC teacher to SSM student.
    
    Features:
        - KL + MSE + CE combined loss
        - Layer-wise hidden state matching
        - Progressive layer unfreezing
        - Validation on teacher-student agreement
    
    Args:
        teacher: Trained SHC Transformer (frozen)
        student: SSM student to train
        config: DistillationConfig
        train_dataloader: Training data
        eval_dataloader: Optional evaluation data
        
    Example:
        >>> teacher = SHCTransformer.from_pretrained('path/to/teacher')
        >>> student = SSMStudent.from_teacher_config(teacher.config)
        >>> trainer = DistillationTrainer(teacher, student, config, train_loader)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        teacher: SHCTransformer,
        student: SSMStudent,
        config: DistillationConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Loss function
        self.criterion = DistillationLoss(config)
        
        # Optimizer (only for student)
        self.optimizer = create_optimizer(
            self.student,
            learning_rate=config.learning_rate,
        )
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
        )
        
        # Gradient clipper
        self.grad_clipper = GradientClipper(config.max_grad_norm)
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Layer mapping for hidden state matching
        self._setup_layer_mapping()
    
    def _setup_layer_mapping(self):
        """Setup mapping between teacher and student layers."""
        n_teacher = self.teacher.config.n_layers
        n_student = self.student.config.n_layers
        
        if self.config.layer_mapping == 'uniform':
            # Uniform spacing
            step = n_teacher / n_student
            self.layer_indices = [int(i * step) for i in range(n_student)]
        elif self.config.layer_mapping == 'first_last':
            # First and last layers only
            self.layer_indices = [0, n_teacher - 1]
        else:
            # All layers (requires same count)
            self.layer_indices = list(range(min(n_teacher, n_student)))
    
    def _get_teacher_hidden(self, input_ids: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Get teacher logits and hidden states."""
        with torch.no_grad():
            # Forward through teacher, collecting hidden states
            x = self.teacher.token_embed(input_ids)
            hidden_states = []
            
            for i, layer in enumerate(self.teacher.layers):
                x, _ = layer(x)
                if i in self.layer_indices:
                    hidden_states.append(x.clone())
            
            x = self.teacher.final_norm(x)
            logits = self.teacher.lm_head(x)
            
        return logits, hidden_states
    
    def _get_student_hidden(self, input_ids: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Get student logits and hidden states."""
        x = self.student.embed(input_ids)
        hidden_states = []
        
        for i, layer in enumerate(self.student.layers):
            x = layer(x)
            if i < len(self.layer_indices):
                hidden_states.append(x)
        
        x = self.student.final_norm(x)
        logits = self.student.lm_head(x)
        
        return logits, hidden_states
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.student.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Get teacher outputs (no grad)
        teacher_logits, teacher_hidden = self._get_teacher_hidden(input_ids)
        
        # Get student outputs
        student_logits, student_hidden = self._get_student_hidden(input_ids)
        
        # Compute loss
        loss, loss_dict = self.criterion(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            labels=labels,
        )
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        grad_norm = self.grad_clipper.clip(self.student.parameters())
        loss_dict['grad_norm'] = grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        loss_dict['lr'] = self.scheduler.get_last_lr()[0]
        
        return loss_dict
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on eval dataset."""
        if self.eval_dataloader is None:
            return {}
        
        self.student.eval()
        total_loss = 0.0
        total_agreement = 0.0
        n_batches = 0
        
        for batch in self.eval_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch.get('labels', input_ids).to(self.device)
            
            teacher_logits, teacher_hidden = self._get_teacher_hidden(input_ids)
            student_logits, student_hidden = self._get_student_hidden(input_ids)
            
            loss, _ = self.criterion(
                student_logits, teacher_logits,
                student_hidden, teacher_hidden,
                labels,
            )
            
            # Compute top-1 agreement
            teacher_preds = teacher_logits.argmax(dim=-1)
            student_preds = student_logits.argmax(dim=-1)
            agreement = (teacher_preds == student_preds).float().mean()
            
            total_loss += loss.item()
            total_agreement += agreement.item()
            n_batches += 1
        
        return {
            'eval_loss': total_loss / max(n_batches, 1),
            'agreement': total_agreement / max(n_batches, 1),
        }
    
    def train(self):
        """Main training loop."""
        if is_main_process():
            print(f"Starting distillation for {self.config.max_steps} steps")
            print(f"Teacher: {self.teacher.get_num_params() / 1e6:.1f}M params")
            print(f"Student: {sum(p.numel() for p in self.student.parameters()) / 1e6:.1f}M params")
        
        train_iter = iter(self.train_dataloader)
        
        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            # Train step
            loss_dict = self.train_step(batch)
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_steps == 0 and is_main_process():
                print(
                    f"Step {self.global_step} | "
                    f"Loss: {loss_dict['total_loss']:.4f} | "
                    f"KL: {loss_dict.get('kl_loss', 0):.4f} | "
                    f"MSE: {loss_dict.get('mse_loss', 0):.4f} | "
                    f"LR: {loss_dict['lr']:.2e}"
                )
            
            # Evaluation
            if self.global_step % self.config.eval_steps == 0:
                eval_dict = self.evaluate()
                if is_main_process() and eval_dict:
                    print(
                        f"Eval | Loss: {eval_dict['eval_loss']:.4f} | "
                        f"Agreement: {eval_dict['agreement']:.2%}"
                    )
                    
                    if eval_dict['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_dict['eval_loss']
                        self.save_student('best')
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_student(f'step_{self.global_step}')
        
        # Final save
        self.save_student('final')
        if is_main_process():
            print("Distillation complete!")
    
    def save_student(self, name: str, output_dir: str = './distilled'):
        """Save student model."""
        if not is_main_process():
            return
        
        from pathlib import Path
        import json
        
        path = Path(output_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {k: v for k, v in self.student.config.__dict__.items()}
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.student.state_dict(), path / 'model.pt')
        
        print(f"Saved student to {path}")


def distill(
    teacher_path: str,
    output_dir: str = './distilled',
    max_steps: int = 10000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
) -> SSMStudent:
    """
    Convenience function for distillation.
    
    Args:
        teacher_path: Path to trained SHC teacher
        output_dir: Output directory for student
        max_steps: Training steps
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Trained SSM student
    """
    from shc.data.dataset import SyntheticDataset
    from shc.data.dataloader import create_dataloader
    
    # Load teacher
    teacher = SHCTransformer.from_pretrained(teacher_path)
    
    # Create student
    student = SSMStudent.from_teacher_config(teacher.config)
    
    # Create data (using synthetic for demo)
    dataset = SyntheticDataset(
        num_samples=max_steps * batch_size,
        seq_len=teacher.config.max_seq_len,
        vocab_size=teacher.config.vocab_size,
    )
    dataloader = create_dataloader(dataset, batch_size=batch_size)
    
    # Config
    config = DistillationConfig(
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # Train
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        config=config,
        train_dataloader=dataloader,
    )
    trainer.train()
    
    return student
