"""
SHC Trainer

Main training loop with support for:
- Multi-GPU training (DDP/FSDP)
- Mixed precision (BF16/FP16)
- Gradient accumulation
- Checkpointing
- Logging (TensorBoard, WandB)
"""

from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import os
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from shc.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_world_size,
    get_rank,
    all_reduce_mean,
    wrap_model_ddp,
    wrap_model_fsdp,
)
from shc.training.optimizer import (
    create_optimizer,
    create_scheduler,
    GradientAccumulator,
    GradientClipper,
)


@dataclass
class TrainingArgs:
    """Arguments for training configuration."""
    
    # Output
    output_dir: str = './output'
    run_name: str = 'shc_training'
    
    # Training hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 100000
    warmup_steps: int = 2000
    
    # Batch configuration
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Precision
    mixed_precision: str = 'bf16'  # 'fp32', 'fp16', 'bf16'
    
    # Distributed
    use_ddp: bool = True
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = 'FULL_SHARD'
    
    # Checkpointing
    save_steps: int = 5000
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    log_steps: int = 100
    eval_steps: int = 1000
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # Scheduler
    scheduler_type: str = 'cosine'
    min_lr_ratio: float = 0.1
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Validate and process arguments."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class SHCTrainer:
    """
    Main trainer class for SHC models.
    
    Handles:
        - Distributed training setup (DDP/FSDP)
        - Mixed precision training
        - Gradient accumulation
        - Checkpointing and resuming
        - Logging to TensorBoard/WandB
    
    Args:
        model: SHC model to train
        args: Training arguments
        train_dataloader: Training data loader
        eval_dataloader: Optional evaluation data loader
        compute_loss: Optional custom loss function
        
    Example:
        >>> model = SHCTransformer(config)
        >>> trainer = SHCTrainer(model, args, train_loader)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArgs,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_loss: Optional[Callable] = None,
    ):
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_loss = compute_loss or self._default_loss
        
        # Setup distributed
        self.rank, self.local_rank, self.world_size = setup_distributed()
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap model for distributed training
        if self.world_size > 1:
            if args.use_fsdp:
                self.model = wrap_model_fsdp(
                    model,
                    mixed_precision=(args.mixed_precision != 'fp32'),
                    sharding_strategy=args.fsdp_sharding_strategy,
                )
            elif args.use_ddp:
                self.model = wrap_model_ddp(model)
            else:
                self.model = model
        else:
            self.model = model
        
        # Create optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=args.scheduler_type,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        
        # Mixed precision
        self.use_amp = args.mixed_precision != 'fp32'
        self.amp_dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float16
        self.scaler = GradScaler(enabled=(args.mixed_precision == 'fp16'))
        
        # Gradient utilities
        self.gradient_accumulator = GradientAccumulator(args.gradient_accumulation_steps)
        self.gradient_clipper = GradientClipper(args.max_grad_norm)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Logging
        self.writer = None
        if is_main_process() and args.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=args.output_dir / 'logs')
        
        # Set seed
        self._set_seed(args.seed)
        
        # Resume if specified
        if args.resume_from_checkpoint:
            self.load_checkpoint(args.resume_from_checkpoint)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        torch.manual_seed(seed + self.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + self.rank)
    
    def _default_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Default language modeling loss."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids[:, 1:]).to(self.device)
        
        logits = model(input_ids[:, :-1])
        
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dict of metrics (loss, grad_norm, etc.)
        """
        self.model.train()
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            loss = self.compute_loss(self.model, batch)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Accumulate gradients
        self.gradient_accumulator.current_step += 1
        
        metrics = {'loss': loss.item()}
        
        if self.gradient_accumulator.should_step():
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients
            grad_norm = self.gradient_clipper.clip(self.model.parameters())
            metrics['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Scheduler step
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Reset accumulator
            self.gradient_accumulator.reset()
            
            # Increment global step
            self.global_step += 1
            
            metrics['lr'] = self.scheduler.get_last_lr()[0]
        
        return metrics
    
    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            loss = self.compute_loss(self.model, batch)
        
        return {'eval_loss': loss.item()}
    
    def evaluate(self) -> Dict[str, float]:
        """Run full evaluation on eval_dataloader."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            metrics = self.eval_step(batch)
            total_loss += metrics['eval_loss']
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Reduce across processes
        avg_loss = all_reduce_mean(torch.tensor(avg_loss, device=self.device)).item()
        
        return {'eval_loss': avg_loss}
    
    def train(self):
        """Main training loop."""
        self.log(f"Starting training on {self.world_size} GPUs")
        self.log(f"Total steps: {self.args.max_steps}")
        self.log(f"Effective batch size: {self.args.batch_size * self.args.gradient_accumulation_steps * self.world_size}")
        
        train_iterator = iter(self.train_dataloader)
        
        start_time = time.time()
        running_loss = 0.0
        num_steps_in_window = 0
        
        while self.global_step < self.args.max_steps:
            # Get next batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            
            # Training step
            metrics = self.train_step(batch)
            running_loss += metrics['loss']
            num_steps_in_window += 1
            
            # Logging
            if self.global_step > 0 and self.global_step % self.args.log_steps == 0:
                avg_loss = running_loss / num_steps_in_window
                elapsed = time.time() - start_time
                steps_per_sec = self.global_step / elapsed
                
                if is_main_process():
                    self.log(
                        f"Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {metrics.get('lr', 0):.2e} | "
                        f"Steps/sec: {steps_per_sec:.2f}"
                    )
                    
                    if self.writer:
                        self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                        self.writer.add_scalar('train/lr', metrics.get('lr', 0), self.global_step)
                        if 'grad_norm' in metrics:
                            self.writer.add_scalar('train/grad_norm', metrics['grad_norm'], self.global_step)
                
                running_loss = 0.0
                num_steps_in_window = 0
            
            # Evaluation
            if self.global_step > 0 and self.global_step % self.args.eval_steps == 0:
                eval_metrics = self.evaluate()
                if is_main_process() and eval_metrics:
                    self.log(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
                    if self.writer:
                        self.writer.add_scalar('eval/loss', eval_metrics['eval_loss'], self.global_step)
                    
                    # Track best model
                    if eval_metrics['eval_loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['eval_loss']
                        self.save_checkpoint('best')
            
            # Checkpointing
            if self.global_step > 0 and self.global_step % self.args.save_steps == 0:
                self.save_checkpoint(f'step_{self.global_step}')
        
        # Final save
        self.save_checkpoint('final')
        
        if self.writer:
            self.writer.close()
        
        self.log("Training complete!")
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint."""
        if not is_main_process():
            return
        
        checkpoint_dir = self.args.output_dir / 'checkpoints' / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state (handle DDP/FSDP)
        model_to_save = self.model
        if hasattr(self.model, 'module'):
            model_to_save = self.model.module
        
        checkpoint = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_eval_loss': self.best_eval_loss,
            'args': self.args,
        }
        
        torch.save(checkpoint, checkpoint_dir / 'checkpoint.pt')
        self.log(f"Saved checkpoint: {checkpoint_dir}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / 'checkpoint.pt'
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        model_to_load = self.model
        if hasattr(self.model, 'module'):
            model_to_load = self.model.module
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        
        self.log(f"Resumed from checkpoint: {checkpoint_path} (step {self.global_step})")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoint_dir = self.args.output_dir / 'checkpoints'
        
        if not checkpoint_dir.exists():
            return
        
        # Get all step checkpoints
        step_dirs = sorted([
            d for d in checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith('step_')
        ], key=lambda d: int(d.name.split('_')[1]))
        
        # Remove oldest checkpoints
        while len(step_dirs) > self.args.save_total_limit:
            oldest = step_dirs.pop(0)
            import shutil
            shutil.rmtree(oldest)
    
    def log(self, message: str):
        """Log message (only on main process)."""
        if is_main_process():
            print(f"[Rank {self.rank}] {message}")
