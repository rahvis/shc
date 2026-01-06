#!/usr/bin/env python3
"""
SHC Training Script

Train SHC Transformer models with multi-GPU support.

Usage:
    # Single GPU
    python -m shc.scripts.train --config configs/500m.yaml
    
    # Multi-GPU with DDP
    torchrun --nproc_per_node=8 -m shc.scripts.train --config configs/3b.yaml
    
    # FSDP for large models
    torchrun --nproc_per_node=8 -m shc.scripts.train --config configs/7b.yaml --use_fsdp
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shc.models.transformer import SHCTransformer, SHCTransformerConfig, get_config
from shc.training.trainer import SHCTrainer, TrainingArgs
from shc.training.distributed import setup_distributed, cleanup_distributed, is_main_process
from shc.data.dataset import SyntheticDataset, TokenizedDataset
from shc.data.dataloader import create_distributed_dataloader


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train SHC Transformer')
    
    # Model
    parser.add_argument('--model_size', type=str, default='500m',
                       choices=['500m', '1b', '3b', '7b'],
                       help='Predefined model size')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    # Training
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for checkpoints')
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name for logging')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Peak learning rate')
    parser.add_argument('--max_steps', type=int, default=100000,
                       help='Maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                       help='Warmup steps')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Distributed
    parser.add_argument('--use_ddp', action='store_true', default=True,
                       help='Use DDP for multi-GPU')
    parser.add_argument('--use_fsdp', action='store_true', default=False,
                       help='Use FSDP for memory-efficient training')
    
    # Precision
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                       choices=['fp32', 'fp16', 'bf16'],
                       help='Mixed precision mode')
    
    # Logging
    parser.add_argument('--log_steps', type=int, default=100,
                       help='Log every N steps')
    parser.add_argument('--save_steps', type=int, default=5000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                       help='Evaluate every N steps')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with small model and synthetic data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def create_model(args) -> SHCTransformer:
    """Create SHC Transformer model."""
    if args.config:
        # Load config from YAML
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = SHCTransformerConfig(**config_dict.get('model', {}))
    else:
        # Use predefined config
        config = get_config(args.model_size)
    
    # Override for debug mode
    if args.debug:
        config = SHCTransformerConfig(
            vocab_size=1000,
            hidden_dim=256,
            n_layers=4,
            n_heads=4,
            max_seq_len=512,
            n_streams=4,
            k_mixture=2,
        )
    
    # Enable gradient checkpointing for large models or FSDP
    if args.use_fsdp or args.model_size in ['7b']:
        config.use_gradient_checkpointing = True
    
    model = SHCTransformer(config)
    
    if is_main_process():
        print(f"Created model: {model.get_num_params() / 1e6:.1f}M parameters")
        print(f"Memory: {model.get_memory_footprint()['params_gb']:.2f} GB")
    
    return model


def create_dataloaders(args, config: SHCTransformerConfig):
    """Create training and evaluation dataloaders."""
    if args.debug:
        # Synthetic data for debugging
        train_dataset = SyntheticDataset(
            num_samples=10000,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed,
        )
        eval_dataset = SyntheticDataset(
            num_samples=1000,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed + 1,
        )
    else:
        # TODO: Load real dataset
        # For now, use synthetic data as placeholder
        train_dataset = SyntheticDataset(
            num_samples=100000,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed,
        )
        eval_dataset = SyntheticDataset(
            num_samples=1000,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            seed=args.seed + 1,
        )
    
    train_loader = create_distributed_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
    )
    
    eval_loader = create_distributed_dataloader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
    )
    
    return train_loader, eval_loader


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    if is_main_process():
        print(f"Starting training on {world_size} GPU(s)")
        print(f"Model size: {args.model_size}")
        print(f"Output dir: {args.output_dir}")
    
    try:
        # Create model
        model = create_model(args)
        
        # Create dataloaders
        train_loader, eval_loader = create_dataloaders(args, model.config)
        
        # Create training args
        run_name = args.run_name or f"shc_{args.model_size}"
        training_args = TrainingArgs(
            output_dir=args.output_dir,
            run_name=run_name,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            use_ddp=args.use_ddp,
            use_fsdp=args.use_fsdp,
            log_steps=args.log_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            resume_from_checkpoint=args.resume,
            seed=args.seed,
        )
        
        # Create trainer
        trainer = SHCTrainer(
            model=model,
            args=training_args,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
        )
        
        # Train
        trainer.train()
        
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
