"""
Training Example

Demonstrates how to train an SHC Transformer model with
distributed training support.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from shc.models import SHCTransformer, SHCTransformerConfig
from shc.training import SHCTrainer, TrainingArgs


def create_dummy_dataset(size: int, seq_len: int, vocab_size: int):
    """Create a dummy dataset for demonstration."""
    input_ids = torch.randint(0, vocab_size, (size, seq_len))
    labels = torch.randint(0, vocab_size, (size, seq_len))
    return TensorDataset(input_ids, labels)


def main():
    """Training example for SHC Transformer."""
    
    # Create a small model for demonstration
    config = SHCTransformerConfig(
        vocab_size=1000,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=512,
    )
    
    model = SHCTransformer(config)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Create dummy training data
    train_dataset = create_dummy_dataset(
        size=1000,
        seq_len=128,
        vocab_size=config.vocab_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )
    
    # Create evaluation data
    eval_dataset = create_dummy_dataset(
        size=100,
        seq_len=128,
        vocab_size=config.vocab_size
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
    )
    
    # Configure training
    training_args = TrainingArgs(
        output_dir="./output",
        run_name="shc_demo",
        learning_rate=3e-4,
        weight_decay=0.1,
        max_steps=100,  # Small for demo
        warmup_steps=10,
        batch_size=8,
        gradient_accumulation_steps=1,
        log_steps=10,
        eval_steps=50,
        save_steps=100,
        mixed_precision="fp32",  # Use fp32 for CPU demo
    )
    
    # Create trainer
    trainer = SHCTrainer(
        model=model,
        args=training_args,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    model.save_pretrained("./output/final")
    print("\nTraining complete! Model saved to ./output/final")


if __name__ == "__main__":
    main()
