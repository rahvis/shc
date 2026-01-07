"""
Distillation Example

Demonstrates how to distill an SHC Transformer teacher
into an SSM student for O(L) inference.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from shc.models import SHCTransformer, SHCTransformerConfig, SSMStudent
from shc.training import DistillationTrainer, DistillationConfig


def create_dummy_dataset(size: int, seq_len: int, vocab_size: int):
    """Create a dummy dataset for demonstration."""
    input_ids = torch.randint(0, vocab_size, (size, seq_len))
    return TensorDataset(input_ids)


def main():
    """Distillation example: SHC Transformer → SSM Student."""
    
    # Create teacher model
    teacher_config = SHCTransformerConfig(
        vocab_size=1000,
        hidden_dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=512,
    )
    
    teacher = SHCTransformer(teacher_config)
    teacher.eval()  # Teacher stays in eval mode
    
    print(f"Teacher parameters: {teacher.get_num_params():,}")
    
    # Create student model matching teacher dimensions
    student = SSMStudent.from_teacher_config(teacher_config)
    print(f"Student parameters: {student.get_num_params():,}")
    
    # Create training data
    train_dataset = create_dummy_dataset(
        size=1000,
        seq_len=128,
        vocab_size=teacher_config.vocab_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )
    
    # Configure distillation
    distill_config = DistillationConfig(
        max_steps=100,  # Small for demo
        learning_rate=1e-4,
        temperature=2.0,
        alpha_ce=0.5,      # Weight for cross-entropy loss
        alpha_kd=0.5,      # Weight for knowledge distillation loss
        warmup_steps=10,
    )
    
    # Create distillation trainer
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        config=distill_config,
        train_dataloader=train_loader,
    )
    
    # Run distillation
    print("\nStarting distillation...")
    trainer.train()
    
    # Compare inference speed
    print("\nComparing inference speed...")
    
    test_input = torch.randint(0, teacher_config.vocab_size, (1, 128))
    
    # Teacher (with KV cache)
    import time
    start = time.time()
    with torch.no_grad():
        teacher_out = teacher.generate(test_input[:, :10], max_new_tokens=50)
    teacher_time = time.time() - start
    
    # Student (O(1) per step, no KV cache)
    start = time.time()
    with torch.no_grad():
        student_out = student.generate(test_input[:, :10], max_new_tokens=50)
    student_time = time.time() - start
    
    print(f"Teacher generation time: {teacher_time:.3f}s")
    print(f"Student generation time: {student_time:.3f}s")
    print(f"Speedup: {teacher_time / student_time:.2f}x")
    
    # Save student model
    student.save_pretrained("./output/ssm_student")
    print("\nDistillation complete! Student saved to ./output/ssm_student")


if __name__ == "__main__":
    main()
