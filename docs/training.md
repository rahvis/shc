# Training Guide

This guide covers training SHC models from scratch with multi-GPU support.

## Basic Training

### Single GPU

```python
from shc.models import SHCTransformer, get_config
from shc.training import SHCTrainer, TrainingArgs

# Create model
model = SHCTransformer(get_config('500m'))

# Configure training
args = TrainingArgs(
    output_dir='./output',
    learning_rate=3e-4,
    max_steps=100000,
    batch_size=32,
)

# Create trainer
trainer = SHCTrainer(
    model=model,
    args=args,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
)

# Train
trainer.train()
```

### Multi-GPU with DDP

```bash
torchrun --nproc_per_node=8 -m shc.scripts.train \
    --model_size 3b \
    --batch_size 32 \
    --learning_rate 3e-4
```

### FSDP for Large Models

For 7B+ parameters, use Fully Sharded Data Parallel:

```bash
torchrun --nproc_per_node=8 -m shc.scripts.train \
    --model_size 7b \
    --use_fsdp \
    --mixed_precision bf16
```

## Training Configuration

```python
from shc.training import TrainingArgs

args = TrainingArgs(
    # Output
    output_dir='./output',
    run_name='shc_3b_run1',
    
    # Optimization
    learning_rate=3e-4,
    weight_decay=0.1,
    max_steps=100000,
    warmup_steps=2000,
    
    # Batch Configuration
    batch_size=32,
    gradient_accumulation_steps=8,
    
    # Precision
    mixed_precision='bf16',  # 'fp32', 'fp16', 'bf16'
    
    # Distributed
    use_ddp=True,
    use_fsdp=False,
    
    # Logging
    log_steps=100,
    eval_steps=1000,
    save_steps=5000,
    
    # Reproducibility
    seed=42,
)
```

## Hyperparameters by Scale

| Parameter | 500M | 3B | 7B |
|-----------|------|-----|-----|
| Learning Rate | 3e-4 | 3e-4 | 1.5e-4 |
| Batch Size | 1024 | 1024 | 512 |
| Warmup Steps | 2000 | 2000 | 4000 |
| Weight Decay | 0.1 | 0.1 | 0.1 |

## Data Loading

```python
from shc.data import TokenizedDataset, create_dataloader

# Create dataset
dataset = TokenizedDataset(
    data_path='path/to/tokenized_data',
    max_seq_len=2048,
)

# Create distributed dataloader
dataloader = create_dataloader(
    dataset,
    batch_size=32,
    num_workers=8,
    shuffle=True,
)
```

## Gradient Checkpointing

For memory-efficient training:

```python
config = get_config('7b')
config.use_gradient_checkpointing = True
model = SHCTransformer(config)
```

## Mixed Precision Training

```python
from shc.training import setup_distributed

# BF16 is preferred for stability
args = TrainingArgs(mixed_precision='bf16')

# FP16 requires loss scaling
args = TrainingArgs(mixed_precision='fp16')
```

## Checkpointing

```python
# Save checkpoint
trainer.save_checkpoint('checkpoint_50k')

# Load and resume
trainer.load_checkpoint('checkpoint_50k')
trainer.train()
```

## Monitoring

### TensorBoard

```python
args = TrainingArgs(
    use_tensorboard=True,
    log_steps=100,
)
```

```bash
tensorboard --logdir=./output
```

### Weights & Biases

```python
args = TrainingArgs(
    use_wandb=True,
    run_name='shc_experiment',
)
```

## SSM Distillation

For O(L) inference, distill trained model:

```python
from shc.training import DistillationTrainer, DistillationConfig

# Load trained teacher
teacher = SHCTransformer.from_pretrained('path/to/teacher')
teacher.eval()

# Create student
from shc.models import SSMStudent
student = SSMStudent.from_teacher_config(teacher.config)

# Configure distillation
config = DistillationConfig(
    max_steps=10000,
    learning_rate=1e-4,
    temperature=2.0,
    alpha_ce=0.5,
    alpha_kd=0.5,
)

# Distill
trainer = DistillationTrainer(teacher, student, config, train_loader)
trainer.train()

# Save student
student.save_pretrained('path/to/student')
```

## Best Practices

1. **Start Small**: Verify training works on 500M before scaling
2. **Monitor Spectral Norms**: Check `get_routing_stats()` periodically
3. **Use BF16**: More stable than FP16 for SHC training
4. **Gradient Clipping**: Default max_grad_norm=1.0 is recommended
