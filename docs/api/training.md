# Training API Reference

Training infrastructure with multi-GPU support.

## SHCTrainer

Main trainer class for SHC models.

```{eval-rst}
.. autoclass:: shc.training.SHCTrainer
   :members:
   :undoc-members:
   :show-inheritance:
```

## TrainingArgs

Configuration for training.

```{eval-rst}
.. autoclass:: shc.training.TrainingArgs
   :members:
   :undoc-members:
```

## DistillationTrainer

Trainer for SSM distillation.

```{eval-rst}
.. autoclass:: shc.training.DistillationTrainer
   :members:
   :undoc-members:
   :show-inheritance:
```

## DistillationConfig

Configuration for distillation.

```{eval-rst}
.. autoclass:: shc.training.DistillationConfig
   :members:
   :undoc-members:
```

## Optimizer Utilities

### create_optimizer

```{eval-rst}
.. autofunction:: shc.training.create_optimizer
```

### create_scheduler

```{eval-rst}
.. autofunction:: shc.training.create_scheduler
```

## Distributed Utilities

### setup_distributed

```{eval-rst}
.. autofunction:: shc.training.setup_distributed
```

### cleanup_distributed

```{eval-rst}
.. autofunction:: shc.training.cleanup_distributed
```
