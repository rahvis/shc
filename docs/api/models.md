# Models API Reference

Complete model architectures built with SHC components.

## SHCTransformer

The main transformer model with sparse orthogonal routing.

```{eval-rst}
.. autoclass:: shc.models.SHCTransformer
   :members:
   :undoc-members:
   :show-inheritance:
```

## SHCTransformerConfig

Configuration dataclass for SHCTransformer.

```{eval-rst}
.. autoclass:: shc.models.SHCTransformerConfig
   :members:
   :undoc-members:
```

## SSMStudent

State Space Model for O(L) inference.

```{eval-rst}
.. autoclass:: shc.models.SSMStudent
   :members:
   :undoc-members:
   :show-inheritance:
```

## SSMConfig

Configuration for SSMStudent.

```{eval-rst}
.. autoclass:: shc.models.SSMConfig
   :members:
   :undoc-members:
```

## Utility Functions

### get_config

```{eval-rst}
.. autofunction:: shc.models.get_config
```

## Embedding Layers

### TokenEmbedding

```{eval-rst}
.. autoclass:: shc.models.TokenEmbedding
   :members:
   :undoc-members:
```

### PositionalEmbedding

```{eval-rst}
.. autoclass:: shc.models.PositionalEmbedding
   :members:
   :undoc-members:
```

### RMSNorm

```{eval-rst}
.. autoclass:: shc.models.RMSNorm
   :members:
   :undoc-members:
```
