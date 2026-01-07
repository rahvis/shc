"""
Basic Usage Example

Demonstrates how to create and use an SHC Transformer model
for text generation.
"""

import torch
from shc.models import SHCTransformer, SHCTransformerConfig, get_config


def main():
    """Basic usage example for SHC Transformer."""
    
    # Option 1: Use predefined configuration
    print("Loading 500M configuration...")
    config = get_config('500m')
    
    # Option 2: Create custom configuration
    # config = SHCTransformerConfig(
    #     vocab_size=32000,
    #     hidden_dim=1024,
    #     n_layers=24,
    #     n_heads=16,
    #     max_seq_len=4096,
    # )
    
    print(f"Configuration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    
    # Create model
    print("\nCreating model...")
    model = SHCTransformer(config)
    
    # Print model info
    n_params = model.get_num_params()
    print(f"Model parameters: {n_params:,}")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Example forward pass
    print("\nRunning forward pass...")
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Example generation
    print("\nGenerating text...")
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device)
    
    output = model.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    
    print(f"Prompt tokens: {prompt.shape[1]}")
    print(f"Generated tokens: {output.shape[1]}")
    
    # Get routing statistics
    print("\nRouting statistics:")
    stats = model.get_routing_stats(input_ids)
    for layer_idx, layer_stats in stats.items():
        if isinstance(layer_stats, dict):
            print(f"  Layer {layer_idx}: spectral_norm={layer_stats.get('spectral_norm', 'N/A')}")


if __name__ == "__main__":
    main()
