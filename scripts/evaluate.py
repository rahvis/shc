#!/usr/bin/env python3
"""
SHC Evaluation Script

Run benchmarks and efficiency profiling on SHC models.

Usage:
    python -m shc.scripts.evaluate --model_path path/to/model --benchmarks bbh gsm8k mmlu
    python -m shc.scripts.evaluate --model_path path/to/model --profile
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List

import torch

from shc.models.transformer import SHCTransformer
from shc.models.ssm_student import SSMStudent
from shc.evaluation.benchmarks import BenchmarkSuite, BenchmarkResult
from shc.evaluation.profiler import EfficiencyProfiler, RoutingAnalyzer, profile_kv_cache_savings


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SHC models')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='shc',
                       choices=['shc', 'ssm'],
                       help='Model type')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Tokenizer name or path')
    
    # Benchmarks
    parser.add_argument('--benchmarks', type=str, nargs='+',
                       default=['bbh', 'gsm8k', 'mmlu'],
                       help='Benchmarks to run')
    parser.add_argument('--bbh_path', type=str, default=None,
                       help='Path to BBH data')
    parser.add_argument('--gsm8k_path', type=str, default=None,
                       help='Path to GSM8K data')
    parser.add_argument('--mmlu_path', type=str, default=None,
                       help='Path to MMLU data')
    
    # Profiling
    parser.add_argument('--profile', action='store_true',
                       help='Run efficiency profiling')
    parser.add_argument('--profile_generation', action='store_true',
                       help='Profile generation speed')
    parser.add_argument('--analyze_routing', action='store_true',
                       help='Analyze SHC routing behavior')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for profiling')
    parser.add_argument('--seq_length', type=int, default=512,
                       help='Sequence length for profiling')
    
    return parser.parse_args()


def load_model(args) -> torch.nn.Module:
    """Load model from checkpoint."""
    print(f"Loading model from {args.model_path}...")
    
    if args.model_type == 'shc':
        model = SHCTransformer.from_pretrained(args.model_path)
    else:
        # Load SSM student
        from shc.models.ssm_student import SSMConfig
        import json
        
        config_path = Path(args.model_path) / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = SSMConfig(**config_dict)
        
        model = SSMStudent(config)
        weights = torch.load(Path(args.model_path) / 'model.pt', map_location='cpu')
        model.load_state_dict(weights)
    
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    
    return model


def load_tokenizer(args):
    """Load tokenizer."""
    try:
        from transformers import AutoTokenizer
        
        tokenizer_name = args.tokenizer or 'meta-llama/Llama-2-7b-hf'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return tokenizer
    except ImportError:
        print("transformers not installed, using dummy tokenizer")
        
        # Simple dummy tokenizer
        class DummyTokenizer:
            def encode(self, text, return_tensors=None):
                tokens = [i % 1000 for i in range(len(text.split()))]
                if return_tensors == 'pt':
                    return torch.tensor([tokens])
                return tokens
            
            def decode(self, tokens):
                return ' '.join(['word'] * len(tokens))
        
        return DummyTokenizer()


def run_benchmarks(model, tokenizer, args) -> List[BenchmarkResult]:
    """Run benchmark suite."""
    print("\n" + "=" * 50)
    print("RUNNING BENCHMARKS")
    print("=" * 50)
    
    suite = BenchmarkSuite(
        bbh_path=args.bbh_path,
        gsm8k_path=args.gsm8k_path,
        mmlu_path=args.mmlu_path,
    )
    
    device = torch.device(args.device)
    results = suite.evaluate_all(model, tokenizer, device, args.benchmarks)
    
    suite.print_results(results)
    
    return results


def run_profiling(model, args) -> dict:
    """Run efficiency profiling."""
    print("\n" + "=" * 50)
    print("RUNNING EFFICIENCY PROFILING")
    print("=" * 50)
    
    device = torch.device(args.device)
    profiler = EfficiencyProfiler(model, device)
    
    # Create synthetic input
    input_ids = torch.randint(
        0, model.config.vocab_size if hasattr(model, 'config') else 32000,
        (args.batch_size, args.seq_length),
        device=device,
    )
    
    results = {}
    
    # Forward profiling
    print("\nProfiling forward pass...")
    fwd_result = profiler.profile_forward(input_ids)
    results['forward'] = fwd_result.to_dict()
    print(f"  Throughput: {fwd_result.tokens_per_second:.0f} tokens/sec")
    print(f"  Latency: {fwd_result.forward_time_ms:.2f} ms")
    print(f"  Memory: {fwd_result.peak_memory_mb:.0f} MB")
    
    # Training step profiling
    print("\nProfiling training step...")
    labels = input_ids.clone()
    train_result = profiler.profile_training_step(input_ids, labels)
    results['training'] = train_result.to_dict()
    print(f"  Forward: {train_result.forward_time_ms:.2f} ms")
    print(f"  Backward: {train_result.backward_time_ms:.2f} ms")
    print(f"  Memory: {train_result.peak_memory_mb:.0f} MB")
    
    # Generation profiling
    if args.profile_generation:
        print("\nProfiling generation...")
        prompt = input_ids[:, :32]  # Short prompt
        gen_result = profiler.profile_generation(prompt, max_new_tokens=100)
        results['generation'] = gen_result.to_dict()
        print(f"  Throughput: {gen_result.tokens_per_second:.0f} tokens/sec")
    
    # SHC routing profiling
    if args.analyze_routing and hasattr(model, 'layers'):
        print("\nProfiling SHC routing...")
        routing_results = profiler.profile_shc_routing(input_ids)
        results['routing'] = routing_results
        print(f"  Routing overhead: {routing_results['routing_time_ms']:.2f} ms")
        
        if routing_results['spectral_norms']:
            norms = list(routing_results['spectral_norms'].values())
            print(f"  Avg spectral norm: {sum(norms)/len(norms):.4f}")
    
    # KV cache savings
    print("\nEstimating KV cache savings...")
    cache_results = profile_kv_cache_savings(model)
    results['kv_cache'] = cache_results
    for seq_len, stats in cache_results.items():
        print(f"  Seq {seq_len}: {stats['reduction']:.1f}x reduction")
    
    return results


def run_routing_analysis(model, args) -> dict:
    """Analyze SHC routing behavior."""
    print("\n" + "=" * 50)
    print("ANALYZING SHC ROUTING")
    print("=" * 50)
    
    device = torch.device(args.device)
    analyzer = RoutingAnalyzer(model)
    
    # Generate synthetic batches
    vocab_size = model.config.vocab_size if hasattr(model, 'config') else 32000
    
    print("Collecting routing statistics...")
    for _ in range(10):  # 10 batches
        input_ids = torch.randint(
            0, vocab_size,
            (args.batch_size, args.seq_length),
            device=device,
        )
        analyzer.analyze_batch(input_ids)
    
    summary = analyzer.get_summary()
    
    print("\nRouting Statistics:")
    print(f"  Spectral norm (mean): {summary['spectral_norms']['mean']:.4f}")
    print(f"  Spectral norm (max): {summary['spectral_norms']['max']:.4f}")
    print(f"  Mixing entropy (mean): {summary['mixing_entropy']['mean']:.4f}")
    print(f"  Effective rank (mean): {summary['effective_ranks']['mean']:.2f}")
    
    return summary


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    model = load_model(args)
    tokenizer = load_tokenizer(args)
    
    results = {}
    
    # Run benchmarks
    if args.benchmarks:
        benchmark_results = run_benchmarks(model, tokenizer, args)
        results['benchmarks'] = [r.to_dict() for r in benchmark_results]
    
    # Run profiling
    if args.profile:
        profile_results = run_profiling(model, args)
        results['profiling'] = profile_results
    
    # Run routing analysis
    if args.analyze_routing:
        routing_results = run_routing_analysis(model, args)
        results['routing_analysis'] = routing_results
    
    # Save results
    output_file = output_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
