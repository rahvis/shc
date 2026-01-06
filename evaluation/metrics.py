"""
Evaluation Metrics

Common metrics for language model evaluation.
"""

from typing import Optional, List, Dict, Any
import math

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_perplexity(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute perplexity from logits and labels.
    
    Perplexity = exp(cross_entropy_loss)
    
    Args:
        logits: Model output logits (batch, seq_len, vocab_size)
        labels: Target labels (batch, seq_len)
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Perplexity value
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Compute cross-entropy
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction='mean',
    )
    
    return math.exp(loss.item())


def compute_accuracy(
    predictions: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        predictions: Predicted token IDs
        targets: Target token IDs
        ignore_index: Index to ignore
        
    Returns:
        Accuracy (0 to 1)
    """
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    return correct.sum().item() / mask.sum().item()


def compute_f1(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute F1 score for text generation.
    
    Uses token-level overlap (bag of words).
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        
    Returns:
        Dict with precision, recall, f1
    """
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue
        
        overlap = len(pred_tokens & ref_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    n = len(predictions)
    return {
        'precision': total_precision / n,
        'recall': total_recall / n,
        'f1': total_f1 / n,
    }


def compute_exact_match(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        normalize: Normalize whitespace and case
        
    Returns:
        Exact match ratio
    """
    correct = 0
    
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = ' '.join(pred.lower().split())
            ref = ' '.join(ref.lower().split())
        
        if pred == ref:
            correct += 1
    
    return correct / len(predictions)


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU score.
    
    Args:
        predictions: List of predicted strings
        references: List of reference strings
        max_n: Maximum n-gram order
        
    Returns:
        Dict with BLEU scores
    """
    from collections import Counter
    
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    precisions = []
    
    for n in range(1, max_n + 1):
        matches = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            for ngram, count in pred_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
                total += count
        
        precision = matches / max(total, 1)
        precisions.append(precision)
    
    # Geometric mean
    if all(p > 0 for p in precisions):
        log_bleu = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = math.exp(log_bleu)
    else:
        bleu = 0.0
    
    return {
        f'bleu_{n}': precisions[n-1] for n in range(1, max_n + 1)
    } | {'bleu': bleu}


class MetricAccumulator:
    """
    Accumulate metrics across batches.
    
    Example:
        >>> acc = MetricAccumulator()
        >>> for batch in dataloader:
        ...     acc.update(loss=loss.item(), accuracy=acc_val)
        >>> results = acc.compute()
    """
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, **kwargs):
        """Add metric values."""
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = 0.0
                self.counts[name] = 0
            self.metrics[name] += value
            self.counts[name] += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute averaged metrics."""
        return {
            name: self.metrics[name] / self.counts[name]
            for name in self.metrics
        }
    
    def reset(self):
        """Reset accumulator."""
        self.metrics = {}
        self.counts = {}
