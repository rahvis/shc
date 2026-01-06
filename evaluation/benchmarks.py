"""
Benchmark Suite

Evaluation on standard LLM benchmarks:
- BBH (BIG-Bench Hard): Reasoning tasks
- GSM8K: Grade school math
- MMLU: Multitask language understanding

Reference: Table I and II of the SHC paper
"""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import re

import torch
import torch.nn.functional as F
from torch import Tensor

from shc.evaluation.metrics import compute_exact_match, MetricAccumulator


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    
    benchmark_name: str
    accuracy: float
    num_examples: int
    num_correct: int
    
    # Additional metrics
    per_category: Dict[str, float] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'benchmark': self.benchmark_name,
            'accuracy': self.accuracy,
            'num_examples': self.num_examples,
            'num_correct': self.num_correct,
            'per_category': self.per_category,
        }
    
    def __repr__(self) -> str:
        return f"{self.benchmark_name}: {self.accuracy:.2%} ({self.num_correct}/{self.num_examples})"


class BBHBenchmark:
    """
    BIG-Bench Hard benchmark for reasoning tasks.
    
    23 challenging tasks from BIG-Bench that require
    multi-step reasoning.
    
    Args:
        data_path: Path to BBH data directory
        n_shot: Number of few-shot examples
    """
    
    TASKS = [
        'boolean_expressions',
        'causal_judgement',
        'date_understanding',
        'disambiguation_qa',
        'dyck_languages',
        'formal_fallacies',
        'geometric_shapes',
        'hyperbaton',
        'logical_deduction_five_objects',
        'logical_deduction_seven_objects',
        'logical_deduction_three_objects',
        'movie_recommendation',
        'multistep_arithmetic_two',
        'navigate',
        'object_counting',
        'penguins_in_a_table',
        'reasoning_about_colored_objects',
        'ruin_names',
        'salient_translation_error_detection',
        'snarks',
        'sports_understanding',
        'temporal_sequences',
        'tracking_shuffled_objects_five_objects',
    ]
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        n_shot: int = 3,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.n_shot = n_shot
    
    def load_task(self, task_name: str) -> List[Dict[str, str]]:
        """Load examples for a task."""
        if self.data_path is None:
            # Return synthetic examples for testing
            return [
                {'input': f'Question {i}: What is 2+2?', 'target': '4'}
                for i in range(10)
            ]
        
        task_file = self.data_path / f'{task_name}.json'
        if task_file.exists():
            with open(task_file, 'r') as f:
                data = json.load(f)
            return data['examples']
        return []
    
    def format_prompt(
        self,
        question: str,
        few_shot_examples: List[Dict[str, str]],
    ) -> str:
        """Format prompt with few-shot examples."""
        prompt = ""
        
        for ex in few_shot_examples[:self.n_shot]:
            prompt += f"Q: {ex['input']}\nA: {ex['target']}\n\n"
        
        prompt += f"Q: {question}\nA:"
        return prompt
    
    def extract_answer(self, generated_text: str) -> str:
        """Extract answer from generated text."""
        # Take first line after "A:"
        lines = generated_text.strip().split('\n')
        if lines:
            return lines[0].strip()
        return ""
    
    def evaluate(
        self,
        model,
        tokenizer,
        device: torch.device,
        max_samples_per_task: int = 50,
    ) -> BenchmarkResult:
        """
        Evaluate model on BBH.
        
        Args:
            model: Model with generate() method
            tokenizer: Tokenizer
            device: Device to run on
            max_samples_per_task: Max samples per task
            
        Returns:
            BenchmarkResult with accuracy
        """
        total_correct = 0
        total_examples = 0
        per_task_results = {}
        
        for task in self.TASKS:
            examples = self.load_task(task)[:max_samples_per_task]
            if not examples:
                continue
            
            task_correct = 0
            
            for i, ex in enumerate(examples):
                # Get few-shot examples (exclude current)
                few_shot = [e for j, e in enumerate(examples) if j != i]
                
                # Format prompt
                prompt = self.format_prompt(ex['input'], few_shot)
                
                # Tokenize
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                
                # Generate
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=32,
                        temperature=0.0,  # Greedy
                        do_sample=False,
                    )
                
                # Decode and extract answer
                generated = tokenizer.decode(output_ids[0][input_ids.size(1):])
                predicted = self.extract_answer(generated)
                
                # Check correctness
                if predicted.lower().strip() == ex['target'].lower().strip():
                    task_correct += 1
            
            task_acc = task_correct / len(examples) if examples else 0
            per_task_results[task] = task_acc
            total_correct += task_correct
            total_examples += len(examples)
        
        return BenchmarkResult(
            benchmark_name='BBH',
            accuracy=total_correct / max(total_examples, 1),
            num_examples=total_examples,
            num_correct=total_correct,
            per_category=per_task_results,
        )


class GSM8KBenchmark:
    """
    Grade School Math 8K benchmark.
    
    8,500 math word problems requiring multi-step reasoning.
    
    Args:
        data_path: Path to GSM8K data file
        n_shot: Number of few-shot examples
    """
    
    CHAIN_OF_THOUGHT_PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

"""
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        n_shot: int = 3,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.n_shot = n_shot
    
    def load_data(self) -> List[Dict[str, str]]:
        """Load GSM8K examples."""
        if self.data_path is None or not self.data_path.exists():
            # Return synthetic examples
            return [
                {
                    'question': 'If you have 5 apples and get 3 more, how many do you have?',
                    'answer': '8'
                }
                for _ in range(10)
            ]
        
        examples = []
        with open(self.data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                examples.append({
                    'question': data['question'],
                    'answer': self._extract_final_answer(data['answer']),
                })
        return examples
    
    def _extract_final_answer(self, answer_text: str) -> str:
        """Extract numerical answer from solution."""
        # Look for "#### <number>"
        match = re.search(r'####\s*([\d,.-]+)', answer_text)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback: find last number
        numbers = re.findall(r'[\d,.-]+', answer_text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def format_prompt(self, question: str) -> str:
        """Format with chain-of-thought examples."""
        return f"{self.CHAIN_OF_THOUGHT_PROMPT}Q: {question}\nA:"
    
    def extract_answer(self, generated_text: str) -> str:
        """Extract numerical answer from generation."""
        # Look for "The answer is X"
        match = re.search(r'[Tt]he answer is\s*([\d,.-]+)', generated_text)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback: last number
        numbers = re.findall(r'[\d,.-]+', generated_text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    def evaluate(
        self,
        model,
        tokenizer,
        device: torch.device,
        max_samples: int = 500,
    ) -> BenchmarkResult:
        """Evaluate on GSM8K."""
        examples = self.load_data()[:max_samples]
        
        correct = 0
        
        for ex in examples:
            prompt = self.format_prompt(ex['question'])
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                )
            
            generated = tokenizer.decode(output_ids[0][input_ids.size(1):])
            predicted = self.extract_answer(generated)
            
            if predicted == ex['answer']:
                correct += 1
        
        return BenchmarkResult(
            benchmark_name='GSM8K',
            accuracy=correct / len(examples),
            num_examples=len(examples),
            num_correct=correct,
        )


class MMLUBenchmark:
    """
    Massive Multitask Language Understanding benchmark.
    
    57 subjects across STEM, humanities, social sciences, etc.
    
    Args:
        data_path: Path to MMLU data directory
        n_shot: Number of few-shot examples
    """
    
    SUBJECTS = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
        'clinical_knowledge', 'college_biology', 'college_chemistry',
        'college_computer_science', 'college_mathematics', 'college_medicine',
        'college_physics', 'computer_security', 'conceptual_physics',
        'econometrics', 'electrical_engineering', 'elementary_mathematics',
        'formal_logic', 'global_facts', 'high_school_biology',
        'high_school_chemistry', 'high_school_computer_science',
        'high_school_european_history', 'high_school_geography',
        'high_school_government_and_politics', 'high_school_macroeconomics',
        'high_school_mathematics', 'high_school_microeconomics',
        'high_school_physics', 'high_school_psychology', 'high_school_statistics',
        'high_school_us_history', 'high_school_world_history', 'human_aging',
        'human_sexuality', 'international_law', 'jurisprudence',
        'logical_fallacies', 'machine_learning', 'management', 'marketing',
        'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
        'nutrition', 'philosophy', 'prehistory', 'professional_accounting',
        'professional_law', 'professional_medicine', 'professional_psychology',
        'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'virology', 'world_religions',
    ]
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        n_shot: int = 5,
    ):
        self.data_path = Path(data_path) if data_path else None
        self.n_shot = n_shot
    
    def load_subject(self, subject: str) -> Tuple[List, List]:
        """Load dev (few-shot) and test examples for a subject."""
        if self.data_path is None:
            # Synthetic examples
            dev = [
                {'question': 'What is 2+2?', 'choices': ['3', '4', '5', '6'], 'answer': 1}
                for _ in range(5)
            ]
            test = dev.copy()
            return dev, test
        
        dev_file = self.data_path / 'dev' / f'{subject}_dev.csv'
        test_file = self.data_path / 'test' / f'{subject}_test.csv'
        
        dev = self._load_csv(dev_file)
        test = self._load_csv(test_file)
        
        return dev, test
    
    def _load_csv(self, path: Path) -> List[Dict]:
        """Load MMLU CSV file."""
        if not path.exists():
            return []
        
        examples = []
        import csv
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    examples.append({
                        'question': row[0],
                        'choices': [row[1], row[2], row[3], row[4]],
                        'answer': ord(row[5].upper()) - ord('A'),
                    })
        return examples
    
    def format_prompt(
        self,
        question: str,
        choices: List[str],
        few_shot: List[Dict],
        subject: str,
    ) -> str:
        """Format MMLU prompt."""
        prompt = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
        
        for ex in few_shot[:self.n_shot]:
            prompt += f"Question: {ex['question']}\n"
            for i, c in enumerate(ex['choices']):
                prompt += f"{chr(ord('A') + i)}. {c}\n"
            prompt += f"Answer: {chr(ord('A') + ex['answer'])}\n\n"
        
        prompt += f"Question: {question}\n"
        for i, c in enumerate(choices):
            prompt += f"{chr(ord('A') + i)}. {c}\n"
        prompt += "Answer:"
        
        return prompt
    
    def evaluate(
        self,
        model,
        tokenizer,
        device: torch.device,
        subjects: Optional[List[str]] = None,
        max_samples_per_subject: int = 100,
    ) -> BenchmarkResult:
        """Evaluate on MMLU."""
        subjects = subjects or self.SUBJECTS[:10]  # Subset for speed
        
        total_correct = 0
        total_examples = 0
        per_subject_results = {}
        
        for subject in subjects:
            dev, test = self.load_subject(subject)
            test = test[:max_samples_per_subject]
            
            if not test:
                continue
            
            subject_correct = 0
            
            for ex in test:
                prompt = self.format_prompt(
                    ex['question'],
                    ex['choices'],
                    dev,
                    subject,
                )
                
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                
                # Score each choice
                with torch.no_grad():
                    logits = model(input_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    
                    # Get logits for A, B, C, D tokens at last position
                    last_logits = logits[0, -1]
                    
                    # Assuming tokenizer has these tokens
                    try:
                        choice_ids = [tokenizer.encode(c)[0] for c in ['A', 'B', 'C', 'D']]
                        choice_logits = last_logits[choice_ids]
                        predicted = choice_logits.argmax().item()
                    except:
                        # Fallback: generate and parse
                        output_ids = model.generate(input_ids, max_new_tokens=1)
                        generated = tokenizer.decode(output_ids[0, -1:])
                        predicted = ord(generated.upper()[0]) - ord('A') if generated else -1
                
                if predicted == ex['answer']:
                    subject_correct += 1
            
            per_subject_results[subject] = subject_correct / len(test)
            total_correct += subject_correct
            total_examples += len(test)
        
        return BenchmarkResult(
            benchmark_name='MMLU',
            accuracy=total_correct / max(total_examples, 1),
            num_examples=total_examples,
            num_correct=total_correct,
            per_category=per_subject_results,
        )


class BenchmarkSuite:
    """
    Run all benchmarks and aggregate results.
    
    Args:
        bbh_path: Path to BBH data
        gsm8k_path: Path to GSM8K data
        mmlu_path: Path to MMLU data
        
    Example:
        >>> suite = BenchmarkSuite()
        >>> results = suite.evaluate_all(model, tokenizer, device)
        >>> for r in results:
        ...     print(r)
    """
    
    def __init__(
        self,
        bbh_path: Optional[str] = None,
        gsm8k_path: Optional[str] = None,
        mmlu_path: Optional[str] = None,
    ):
        self.bbh = BBHBenchmark(bbh_path)
        self.gsm8k = GSM8KBenchmark(gsm8k_path)
        self.mmlu = MMLUBenchmark(mmlu_path)
    
    def evaluate_all(
        self,
        model,
        tokenizer,
        device: torch.device,
        benchmarks: Optional[List[str]] = None,
    ) -> List[BenchmarkResult]:
        """
        Run all specified benchmarks.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            device: Device
            benchmarks: List of benchmarks ('bbh', 'gsm8k', 'mmlu')
            
        Returns:
            List of BenchmarkResult
        """
        benchmarks = benchmarks or ['bbh', 'gsm8k', 'mmlu']
        results = []
        
        if 'bbh' in benchmarks:
            print("Evaluating BBH...")
            results.append(self.bbh.evaluate(model, tokenizer, device))
        
        if 'gsm8k' in benchmarks:
            print("Evaluating GSM8K...")
            results.append(self.gsm8k.evaluate(model, tokenizer, device))
        
        if 'mmlu' in benchmarks:
            print("Evaluating MMLU...")
            results.append(self.mmlu.evaluate(model, tokenizer, device))
        
        return results
    
    def print_results(self, results: List[BenchmarkResult]):
        """Print formatted results table."""
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        
        for r in results:
            print(f"\n{r.benchmark_name}:")
            print(f"  Accuracy: {r.accuracy:.2%}")
            print(f"  Correct: {r.num_correct}/{r.num_examples}")
            
            if r.per_category:
                print("  Per-category:")
                for cat, acc in sorted(r.per_category.items(), key=lambda x: -x[1])[:5]:
                    print(f"    {cat}: {acc:.2%}")
        
        print("\n" + "=" * 50)
    
    def save_results(self, results: List[BenchmarkResult], path: str):
        """Save results to JSON."""
        output = {
            'results': [r.to_dict() for r in results],
            'summary': {
                r.benchmark_name: r.accuracy for r in results
            },
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
