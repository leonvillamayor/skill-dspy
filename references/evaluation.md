# DSPy Evaluation, Metrics, Datasets & Assertions

## dspy.Evaluate

The primary class for running systematic evaluations.

```python
evaluator = dspy.Evaluate(
    devset=devset,              # list[dspy.Example]
    metric=my_metric,           # callable(example, pred, trace=None) -> float/bool
    num_threads=8,              # parallel threads
    display_progress=True,      # show tqdm progress bar
    display_table=5,            # show top N results in table
    max_errors=None,            # max errors before stopping (None = unlimited)
    provide_traceback=False,    # include tracebacks in errors
    failure_score=0.0,          # score assigned when evaluation raises an exception
    save_as_csv="results.csv",  # auto-save results to CSV
    save_as_json="results.json", # auto-save results to JSON
)

# Run evaluation — returns EvaluationResult
result = evaluator(program)
print(result.score)     # float percentage (0-100)
print(result.results)   # list of (example, prediction, score)

# Override at call time — useful for comparing metrics
result = evaluator(program, metric=other_metric, devset=other_devset,
                   callback_metadata={"run_id": 1}, save_as_csv="run1.csv")

# BREAKING CHANGE: `return_outputs` parameter has been removed.
# OLD (no longer works): score, outputs, scores = evaluate(program, return_outputs=True)
# NEW: use result.results instead
```

---

## Writing Metrics

A metric is any callable `(example, prediction, trace=None) -> float | bool`.

```python
# Boolean metric (pass/fail)
def exact_match(example, pred, trace=None):
    return example.answer.lower().strip() == pred.answer.lower().strip()

# Float metric (0.0-1.0 or 0-100)
def f1_score(example, pred, trace=None):
    gold_tokens = set(example.answer.lower().split())
    pred_tokens = set(pred.answer.lower().split())
    if not gold_tokens or not pred_tokens:
        return 0.0
    precision = len(gold_tokens & pred_tokens) / len(pred_tokens)
    recall = len(gold_tokens & pred_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Metric using another LM (LM-as-judge)
judge = dspy.ChainOfThought("question, answer, gold_answer -> correct: bool")

def llm_judge(example, pred, trace=None):
    result = judge(
        question=example.question,
        answer=pred.answer,
        gold_answer=example.answer
    )
    return result.correct

# Composite metric
def combined_metric(example, pred, trace=None):
    correctness = exact_match(example, pred)
    conciseness = len(pred.answer.split()) < 50
    return 0.7 * correctness + 0.3 * conciseness

# Optimization-aware metric (trace=None for eval, trace!=None for optimization)
def smart_metric(example, pred, trace=None):
    if trace is not None:
        # During optimization: stricter criteria
        return example.answer.lower() == pred.answer.lower()
    else:
        # During evaluation: allow partial matches
        return example.answer.lower() in pred.answer.lower()
```

---

## Built-in Metrics

```python
# Exact answer match
dspy.evaluate.answer_exact_match(example, pred, trace=None)

# With F1-threshold mode (frac parameter):
# Returns True if max token F1 between pred and any reference >= frac
dspy.evaluate.answer_exact_match(example, pred, frac=0.5)  # partial match

# Answer found in passage(s)
dspy.evaluate.answer_passage_match(example, pred, trace=None)

# SemanticF1 — LLM-based metric for open-ended generation
# Uses a ChainOfThought internally to compare key ideas
metric = dspy.SemanticF1(
    threshold=0.66,       # F1 threshold for pass/fail
    decompositional=True, # True = decompose into key ideas before comparing (better for long outputs)
)
score = metric(example, pred)   # returns float 0-1

# CompleteAndGrounded — evaluates completeness + groundedness for RAG
# Expects: example.question, example.response, pred.response, pred.context
metric = dspy.CompleteAndGrounded(threshold=0.66)
score = metric(example, pred)   # F1 of completeness and groundedness

# EM — alias for exact match
dspy.evaluate.EM  # same as answer_exact_match

# normalize_text — utility for text comparison
dspy.evaluate.normalize_text("Hello, World!!")  # "hello world"

# GSM8K built-in metric
from dspy.datasets.gsm8k import gsm8k_metric
gsm8k_metric(gold, pred)  # compares parsed integer answers

# Multi-reference answers work with answer_exact_match:
example = dspy.Example(answer=["Paris", "The city of Paris"])
pred = dspy.Prediction(answer="Paris")
dspy.evaluate.answer_exact_match(example, pred)  # True
```

---

## Built-in Datasets

```python
# HotPotQA — multi-hop question answering
from dspy.datasets import HotPotQA
dataset = HotPotQA(train_seed=2024, train_size=200, eval_seed=42)
trainset = dataset.train  # list[Example(question, answer)]
devset = dataset.dev

# GSM8K — grade school math
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
dataset = GSM8K()
trainset, devset = dataset.train, dataset.dev

# MATH dataset
from dspy.datasets.math import MATH
dataset = MATH()

# Colors dataset (~140 matplotlib color names, 60/40 split)
from dspy.datasets import Colors
dataset = Colors(sort_by_suffix=True)
trainset, devset = dataset.train, dataset.dev

# DataLoader — universal data loading
from dspy.datasets import DataLoader
loader = DataLoader()
dataset = loader.from_huggingface("dataset_name", split="train",
                                   fields=["question", "answer"], input_keys=["question"])
dataset = loader.from_csv("data.csv", fields=["q", "a"], input_keys=["q"])
dataset = loader.from_json("data.json")
dataset = loader.from_parquet("data.parquet")
dataset = loader.from_pandas(df, fields=["col1", "col2"])
train, test = loader.train_test_split(dataset, train_size=0.8)
sample = loader.sample(dataset, n=50)

# Dataset base class — for custom dataset classes
from dspy.datasets import Dataset
class MyDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.train, self.dev, self.test = ...

# Prepare custom dataset (simple)
examples = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3*3?", answer="9").with_inputs("question"),
]
```

---

## Runtime Constraints (formerly Assertions)

**`dspy.Assert` and `dspy.Suggest` were deprecated in DSPy 3.x.** Use `dspy.Refine` and `dspy.BestOfN` instead.

### dspy.Refine — Iterative improvement with feedback (replaces Assert)
After each failed attempt, DSPy generates hints ("Past Output" + "Instruction" fields) for the next run.

```python
def quality_check(example, pred, trace=None):
    has_sources = len(pred.sources) >= 1
    is_substantive = len(pred.answer.split()) >= 5
    return has_sources and is_substantive

refine = dspy.Refine(
    dspy.ChainOfThought("question -> answer, sources: list[str]"),
    N=3,              # max retry attempts with feedback
    reward_fn=quality_check,
    threshold=1.0,    # minimum reward to accept
)
result = refine(question="What causes tides?")
```

### dspy.BestOfN — Independent sampling (replaces Suggest)
N independent runs, picks best. No feedback between attempts.

```python
bon = dspy.BestOfN(
    dspy.ChainOfThought("question -> answer"),
    N=5,
    reward_fn=quality_check,
    threshold=1.0,
)
result = bon(question="What causes tides?")
```

---

## Evaluating with Traces

During optimization, DSPy passes `trace` to metrics to enable more informative scoring:

```python
def traced_metric(example, pred, trace=None):
    """
    trace: None during evaluation
    trace: list of (module_name, inputs, outputs) during compilation
    """
    score = compute_score(example, pred)

    if trace is not None:
        # Check intermediate reasoning quality
        for step_name, step_in, step_out in trace:
            if "reasoning" in step_out:
                if len(step_out.reasoning) < 20:
                    score *= 0.5   # penalize shallow reasoning

    return score
```

---

## Evaluation Patterns

### Split data properly
```python
import random
random.seed(42)
random.shuffle(all_examples)

n = len(all_examples)
trainset = all_examples[:int(0.6*n)]
devset = all_examples[int(0.6*n):int(0.8*n)]
testset = all_examples[int(0.8*n):]
```

### Compare programs
```python
evaluator = dspy.Evaluate(devset=devset, metric=metric, num_threads=8)

baseline_score = evaluator(baseline_program)
optimized_score = evaluator(optimized_program)
print(f"Improvement: {optimized_score.score - baseline_score.score:.1f}%")
```

### Save evaluation results
```python
evaluator = dspy.Evaluate(
    devset=devset,
    metric=metric,
    save_as_csv="eval_results.csv",
    save_as_json="eval_results.json",
)
result = evaluator(program)

# Access individual results
for example, pred, score in result.results:
    if score == 0:
        print(f"Failed: {example.question}")
        print(f"  Expected: {example.answer}")
        print(f"  Got: {pred.answer}")
```
