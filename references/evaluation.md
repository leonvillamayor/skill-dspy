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

# Usage example
evaluator = dspy.Evaluate(
    devset=devset,
    metric=dspy.SemanticF1(decompositional=True),
    num_threads=8,
)

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

# Prepare custom dataset
examples = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is 3*3?", answer="9").with_inputs("question"),
]
```

---

## dspy.Assert and dspy.Suggest

Constrain LM outputs with runtime assertions. DSPy automatically retries with feedback when constraints fail.

### dspy.Assert (hard constraint)
Raises `dspy.primitives.assertions.DSPyAssertionError` if the condition fails after max retries.

```python
class FactualAnswer(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer, sources: list[str]")

    def forward(self, question):
        pred = self.answer(question=question)

        # Hard constraint: must have at least one source
        dspy.Assert(
            len(pred.sources) >= 1,
            "You must provide at least one source for your answer."
        )

        # Hard constraint: answer must be substantive
        dspy.Assert(
            len(pred.answer.split()) >= 5,
            "Answer must be at least 5 words long."
        )

        return pred
```

### dspy.Suggest (soft constraint)
Logs a warning and provides feedback but does not raise an exception.

```python
class QualityAnswer(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        pred = self.answer(question=question)

        # Soft: suggest without enforcing
        dspy.Suggest(
            "?" not in pred.answer,
            "The answer should be definitive, not a question."
        )

        dspy.Suggest(
            len(pred.answer.split()) >= 10,
            "Consider providing a more detailed answer."
        )

        return pred
```

### Assertion configuration

```python
# Control max retries globally
dspy.configure(max_errors=5)

# Programmatic backtracking wrapper (alternative to inline dspy.Assert):
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

module_with_assertions = assert_transform_module(
    MyModule(),
    backtrack_handler,
    max_backtracks=3   # number of retry attempts per assertion failure
)
# This wraps the module so assertions trigger automatic retries with feedback,
# without needing to inline dspy.Assert/dspy.Suggest in your forward() method.
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
