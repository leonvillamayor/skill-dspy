# DSPy Optimizers (Teleprompters) — Complete Guide

Optimizers automatically improve your DSPy programs by tuning prompts, few-shot examples, and/or LM weights.

## How to Choose an Optimizer

```
Do you have labeled training data?
├── No → Use BestOfN or Refine (no optimization, runtime quality)
└── Yes →
    How many examples?
    ├── < 10 → LabeledFewShot (just sample labeled demos directly)
    ├── 10-50 → BootstrapFewShot (fast, minimal data)
    ├── 50-200 → MIPROv2 (auto="light") or COPRO or SIMBA
    └── > 200 → MIPROv2 (auto="medium"/"heavy") or BootstrapFinetune or GRPO

    Want to extract rules from data?
    └── Yes → InferRules

    Do you want RL-based optimization?
    └── Yes → GRPO or ArborGRPO (pip install arbor-ai)

    Do you want to fine-tune model weights?
    └── Yes → BootstrapFinetune

    Do you have multiple programs to combine?
    └── Yes → Ensemble
```

---

## LabeledFewShot

The simplest optimizer — randomly samples k labeled examples from the training set as demonstrations. No bootstrapping or generation.

```python
optimizer = dspy.LabeledFewShot(k=4)
optimized = optimizer.compile(program, trainset=trainset)
```

**Best for:** Quick baseline when you have good labeled examples and just want to use them directly.

---

## BootstrapFewShot

Generates labeled demonstrations from training data using the teacher model, then selects the best ones.

```python
optimizer = dspy.BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,    # max bootstrapped examples per predictor
    max_labeled_demos=4,         # max labeled examples from training set
    max_rounds=1,                # rounds of bootstrapping
    max_errors=5,                # stop after this many errors
    teacher_settings=dict(),     # override settings for teacher LM
    num_threads=1,
)
optimized = optimizer.compile(program, trainset=trainset)
```

**Best for:** Quick baseline optimization with few examples (5-50).

---

## BootstrapFewShotWithRandomSearch

Extends BootstrapFewShot by trying random subsets of demos and keeping the best combination.

```python
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=16,   # number of random candidates to try
    num_threads=8,
    max_rounds=1,
)
optimized = optimizer.compile(program, trainset=trainset, valset=devset)
```

**Best for:** Getting better few-shot selection than pure BootstrapFewShot without full optimization.

---

## BootstrapFewShotWithOptuna

Extends BootstrapFewShot with Optuna-based hyperparameter search for smarter demo selection.

```python
optimizer = dspy.BootstrapFewShotWithOptuna(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=16,
    num_threads=8,
)
optimized = optimizer.compile(program, trainset=trainset, valset=devset)
```

**Best for:** When you want automated hyperparameter tuning for few-shot selection.

---

## MIPROv2 (recommended default)

The most powerful general-purpose optimizer. Optimizes both instructions AND demonstrations using Bayesian optimization. Supports `auto` mode for hands-off configuration.

```python
optimizer = dspy.MIPROv2(
    metric=my_metric,
    auto="medium",          # "light", "medium", "heavy" — controls budget
    num_threads=16,
    prompt_model=dspy.LM('openai/gpt-4o'),   # optional: stronger model for generating proposals
    task_model=None,        # model to optimize for (defaults to configured LM)
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    num_batches=None,       # set by auto
    num_candidates=None,    # set by auto
    seed=42,
    verbose=False,
)
optimized = optimizer.compile(
    program,
    trainset=trainset,
    valset=devset,          # optional validation set
    max_bootstrapped_demos=3,
    max_labeled_demos=4,
    requires_permission_to_run=False,
)
```

**Auto mode budgets:**
- `"light"`: Fast run, ~1-2 min, good for development
- `"medium"`: Balanced, ~5-10 min, good for production
- `"heavy"`: Thorough, 30+ min, for maximum performance

**Best for:** Most use cases where you want strong prompt optimization.

---

## COPRO (Coordinate Ascent Prompt Optimizer)

Focuses on optimizing *instructions* only (not demonstrations). Uses coordinate ascent to improve each predictor's instruction independently.

```python
optimizer = dspy.COPRO(
    metric=my_metric,
    breadth=10,             # candidates per step
    depth=3,                # optimization steps
    init_temperature=1.4,   # sampling temperature for proposals
    verbose=False,
    prompt_model=None,      # model for generating proposals
)
optimized = optimizer.compile(
    program,
    trainset=trainset,
    eval_kwargs=dict(num_threads=8, display_progress=True),
)
```

**Best for:** When you want to optimize instructions without changing the few-shot examples.

---

## SIMBA (Stochastic Introspective Mini-Batch Adaptation)

Mini-batch stochastic optimization with self-reflection. More efficient than MIPROv2 for large programs.

```python
optimizer = dspy.SIMBA(
    metric=my_metric,
    num_threads=8,
    max_steps=8,            # optimization steps
    max_demos=4,            # max demo examples
    bsize=32,               # mini-batch size
)
optimized = optimizer.compile(program, trainset=trainset)
```

**Best for:** Large programs, faster convergence than MIPROv2 on some tasks.

---

## InferRules

Infers natural language rules from training examples for prompt augmentation. Extends BootstrapFewShot.

```python
optimizer = dspy.InferRules(
    metric=my_metric,
    num_candidates=10,
    num_rules=10,
    num_threads=8,
)
optimized = optimizer.compile(program, trainset=trainset)
```

**Best for:** When explicit rules can improve task performance (e.g., classification, formatting).

---

## GEPA (Generalized Evolutionary Prompting Algorithm)

Evolutionary optimizer using Pareto-front optimization and LLM-driven self-reflection. Can resume from checkpoints via `log_dir`.

**CRITICAL:** GEPA's metric must accept **5 arguments** (not 2-3 like other optimizers):

```python
def gepa_metric(gold, pred, trace, pred_name, pred_trace):
    """5-arg metric required for GEPA."""
    return gold.answer.lower() == pred.answer.lower()

optimizer = dspy.GEPA(
    metric=gepa_metric,

    # Budget — exactly one of:
    auto="medium",              # "light", "medium", "heavy"
    # max_full_evals=100,
    # max_metric_calls=500,

    # Reflection LM (required):
    reflection_lm=dspy.LM('openai/gpt-4o', temperature=1.0, max_tokens=32000),

    # Optional config:
    reflection_minibatch_size=3,
    candidate_selection_strategy="pareto",  # "pareto" or "current_best"
    skip_perfect_score=True,
    component_selector="round_robin",       # "round_robin" or "all"
    use_merge=True,
    max_merge_invocations=5,
    num_threads=8,
    log_dir=None,               # set to resume from checkpoint
    seed=0,
)

optimized = optimizer.compile(student=program, trainset=trainset, valset=valset)
```

**Best for:** Complex tasks where evolutionary Pareto optimization outperforms Bayesian approaches. New optimizer — try after MIPROv2 if you need higher performance.

---

## KNNFewShot

Dynamically selects the k nearest neighbors from the training set as few-shot examples at inference time (no compilation needed for selection).

```python
from dspy.retrieve.knn import KNN

optimizer = dspy.KNNFewShot(
    k=3,                    # number of examples to retrieve
    trainset=trainset,
    vectorizer=dspy.SentenceTransformersVectorizer(),  # or other vectorizer
)
optimized = optimizer.compile(program, trainset=trainset)
```

**Best for:** Large training sets where dynamic example selection improves relevance.

---

## Ensemble

Combines multiple compiled programs through voting or selection.

```python
# Compile multiple versions
prog1 = optimizer1.compile(program, trainset=trainset)
prog2 = optimizer2.compile(program, trainset=trainset)
prog3 = optimizer3.compile(program, trainset=trainset)

ensemble = dspy.Ensemble(reduce_fn=dspy.majority)
combined = ensemble.compile([prog1, prog2, prog3])
```

**Best for:** Maximizing performance by combining diverse programs.

---

## BootstrapFinetune

Fine-tunes the LM weights using bootstrapped training data. Requires a model that supports fine-tuning.

```python
optimizer = dspy.BootstrapFinetune(
    metric=my_metric,
    num_threads=16,
    max_bootstrapped_demos=4,
    teacher_settings=dict(),
)
optimized = optimizer.compile(
    program,
    trainset=trainset,
    train_kwargs={"epochs": 1, "lr": 1e-5},
)
```

**Best for:** When you need a smaller/cheaper model to match a larger model's performance.

---

## BetterTogether (Experimental)

Jointly optimizes prompt instructions and LM weights in alternating stages. Requires `dspy.settings.experimental = True`.

```python
import dspy
dspy.settings.experimental = True  # required

from dspy.teleprompt import BetterTogether

optimizer = BetterTogether(
    metric=my_metric,
    prompt_optimizer=None,   # defaults to BootstrapFewShotWithRandomSearch
    weight_optimizer=None,   # defaults to BootstrapFinetune
    seed=42,
)

# Strategy: sequence of "p" (prompt) and "w" (weight) stages
optimized = optimizer.compile(
    student=program,
    trainset=trainset,
    strategy="p -> w -> p",   # prompt opt → weight fine-tune → prompt opt again
    valset_ratio=0.1,
)
```

Valid strategy examples: `"p -> w"`, `"w -> p"`, `"p -> w -> p"`, `"w -> p -> w"`.

**Best for:** When you have enough data for fine-tuning (100+) and want the best of both worlds — better prompts AND better model weights.

---

## GRPO (Group Relative Policy Optimization)

RL-based fine-tuning optimizer. Uses GRPO to optimize model weights via reinforcement learning.

```python
optimizer = dspy.GRPO(
    metric=my_metric,
    num_train_steps=100,
    num_rollouts_per_grpo_step=4,
    num_dspy_examples_per_grpo_step=8,
    seed=42,
)
optimized = optimizer.compile(program, trainset=trainset)
```

**Best for:** RL-based weight optimization when you have enough compute and data.

---

## ArborGRPO (External)

External RL optimizer for multi-module GRPO. NOT part of core DSPy — install separately.

```
pip install arbor-ai
```

```python
from arbor import ArborGRPO
# See arbor-ai documentation for usage
```

---

## MIPROv2 — Teacher/Prompt Model Split

```python
tp = dspy.MIPROv2(
    metric=metric,
    auto="medium",
    num_threads=24,
    teacher_settings=dict(lm=dspy.LM('openai/gpt-4o')),  # LM for bootstrapping examples
    prompt_model=dspy.LM('openai/gpt-4o-mini'),           # LM for generating instructions
    max_errors=999,
)
optimized = tp.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    requires_permission_to_run=False,
)
```

Using different `teacher_settings` and `prompt_model` gives you: strong bootstrap examples (from gpt-4o) with cheap instruction generation (from gpt-4o-mini).

---

## SIMBA — Full Parameters

```python
optimizer = dspy.SIMBA(
    metric=metric,
    bsize=32,                       # mini-batch size
    num_candidates=6,               # candidate programs per iteration
    max_steps=8,                    # optimization steps
    max_demos=4,                    # max demos per predictor
    prompt_model=None,              # LM for evolving program (defaults to global)
    teacher_settings=None,
    demo_input_field_maxlen=100_000,
    num_threads=None,
    temperature_for_sampling=0.2,
    temperature_for_candidates=0.2,
)
optimized = optimizer.compile(student=program, trainset=trainset, seed=6793115)
```

---

## Optimization Best Practices

1. **Always split data:** Use `trainset` for optimization, `devset` for evaluation, `testset` for final assessment.

2. **Start with light optimization:**
   ```python
   # Development: fast iteration
   tp = dspy.MIPROv2(metric=metric, auto="light")

   # Production: full optimization
   tp = dspy.MIPROv2(metric=metric, auto="heavy")
   ```

3. **Use a stronger prompt_model:** Pass a more capable model to generate better proposals.

4. **Monitor with MLflow:**
   ```python
   import mlflow
   mlflow.dspy.autolog()
   optimized = tp.compile(program, trainset=trainset)
   ```

5. **Save compiled programs immediately:**
   ```python
   optimized.save("optimized_program.json")
   ```

6. **Introspect what changed:**
   ```python
   # Check the optimized signatures
   for name, pred in optimized.named_predictors():
       print(f"{name}: {pred.signature}")
       print(f"  Demos: {len(pred.demos)}")
   ```
