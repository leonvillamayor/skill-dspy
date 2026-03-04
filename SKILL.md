---
name: skill-dspy
description: "Expert guide for building AI programs with DSPy — the declarative framework for LM programming with automatic prompt optimization. Use this skill PROACTIVELY whenever: importing dspy, using Signatures/Modules/Optimizers, building RAG/agent/multi-hop pipelines, optimizing with BootstrapFewShot/MIPROv2/COPRO/SIMBA/GEPA/BetterTogether, writing dspy.ChainOfThought/ReAct/Predict/CodeAct, evaluating with dspy.Evaluate, using dspy.Refine/BestOfN for runtime constraints, configuring any LM provider (OpenAI/Anthropic/Gemini/Ollama/reasoning models), saving/loading compiled programs, integrating MCP tools (stdio or HTTP), streaming, async modules, tracking token usage, or debugging with inspect_history. Covers: LM config, Signatures, Modules, Optimizers, Evaluation, Runtime Constraints, Tools, Adapters, Streaming, Async, Callbacks, Embeddings, Retrievers, and Save/Load."
license: MIT
metadata:
  author: skill-dspy
  version: "3.1.3"
---

# DSPy Expert Guide

DSPy is a declarative framework for programming language models. Instead of hand-writing prompts, you define *what* your program should do (via Signatures and Modules), and DSPy figures out *how* to prompt the LM to do it — including automatic optimization.

## Core Mental Model

```
Signature  →  defines I/O schema (what to compute)
Module     →  implements a reasoning strategy (how to compute)
Optimizer  →  tunes prompts/weights automatically (how to improve)
Evaluate   →  measures quality (how to measure)
```

---

## 1. Language Model Setup

DSPy uses [LiteLLM](https://litellm.ai/) under the hood, so any provider is supported.

```python
import dspy

# OpenAI
lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_KEY')

# Anthropic
lm = dspy.LM('anthropic/claude-sonnet-4-5-20250929', api_key='YOUR_KEY')

# Google Gemini
lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='YOUR_KEY')

# Ollama (local)
lm = dspy.LM('ollama_chat/llama3', api_base='http://localhost:11434')

# Custom OpenAI-compatible endpoint
lm = dspy.LM('openai/your-model', api_key='KEY', api_base='YOUR_URL')

# Configure globally
dspy.configure(lm=lm)

# Multiple LMs — use context managers for per-call override
with dspy.context(lm=dspy.LM('openai/gpt-4o')):
    result = my_module(question="...")
```

**Key LM parameters:**
- `temperature` — sampling temperature (default: 0.0 for deterministic)
- `max_tokens` — max output tokens
- `cache=True` — enable response caching (default True)
- `num_retries=3` — retries on transient failures
- `model_type` — `"chat"`, `"text"`, or `"responses"` (use `"responses"` for OpenAI reasoning models like o3/o4)
- `use_developer_role=True` — for OpenAI reasoning models that require developer system prompts
- `finetuning_model` — specify a separate model name for fine-tuning (vs inference)
- `callbacks=[...]` — per-LM callback hooks

**Reasoning models (o3, o4-mini, etc.):**
```python
lm = dspy.LM('openai/o3-mini', model_type='responses', temperature=1.0, max_tokens=16000)
dspy.configure(lm=lm, adapter=dspy.TwoStepAdapter(dspy.LM('openai/gpt-4o-mini')))
```

**`dspy.configure` full options:** `lm`, `adapter` (default ChatAdapter), `callbacks`, `track_usage=True` (token tracking), `allow_tool_async_sync_conversion`, `experimental=True` (enables BetterTogether/BootstrapFinetune), `max_errors`.

**Token tracking:** `dspy.configure(lm=lm, track_usage=True)` → then `result.get_lm_usage()` returns `{'model': {'prompt_tokens': N, 'completion_tokens': N}}`.

---

## 2. Signatures

Signatures define the *input/output schema* of an LM call.

### Inline (string) signatures
```python
# Simple string: "inputs -> outputs"
predict = dspy.Predict("question -> answer")
cot = dspy.ChainOfThought("context, question -> answer")
```

### Class-based signatures (recommended for typed, documented fields)
```python
class Classify(dspy.Signature):
    """Classify sentiment of a product review."""
    review: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="confidence score 0-1")

class RAGAnswer(dspy.Signature):
    """Answer a question given retrieved context."""
    context: list[str] = dspy.InputField(desc="retrieved passages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="concise factual answer")
    citations: list[int] = dspy.OutputField(desc="indices of supporting passages")
```

### Signature manipulation
```python
ExtendedSig = MySignature.append("new_output", dspy.OutputField(desc="..."), type_=str)
UpdatedSig = MySignature.with_updated_fields("answer", desc="detailed explanation")
InstructedSig = MySignature.with_instructions("Always respond in Spanish.")
# Also: .prepend(), .insert(pos, ...), .delete(field), .equals(other), .input_fields, .output_fields

# Create signature programmatically
from dspy.signatures import make_signature
DynamicSig = make_signature({"question": dspy.InputField(), "answer": dspy.OutputField()})
```

**Supported field types:** `str`, `int`, `float`, `bool`, `list[T]`, `dict[K,V]`, `Optional[T]`, `Union[T,U]`, `Literal[...]`, `dspy.Image`, `dspy.Audio`, `dspy.History`, `dspy.Code`, `dspy.File`, custom Pydantic models.

**Field parameters:** `desc` (description), `prefix` (display name), `format` (formatter function), `parser` (output parser function), plus all Pydantic `Field` validators (`gt`, `min_length`, etc.).

**`dspy.Code["python"]`** — typed code output field. **`dspy.File`** — file data in pipelines (3.1.0+). **`dspy.Reasoning`** — captures native reasoning from reasoning models (o3, DeepSeek-R1); auto-injection was reverted in 3.1.3. See Section 10 for full types.

---

## 3. Built-in Modules

All modules inherit from `dspy.Module`. Use them directly or compose them.

| Module | Description | Typical Use |
|--------|-------------|-------------|
| `dspy.Predict` | Single LM call | Simple extraction, classification |
| `dspy.ChainOfThought` | CoT reasoning | Multi-step reasoning, explanation |
| `dspy.ProgramOfThought` | Code-based reasoning (needs Deno) | Math, symbolic computation |
| `dspy.ReAct` | Tool-use agent loop (max_iters=20) | Search, APIs, multi-tool agents |
| `dspy.CodeAct` | Python code execution (pure fns only) | Complex computations via code |
| `dspy.RLM` | Recursive LM — explores large contexts via REPL | Long documents, complex analysis (3.1.1+) |
| `dspy.MultiChainComparison` | Ensemble of CoT chains (M=3, temperature=0.7) | High-accuracy QA |
| `dspy.Refine` | Iterative refinement with feedback | Quality improvement loops |
| `dspy.BestOfN` | Sample N independently, pick best | Reliability via sampling |
| `dspy.Parallel` | Run modules in parallel | Batch processing |
| `dspy.KNN` | K-nearest neighbor example retrieval | Dynamic demo selection |

**`dspy.BestOfN` vs `dspy.Refine`:**
- `BestOfN(module, N=5, reward_fn=fn, threshold=1.0)` — N **independent** runs, picks the best. No feedback between attempts.
- `Refine(module, N=3, reward_fn=fn, threshold=1.0)` — N runs **with automatic feedback**. After each failed attempt, DSPy generates hints ("Past Output" + "Instruction" fields) for the next run. Use `Refine` when each attempt can learn from the previous one.

**`dspy.CodeAct` constraint:** only pure Python functions as tools — no lambdas, callable objects, or external libraries:
```python
act = dspy.CodeAct("n -> factorial_result", tools=[factorial_fn], max_iters=5, interpreter=None)
# max_iters default is 5; interpreter accepts a CodeInterpreter instance (defaults to PythonInterpreter)
```

**`dspy.Parallel` full API (3.1.2: `timeout` and `straggler_limit` now exposed):**
```python
parallel = dspy.Parallel(
    num_threads=8, timeout=120, straggler_limit=3,  # straggler_limit is int (default 3)
    return_failed_examples=False, provide_traceback=None, disable_progress_bar=False,
)
results = parallel([(module, example1), (module, example2)])

# Convenience: every dspy.Module has .batch()
results = my_module.batch(examples=[ex1, ex2, ex3], num_threads=4, return_failed_examples=True)
# If return_failed_examples=True: returns (results, failed_examples, exceptions)
```

**`dspy.RLM` — Recursive Language Model (3.1.1+):** Explores large contexts via sandboxed Python REPL. Requires Deno.
```python
rlm = dspy.RLM("context, query -> answer", max_iterations=20, max_llm_calls=50, sub_lm=None, tools=None)
result = rlm(context="...very large document...", query="What is the revenue?")
print(result.answer, result.trajectory)  # trajectory = list of {code, output} steps
```

**`dspy.PythonInterpreter` for code execution (requires Deno):**
```python
with dspy.PythonInterpreter() as interp:
    result = interp.execute("value = 2*5 + 4\nvalue")  # returns 14
```

**`dspy.KNN` — dynamic example retrieval:**
```python
knn = dspy.KNN(k=3, trainset=trainset, vectorizer=dspy.Embedder('openai/text-embedding-3-small'))
nearest = knn(question="What is DSPy?")  # returns k nearest training examples
```

### Quick usage
```python
result = dspy.Predict("question -> answer")(question="What is 2+2?")
result = dspy.ChainOfThought("question -> answer")(question="Explain gravity")  # adds .reasoning
result = dspy.ReAct("question -> answer", tools=[search_fn])(question="Who won?")
result = dspy.BestOfN(dspy.ChainOfThought("q -> a"), N=5, reward_fn=metric)(q="...")
```

---

## 4. Custom Modules

Build complex programs by composing modules:

```python
class RAG(dspy.Module):
    def __init__(self, num_docs=5):
        self.num_docs = num_docs
        self.retrieve = dspy.Retrieve(k=num_docs)          # if using a retriever
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

class MultiHopRAG(dspy.Module):
    def __init__(self, hops=2):
        self.generate_query = [dspy.ChainOfThought("context, question -> query") for _ in range(hops)]
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = []
        for i, gen_q in enumerate(self.generate_query):
            query = gen_q(context=context, question=question).query
            context += search(query)   # your search function
        return self.generate_answer(context=context, question=question)
```

**Module API:** `forward(**kw)` / `aforward(**kw)`, `__call__` / `acall`, `named_predictors()`, `predictors()`, `set_lm(lm)` / `get_lm()`, `map_named_predictors(fn)`, `deepcopy()`, `reset_copy()`, `save(path)` / `load(path)`, `dump_state()` / `load_state(state)`, `named_parameters()`, `named_sub_modules(type_=None)`, `inspect_history(n=1)`, `batch(examples, num_threads=...)`.

---

## 5. Data & Examples

```python
# dspy.Example — structured training/eval data
example = dspy.Example(question="What is DSPy?", answer="A framework for LM programming")

# Specify which fields are inputs
example = example.with_inputs("question")
print(example.inputs())   # {'question': ...}
print(example.labels())   # {'answer': ...}

# Also: .copy(answer="new"), .without("field"), .toDict(), .keys(), .values(), .items()

# dspy.Prediction — module output
pred = dspy.Prediction(answer="42", reasoning="step by step...")
pred.get_lm_usage()              # token usage if track_usage=True
pred.completions                  # raw completions
Prediction.from_completions(list_or_dict, signature=None)

# Built-in datasets (see references/evaluation.md for DataLoader, Colors, MATH, etc.)
from dspy.datasets import HotPotQA
hotpotqa = HotPotQA(train_seed=2024, train_size=500)
trainset, devset = hotpotqa.train, hotpotqa.dev
```

---

## 6. Optimizers (Teleprompters)

See `references/optimizers.md` for full details. Quick reference:

| Optimizer | Best For | Data Needed | Notes |
|-----------|----------|-------------|-------|
| `LabeledFewShot` | Direct labeled demos, simplest | Any size | Just assigns k random demos |
| `BootstrapFewShot` | Few-shot demos, fast | 5-50 examples | |
| `BootstrapFewShotWithRandomSearch` | Better few-shot selection | ~50-200 examples | Alias: `BootstrapRS` |
| `BootstrapFewShotWithOptuna` | Bayesian demo selection | ~50-200 examples | Requires `pip install optuna` |
| `MIPROv2` | Full prompt + demo optimization | 50-300 examples | **Default recommendation** |
| `COPRO` | Instruction-only optimization | 20-100 examples | |
| `SIMBA` | Mini-batch stochastic optimization | 20-200 examples | Faster for large programs |
| `GEPA` | Evolutionary prompt optimization | 50+ examples | 5-arg metric required |
| `AvatarOptimizer` | Iterative instruction improvement | 20-100 examples | Positive/negative comparison |
| `InferRules` | Rule discovery from examples | 20-100 examples | Induces NL rules into instructions |
| `BetterTogether` | Prompt + weight joint optimization | 100+ examples | Requires `experimental=True` |
| `KNNFewShot` | Dynamic example retrieval | Training set | |
| `Ensemble` | Combine multiple programs | Multiple programs | |
| `BootstrapFinetune` | Fine-tuning LM weights | 100+ examples | Requires `experimental=True` |

**GEPA critical note — its metric must accept 5 arguments:**
```python
def gepa_metric(gold, pred, trace, pred_name, pred_trace):
    return gold.answer.lower() == pred.answer.lower()
optimizer = dspy.GEPA(metric=gepa_metric, auto="medium", reflection_lm=dspy.LM('openai/gpt-4o'))
```

**BetterTogether (experimental) — combines prompt + weight optimization:**
```python
dspy.settings.experimental = True
from dspy.teleprompt import BetterTogether
optimizer = BetterTogether(metric=my_metric)
optimized = optimizer.compile(program, trainset=trainset, strategy="p -> w -> p")
```

```python
# Standard optimization pattern
optimizer = dspy.MIPROv2(metric=my_metric, auto="medium", num_threads=8)
optimized = optimizer.compile(my_program, trainset=trainset)
optimized.save("optimized.json")
```

---

## 7. Evaluation

See `references/evaluation.md` for full details.

```python
# Define a metric
def my_metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Run evaluation
evaluator = dspy.Evaluate(
    devset=devset,
    metric=my_metric,
    num_threads=8,
    display_progress=True,
    display_table=5,
)
score = evaluator(my_program)  # returns EvaluationResult

# Built-in metrics
dspy.evaluate.answer_exact_match    # also: dspy.evaluate.EM (alias)
dspy.evaluate.answer_passage_match
dspy.SemanticF1()                   # LLM-based semantic comparison
dspy.CompleteAndGrounded()          # completeness + groundedness for RAG
```

---

## 8. Runtime Constraints

**Note:** `dspy.Assert` and `dspy.Suggest` were deprecated in DSPy 3.x. Use `dspy.Refine` and `dspy.BestOfN` instead.

```python
# Refine — iterative improvement with automatic feedback (replaces Assert/Suggest)
def quality_check(example, pred, trace=None):
    return len(pred.answer) > 10 and float(pred.confidence) > 0.7

refine = dspy.Refine(
    dspy.ChainOfThought("question -> answer, confidence"),
    N=3, reward_fn=quality_check, threshold=1.0,
)
result = refine(question="...")

# BestOfN — sample N, pick best (no feedback between attempts)
bon = dspy.BestOfN(
    dspy.ChainOfThought("question -> answer, confidence"),
    N=5, reward_fn=quality_check, threshold=1.0,
)
result = bon(question="...")
```

---

## 9. Tools & MCP

See `references/advanced.md` for full details.

```python
# Define tools as Python functions (used in ReAct/CodeAct)
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

def search(query: str, k: int = 3) -> list[str]:
    """Search Wikipedia for passages."""
    results = dspy.ColBERTv2(url='http://...')(query, k=k)
    return [r['text'] for r in results]

agent = dspy.ReAct("question -> answer", tools=[calculator, search])

# MCP tool integration (async)
from mcp import ClientSession
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool_object)
agent = dspy.ReAct("question -> answer", tools=[dspy_tool])
```

---

## 10. Special Data Types

All types inherit from `dspy.Type` — subclass it for custom multimodal types with a `format()` method.

```python
# Image — unified constructor (from_url/from_file deprecated)
img = dspy.Image("https://example.com/image.jpg")  # also: local path, PIL.Image, data URI, bytes

# Audio
audio = dspy.Audio.from_file("speech.mp3")          # also: .from_url(), .from_array(np, sr, fmt)

# File (3.1.0+)
file = dspy.File.from_path("doc.pdf")               # also: .from_bytes(), .from_file_id()

# History — conversation context
history = dspy.History(messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}])

# Code — typed code output: dspy.Code["python"]
# ToolCalls — captures tool call results from LM outputs
# Reasoning — native reasoning from o3/DeepSeek-R1 (auto-injection reverted in 3.1.3)
```

Use in signatures: `image: dspy.Image = dspy.InputField()`, `history: dspy.History = dspy.InputField()`, etc.

---

## 11. Save & Load

```python
# State-only (JSON, recommended)
optimized_program.save("my_program.json")
loaded = MyProgramClass(); loaded.load("my_program.json")

# State-only (pickle) — for non-JSON-serializable state (e.g., dspy.Image)
optimized_program.save("my_program.pkl", save_program=False)

# Full program (architecture + state) — cloudpickle to directory
optimized_program.save("./my_program_dir/", save_program=True)
loaded = dspy.load("./my_program_dir/")
```

**Security:** `.pkl` files can execute arbitrary code — only load from trusted sources.

---

## Quick Patterns

### RAG pipeline
```python
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> answer')
    def forward(self, question):
        context = search(question)   # your retrieval function
        return self.respond(context=context, question=question)

rag = RAG()
tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium")
optimized_rag = tp.compile(rag, trainset=trainset)
```

### Classification with typed output
```python
class Classify(dspy.Module):
    def __init__(self, classes):
        self.classes = classes
        self.predict = dspy.Predict(
            dspy.Signature("text -> label").with_updated_fields(
                "label", type_=Literal[tuple(classes)]
            )
        )
    def forward(self, text):
        return self.predict(text=text)
```

### Inspect LM calls
```python
dspy.inspect_history(n=5)   # show last 5 LM interactions
```

---

## Reference Files

- `references/optimizers.md` — Deep dive: all optimizers with parameters and strategies
- `references/evaluation.md` — Evaluation, metrics, built-in datasets, assertions
- `references/advanced.md` — Adapters, callbacks, streaming, async, MCP, fine-tuning
