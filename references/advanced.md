# DSPy Advanced Features: Adapters, Callbacks, Streaming, Async, Tools, MCP

## Table of Contents
- [Adapters](#adapters)
- [Streaming](#streaming)
- [Async Support](#async-support)
- [Callbacks](#callbacks)
- [Tools](#tools)
- [MCP Integration](#mcp-model-context-protocol-integration)
- [Inspect and Debug](#inspect-and-debug)
- [Configuration Reference](#configuration-reference)
- [Fine-tuning Workflow](#fine-tuning-workflow)

---

## Adapters

Adapters control how DSPy formats messages to/from LMs. They're usually handled automatically, but you can override them.

```python
from dspy.adapters import ChatAdapter, JSONAdapter, XMLAdapter, TwoStepAdapter

# ChatAdapter — default for chat models, uses conversational turn format
dspy.configure(adapter=ChatAdapter())

# With native function calling (for tool use):
dspy.configure(adapter=ChatAdapter(use_native_function_calling=True))

# With automatic fallback to JSONAdapter on parse errors:
dspy.configure(adapter=ChatAdapter(use_json_adapter_fallback=True))

# JSONAdapter — forces JSON output, uses native function calling by default
# More reliable for structured outputs with standard chat models
dspy.configure(adapter=JSONAdapter())
dspy.configure(adapter=JSONAdapter(use_native_function_calling=False))  # disable

# XMLAdapter (3.0+) — formats fields using XML tags; supports streaming
dspy.configure(adapter=XMLAdapter())

# BAMLAdapter (3.0+) — uses BAML format instead of JSON schema for structured outputs
# Better quality for complex nested types; does NOT support streaming
# NOTE: BAMLAdapter is NOT in the default exports; import explicitly:
from dspy.adapters.baml_adapter import BAMLAdapter
dspy.configure(adapter=BAMLAdapter())

# Per-call adapter override via context manager
with dspy.context(adapter=JSONAdapter()):
    result = my_module(question="What is DSPy?")
```

**Streaming support by adapter:** ChatAdapter ✓, JSONAdapter ✓, XMLAdapter ✓, BAMLAdapter ✗

### TwoStepAdapter — for reasoning models (o3, o4-mini)

When the primary LM is a reasoning model that generates free-form output, TwoStep uses a smaller extraction model to parse the structured result:

```python
from dspy.adapters import TwoStepAdapter

# Step 1: o3 generates a free-form response
# Step 2: gpt-4o-mini extracts structured fields from it
dspy.configure(
    lm=dspy.LM('openai/o3-mini', model_type='responses', temperature=1.0, max_tokens=16000),
    adapter=TwoStepAdapter(dspy.LM('openai/gpt-4o-mini'))  # extraction_model must be a dspy.LM
)
program = dspy.ChainOfThought("question -> answer")
result = program(question="What is the capital of France?")
```

Use `TwoStepAdapter` whenever your primary LM struggles with strict JSON formatting.

---

## Streaming

DSPy supports streaming LM responses for real-time output display.

### `dspy.streamify` — full API

```python
import dspy

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

cot = dspy.ChainOfThought("question -> answer, reasoning")

# Wrap any module in streamify
stream_program = dspy.streamify(
    cot,
    # Capture specific output fields as they stream:
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="answer"),
        dspy.streaming.StreamListener(signature_field_name="reasoning"),
    ],
    include_final_prediction_in_output_stream=True,
    is_async_program=False,    # True if module uses aforward()
    async_streaming=True,      # False = return sync generator (no asyncio needed)
    status_message_provider=None,  # optional callback for status messages
)
# StreamListener full init:
# StreamListener(signature_field_name, predict=None, predict_name=None, allow_reuse=False)

# Async usage:
async def stream_async():
    async for value in stream_program(question="Explain quantum entanglement"):
        if isinstance(value, dspy.Prediction):
            final = value
        else:
            print(value, end="", flush=True)

# Sync usage (async_streaming=False):
sync_program = dspy.streamify(cot, async_streaming=False)
for chunk in sync_program(question="Explain quantum entanglement"):
    if isinstance(chunk, dspy.Prediction):
        final = chunk
    else:
        print(chunk, end="", flush=True)
```

---

## Async Support

DSPy modules support async operation via `acall` and `aforward`. All built-in modules expose `.acall()`:

```python
import asyncio
import dspy

dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

cot = dspy.ChainOfThought("question -> answer")

# Async call — works on any module
async def run():
    result = await cot.acall(question="What is async programming?")
    print(result.answer)

asyncio.run(run())

# Custom async module — implement aforward() instead of forward()
class AsyncRAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("context, question -> answer")

    async def aforward(self, question):
        context = await fetch_context_async(question)
        return await self.respond.acall(context=context, question=question)

# Parallel async execution
async def run_parallel(questions):
    module = AsyncRAG()
    tasks = [module.acall(question=q) for q in questions]
    return await asyncio.gather(*tasks)
```

### Async tools in sync context

```python
async def get_weather(city: str) -> str:
    """Get current weather."""
    ...

tool = dspy.Tool(get_weather)

# Option 1: context manager (one-time)
with dspy.context(allow_tool_async_sync_conversion=True):
    agent = dspy.ReAct("question -> answer", tools=[tool])
    result = agent(question="Weather in Tokyo?")

# Option 2: global (all calls)
dspy.configure(allow_tool_async_sync_conversion=True)

# Option 3: native async
result = await tool.acall(city="Tokyo")
```

---

## Callbacks

Callbacks hook into the LM call lifecycle for logging, monitoring, and custom behavior.

```python
from dspy.utils.callback import BaseCallback

class LoggingCallback(BaseCallback):
    # Module lifecycle
    def on_module_start(self, call_id, instance, inputs):
        """Called before each Module forward pass."""
        pass
    def on_module_end(self, call_id, outputs, exception):
        """Called after each Module forward pass."""
        pass

    # LM lifecycle
    def on_lm_start(self, call_id, instance, inputs):
        """Called before each LM request."""
        print(f"LM call starting: {inputs}")
    def on_lm_end(self, call_id, outputs, exception):
        """Called after each LM request."""
        if exception:
            print(f"LM call failed: {exception}")
        else:
            print(f"LM call succeeded: {outputs}")

    # Adapter lifecycle
    def on_adapter_format_start(self, call_id, instance, inputs):
        """Called before adapter format() — message construction."""
        pass
    def on_adapter_format_end(self, call_id, outputs, exception):
        pass
    def on_adapter_parse_start(self, call_id, instance, inputs):
        """Called before adapter parse() — output extraction."""
        pass
    def on_adapter_parse_end(self, call_id, outputs, exception):
        pass

    # Tool lifecycle
    def on_tool_start(self, call_id, instance, inputs):
        """Called before tool execution."""
        pass
    def on_tool_end(self, call_id, outputs, exception):
        pass

    # Evaluation lifecycle
    def on_evaluate_start(self, call_id, instance, inputs):
        """Called before Evaluate runs."""
        pass
    def on_evaluate_end(self, call_id, outputs, exception):
        pass

# Attach callback to LM
lm = dspy.LM('openai/gpt-4o-mini', callbacks=[LoggingCallback()])
dspy.configure(lm=lm)

# Or attach globally
dspy.configure(callbacks=[LoggingCallback()])
```

### MLflow Integration

```python
import mlflow
import dspy

# Full autolog API:
mlflow.dspy.autolog(
    log_compiles=True,             # track optimization process
    log_evals=True,                # track evaluation results
    log_traces_from_compile=True,  # track traces during compile
)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy-Optimization")

with mlflow.start_run():
    optimized = tp.compile(program, trainset=trainset)
    mlflow.log_metric("eval_score", score)
```

---

## Tools

### Python function tools

```python
# Any Python function can be a tool — docstring becomes description
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression safely."""
    import ast
    return eval(compile(ast.parse(expression, mode='eval'), '<string>', 'eval'))

def web_search(query: str, num_results: int = 3) -> list[str]:
    """Search the web and return relevant passages."""
    # your implementation
    ...

def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city. Units: 'celsius' or 'fahrenheit'."""
    # your implementation
    ...

# Use in ReAct
agent = dspy.ReAct(
    "question -> answer",
    tools=[calculator, web_search, get_weather],
    max_iters=10,
)

# Use in CodeAct (executes Python code)
code_agent = dspy.CodeAct("task -> result", tools=[calculator])
```

### dspy.Tool wrapper

```python
# Wrap functions explicitly
tool = dspy.Tool(
    func=my_function,
    name="search",                          # override name
    desc="Search documents for information", # override description
    args={"query": "search query"},          # override args
)

# From MCP tool (see MCP section)
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)
```

### ColBERTv2 Retriever

```python
# Built-in retriever for ColBERT indexes
retriever = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
results = retriever("What is machine learning?", k=5)
# returns: [{"text": "...", "score": 0.9}, ...]

# In a module
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.ColBERTv2(url='http://...')
        self.respond = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        passages = [r["text"] for r in self.retrieve(question, k=3)]
        return self.respond(context=passages, question=question)
```

### dspy.Embedder — standalone embedding class

```python
import dspy

# Hosted model (uses litellm, supports caching)
embedder = dspy.Embedder(
    model="openai/text-embedding-3-small",
    batch_size=200,   # default
    caching=True,     # default
)
vectors = embedder(["text 1", "text 2", "text 3"])  # returns numpy array shape (3, 1536)

# Custom embedding function
import numpy as np
def my_embed(texts: list[str]):
    return np.random.rand(len(texts), 768).astype(np.float32)  # must be 2D

embedder = dspy.Embedder(my_embed)

# Used in KNNFewShot:
from sentence_transformers import SentenceTransformer
optimizer = dspy.KNNFewShot(
    k=3, trainset=trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)
```

### ColBERTv2 Retriever (built-in)

```python
retriever = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
results = retriever("What is machine learning?", k=5)
# returns: [{"text": "...", "score": 0.9}, ...]
```

### dspy.retrievers.Embeddings — Local embeddings-based retrieval

Use any embedding model to build a local semantic search over your own corpus (FAISS-backed by default, falls back to brute-force for small corpora).

```python
import dspy

# Embed the corpus and build an index
embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(
    embedder=embedder,
    corpus=documents,      # list[str] — your document corpus
    k=5,                   # number of results to retrieve
    # brute_force_threshold=30_000  # set to avoid requiring faiss-cpu
)

# Search
results = search("semantic search query")
print(results.passages)   # list[str] of retrieved documents

# In a full RAG pipeline
class RAG(dspy.Module):
    def __init__(self, docs):
        embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
        self.search = dspy.retrievers.Embeddings(embedder=embedder, corpus=docs, k=5)
        self.respond = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        results = self.search(question)
        return self.respond(context=results.passages, question=question)
```

**Note:** Requires `pip install faiss-cpu` unless `brute_force_threshold` is set.

**Persistence for Embeddings retriever:**
```python
# Save index to disk
search.save("my_index/")

# Load from disk
search = dspy.retrievers.Embeddings.__new__(dspy.retrievers.Embeddings)
search.load("my_index/", embedder=embedder)

# Or use class method
search = dspy.retrievers.Embeddings.from_saved("my_index/", embedder=embedder)
```

### DatabricksRM — Databricks Vector Search retriever

```python
from dspy.retrievers.databricks_rm import DatabricksRM

retriever = DatabricksRM(
    databricks_index_name="my_catalog.my_schema.my_index",
    k=3,
    text_column_name="content",
    docs_id_column_name="id",
)
results = retriever("semantic search query", query_type="ANN")  # or "HYBRID"
# Returns: dspy.Prediction(docs=[...], doc_ids=[...])
```

### WeaviateRM — Weaviate vector database retriever

```python
from dspy.retrievers.weaviate_rm import WeaviateRM

retriever = WeaviateRM(
    weaviate_collection_name="MyCollection",
    weaviate_client=weaviate_client,  # v3 or v4 client
    weaviate_collection_text_key="content",
    k=3,
)
results = retriever("semantic search query")
```

### PythonInterpreter (requires Deno)

```python
from dspy.predict.python_interpreter import PythonInterpreter

with PythonInterpreter() as interp:
    result = interp("print(1 + 2)")  # returns "3"
    # Runs in isolated Deno/Pyodide sandbox
```

---

## MCP (Model Context Protocol) Integration

Connect DSPy to any MCP server (stdio or HTTP) to use its tools.

### Stdio MCP server

```python
import asyncio
import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_with_mcp():
    dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

    server_params = StdioServerParameters(
        command="python",
        args=["path/to/your/mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in tools_response.tools
            ]
            agent = dspy.ReAct("user_request -> response", tools=dspy_tools, max_iters=5)
            result = await agent.acall(user_request="Book a flight from NYC to LAX")
            print(result.response)

asyncio.run(run_with_mcp())
```

### HTTP/Streamable MCP server

```python
from mcp.client.streamable_http import streamablehttp_client

async def run_with_http_mcp():
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            dspy_tools = [dspy.Tool.from_mcp_tool(session, t) for t in response.tools]
            agent = dspy.ReAct("task -> response", tools=dspy_tools, max_iters=5)
            result = await agent.acall(task="Check weather in Tokyo")
            print(result.response)
```

---

## Inspect and Debug

```python
# Global: inspect last N LM interactions (pretty-printed)
dspy.inspect_history(n=5)

# Per-LM history — full call log with inputs, outputs, tokens, cost
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
program(question="test")

print(len(lm.history))          # number of calls
print(lm.history[-1].keys())    # model, messages, response, tokens, cost, ...
lm.inspect_history(n=3)         # pretty-print last 3 for this LM

# Inspect predictor state
for name, predictor in program.named_predictors():
    print(f"Predictor: {name}")
    print(f"  Signature: {predictor.signature}")
    print(f"  Demos: {len(predictor.demos)}")
    print(f"  LM: {predictor.lm}")

# Trace execution
with dspy.context(trace=[]):
    result = program(question="test")
    trace = dspy.settings.trace   # list of (module_name, inputs, outputs)

# Token usage per prediction (requires track_usage=True):
dspy.configure(track_usage=True)
result = program(question="test")
print(result.get_lm_usage())
# {'openai/gpt-4o-mini': {'prompt_tokens': 120, 'completion_tokens': 45}}
```

---

## Configuration Reference

```python
# Configure globally
dspy.configure(
    lm=dspy.LM('openai/gpt-4o-mini'),   # default LM
    adapter=dspy.ChatAdapter(),           # output adapter
    callbacks=[...],                      # global callbacks
    max_errors=10,                        # max errors in Evaluate
    trace=[],                             # enable tracing
    track_usage=True,                     # enable token tracking
    allow_tool_async_sync_conversion=True,# allow async tools in sync
    experimental=True,                    # BetterTogether, BootstrapFinetune
)

# Access settings
print(dspy.settings.lm)
print(dspy.settings.adapter)

# Context manager for temporary overrides
with dspy.context(lm=dspy.LM('openai/gpt-4o')):
    result = expensive_module(question="complex question")
    # reverts to gpt-4o-mini after this block

# Per-predictor LM override
module.predictor.lm = dspy.LM('openai/gpt-4o')

# Disable caching (useful during development)
lm = dspy.LM('openai/gpt-4o-mini', cache=False)

# Save and load DSPy settings (3.1.1+)
dspy.settings.save("dspy_settings.json")
dspy.settings.load("dspy_settings.json")  # also: dspy.load_settings(...)

# Configure cache explicitly
dspy.configure_cache(
    enable_disk_cache=True,     # persist cache to disk
    enable_memory_cache=True,   # in-memory cache layer
    cache_dir=None,             # custom cache directory
    disk_size_limit=30*1024**3, # disk size limit (default 30 GB)
    max_memory_entries=1_000_000, # max in-memory entries
)

# Logging control
dspy.enable_logging()                  # enable DSPy logging
dspy.disable_logging()                 # disable DSPy logging
dspy.configure_dspy_loggers()          # fine-grained logger config
dspy.enable_litellm_logging()          # enable LiteLLM debug logging
dspy.disable_litellm_logging()         # disable LiteLLM logging (default)

# Sync/async conversion utilities
async_module = dspy.asyncify(sync_module)   # wrap sync for async
sync_module = dspy.syncify(async_module)    # wrap async for sync

# Token usage tracking — context manager
with dspy.track_usage() as tracker:
    result = program(question="test")
    print(tracker.get_total_tokens())

# DummyLM for testing (no API calls)
from dspy.utils import DummyLM
dspy.configure(lm=DummyLM(["expected answer 1", "expected answer 2"]))
```

---

## Fine-tuning Workflow

```python
# 1. Compile with BootstrapFinetune
lm_to_finetune = dspy.LM('openai/gpt-4o-mini-2024-07-18', finetuning_model='gpt-4o-mini-2024-07-18')
dspy.configure(lm=lm_to_finetune)

optimizer = dspy.BootstrapFinetune(
    metric=my_metric,
    num_threads=16,
    max_bootstrapped_demos=4,
)

optimized = optimizer.compile(
    program,
    trainset=trainset,
    train_kwargs={"epochs": 1, "batch_size": 8},
)

# 2. The optimized program now uses the fine-tuned model
result = optimized(question="test question")

# 3. Save the fine-tuned program
optimized.save("finetuned_program.json")
```
