# D2T Longer Text Agent System

![System architecture](images/image.png)

This repository contains a multi agent data to text generation framework for producing longer, coherent descriptions from structured data such as triples and knowledge graph fragments.

The system implements a multi agent architecture that decomposes data to text generation into three explicit stages:

- Content ordering
- Text structuring
- Surface realization

and coordinates them with an orchestration and quality control loop built on top of large language model agents.

The framework also supports surface realization in both English and Irish and is designed for ablation studies that compare pipeline versus single agent setups and the effect of guardrails and finalization components.

---

## Key Features

- Multi agent data to text pipeline with explicit modules for:
  - Content ordering
  - Text structuring
  - Surface realization
- Orchestrator agent that:
  - Plans and routes between specialist workers
  - Tracks state across iterations
- Guardrail agent that:
  - Reviews worker outputs for fluency, faithfulness, coherence and omissions or additions
  - Feeds back rerun instructions to the orchestrator when needed
- Finalizer agent that:
  - Aggregates validated outputs into a final document
- Support for multiple LLM providers through a unified model wrapper, for example:
  - Ollama
  - OpenAI
  - Anthropic
  - Groq
  - Hugging Face
  - aiXplain
- Surface realization prompts in:
  - English
  - Irish (Gaeilge)
- LangGraph based workflow with several ablation variants:
  - Single module instead of three specialized workers
  - No guardrail
  - No finalizer
  - No orchestrator, fixed CO -> TS -> SR workflow

---

## Project Structure

The core modules in this repository include:

- `agents/llm_model.py`Unified interface over different LLM providers. Encapsulates provider specific configuration and exposes a single unified model interface and a configuration map for model names and parameters.
- `agents/workflow.py`LangGraph based definition of the overall agent workflow and ablation variants, including:

  - Default multi agent workflow with orchestrator, workers, guardrail and finalizer
  - `build_agent_workflow_single_module`
  - `build_agent_workflow_no_guardrail`
  - `build_agent_workflow_no_finalizer`
  - `build_agent_workflow_no_orchestrator`
- `agents/orchestrator.py`Orchestrator agent that:

  - Reads the current `ExecutionState`
  - Chooses the next agent to call
  - Forms prompts for the workers using recent history, user input and guardrail feedback
- `agents/worker.py`Generic worker wrapper used for content ordering, text structuring and surface realization. Each worker is configured by a role specific system prompt.
- `agents/guardrail.py`Guardrail agent that:

  - Evaluates the most recent worker output
  - Produces structured feedback and a correctness judgment
  - Signals whether to rerun or move toward finalization
- `agents/finalizer.py`Finalizer agent that compiles the history of validated outputs into a final response.
- `agents/agent_prompts.py`Prompt definitions for:

  - Content ordering
  - Text structuring
  - Surface realization in English and Irish
  - Guardrail evaluation, additions and omissions checks and revision instructions
- `agents/utilities/utils.py`Utility types and data structures, including:

  - `AgentStepOutput`
  - `ExecutionState` typed dict used as the LangGraph state
- `agents/utilities/agent_utils.py`
  Helper functions to format and summarize agent step history for use by orchestrator and guardrail prompts.

Adjust paths if the package layout changes.

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/NonsoCynthia/D2T-Longer-Text-Agent-System.git
cd D2T-Longer-Text-Agent-System
```

2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate         # on Linux or macOS
# .venv\Scripts\ctivate          # on Windows
```

3. Install dependencies

If the repository includes a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

Otherwise install the main libraries directly, for example:

```bash
pip install langgraph langchain-core langchain-classic             langchain-openai python-dotenv pydantic
```

and any provider specific SDKs you intend to use, such as:

```bash
pip install openai anthropic groq
```

---

## Configuration

Model providers are configured via environment variables. The unified model interface in `llm_model.py` expects the following keys when you use the corresponding provider:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `HF_TOKEN` or similar Hugging Face token variable
- `AIXPLAIN_API_KEY`

Create a `.env` file at the project root and add the keys you need, for example:

```env
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GROQ_API_KEY=your-groq-key
HF_TOKEN=your-huggingface-token
AIXPLAIN_API_KEY=your-aixplain-key
```

The LLM configuration (model names, temperatures) is controlled through a configuration dictionary in `llm_model.py`, for example:

```python
model_name = {
    "ollama": {"model_name": "llama3.2", "temperature": 0.0},
    "openai": {"model_name": "gpt-4.1", "temperature": 0.0},
    "anthropic": {"model_name": "claude-3-5-sonnet-latest", "temperature": 0.0},
    "groq": {"model_name": "deepseek-r1-distill-llama-70b", "temperature": 0.0},
    "hf": {"model_name": "HuggingFaceH4/zephyr-7b-beta", "temperature": 0.0},
    "aixplain": {"model_id": "640b517694bf816d35a59125", "temperature": 0.0},
}
```

You can adapt this map to your preferred models.

---

## Running the Default Workflow

The default multi agent workflow is defined in `build_agent_workflow` inside `workflow.py` (and related helpers). A minimal usage pattern looks like this:

```python
from agents.utilities.utils import ExecutionState
from agents.workflow import build_agent_workflow

provider = "openai"  # or "ollama" or another configured provider

graph = build_agent_workflow(provider=provider)

state: ExecutionState = {
    "user_prompt": "Your structured data or triple set here.",
    "iteration_count": 0,
    "max_iteration": 50,
}

result_state = graph.invoke(state)

print(result_state.get("final_response", "No final response produced."))
```

Adjust the initial `user_prompt` to your dataset format, for example a WebNLG triple set or your own knowledge graph representation.

---

## Language Choice: English vs Irish

Surface realization prompts are defined separately for English and Irish in `agent_prompts.py` and exposed through the workflow builder.

Typical options:

- English surface realizationUse the default configuration or set `language="en"` when calling ablation builders.
- Irish surface realization
  Use `SURFACE_REALIZATION_PROMPT_GA` or pass `language="ga"` to the workflow builders that accept it.

Example for an Irish workflow variant:

```python
from agents.workflow import build_agent_workflow_single_module

graph = build_agent_workflow_single_module(
    provider="openai",
    language="ga",
)
```

The input triples may remain in English or mixed language, but the surface realization agent will produce text in Irish following the dedicated prompt.

---

## Ablation Workflows

To study the contribution of each component, the repository provides several ablation builders in `workflow.py`.

### 1. Single Module Instead of Three Workers

```python
from agents.workflow import build_agent_workflow_single_module

graph = build_agent_workflow_single_module(
    provider="openai",
    language="en",  # or "ga"
)
```

Uses a single generic worker description for content ordering, text structuring and surface realization. The architecture still has orchestrator, guardrail and finalizer, but all three worker roles are instantiated with the same prompt.

### 2. No Guardrail

```python
from agents.workflow import build_agent_workflow_no_guardrail

graph = build_agent_workflow_no_guardrail(
    provider="openai",
    language="en",
)
```

Removes the LLM based guardrail module. Workers send outputs to a simple passthrough node and the orchestrator decides when to move to the finalizer based only on worker history and internal logic.

### 3. No Finalizer

```python
from agents.workflow import build_agent_workflow_no_finalizer

graph = build_agent_workflow_no_finalizer(
    provider="openai",
    language="en",
)
```

Removes the dedicated finalizer agent. The graph terminates after a no op finalizer node and the last worker output in `history_of_steps` is treated as the final answer.

### 4. No Orchestrator, Fixed CO -> TS -> SR

```python
from agents.workflow import build_agent_workflow_no_orchestrator

graph = build_agent_workflow_no_orchestrator(
    provider="openai",
    language="en",
)
```

Removes the orchestrator and replaces dynamic routing with a fixed pipeline. Execution order is:

1. Content ordering
2. Guardrail for content ordering
3. Text structuring
4. Guardrail for text structuring
5. Surface realization
6. Guardrail for surface realization
7. Finalizer

This lets you compare the dynamic agentic system against a deterministic pipeline with the same components.

---

## Experiments

For quantitative experiments, for example on WebNLG or your own triple based datasets:

1. Define a dataset loader that maps each example to a `user_prompt` string or structured payload.
2. Choose a workflow builder (default or ablation).
3. Run the graph for each instance and collect:
   - Final textual output
   - Guardrail reviews
   - Agent step history
4. Evaluate generated texts using standard data to text metrics or human evaluation.

---

## Citation

If you use this repository in academic work, please consider citing the corresponding paper once it is available. A generic BibTeX entry could look like:

```bibtex
TBA
```

Update this with the final citation details for your publication.

---

## License

Specify your chosen license here, for example:

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions, issues or collaboration requests, please open a GitHub issue or contact:

- Chinonso Cynthia Osuji
- {firstname.lastname}@adaptcentre.ie
