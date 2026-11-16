# From Natural Language Requests to Runnable AI Agents:<br>A Multi-Agent Builder for Custom AI Agents (MABCA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**MABCA (Multi-Agent Builder for Custom AI Agents)** is a human-supervised multi-agent framework that transforms natural-language requests into runnable and inspectable source-level AI agent packages.

The framework combines requirement refinement, workflow design, deterministic code scaffolding, structured code annotation, function-level implementation, optional execution-based testing, prompt engineering, and supporting-file generation.

Each construction stage produces explicit intermediate artifacts that can be inspected, reviewed, and revised.

The resulting agents are independently executable LangGraph applications that can be modified and deployed without requiring the MABCA construction framework, once their dependencies and credentials have been configured.

---

## Contents

- [System overview](#system-overview)
- [Architecture](#architecture)
- [Core generation pipeline](#core-generation-pipeline)
- [Supporting agents](#supporting-agents)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the system](#running-the-system)
- [Generated output](#generated-output)
- [Isolated code testing](#isolated-code-testing)
- [Experiments and evaluation](#experiments-and-evaluation)
  - [Models used in the experiments](#models-used-in-the-experiments)
  - [Benchmark pipelines](#benchmark-pipelines)
- [Benchmark results included in the repository](#benchmark-results-included-in-the-repository)
- [Reproducing the benchmarks](#reproducing-the-benchmarks)
  - [Benchmark-specific adaptations](#benchmark-specific-adaptations)
  - [HumanEval / HumanEval+](#humaneval--humaneval)
  - [MBPP / MBPP+](#mbpp--mbpp)
  - [IFEval](#ifeval)
  - [GAIA](#gaia)
- [Ablation study](#ablation-study)
  - [Experimental methodology](#experimental-methodology)
  - [Main workflow ablations](#main-workflow-ablations)
  - [Main findings](#main-findings)
- [Construction cost and runtime](#construction-cost-and-runtime)
- [Development](#development)
- [Reproducibility notes and limitations](#reproducibility-notes-and-limitations)
- [Author](#author)
- [License](#license)

---

## System overview

MABCA accepts a user request such as:

> Build an agent that receives receipt images, extracts the purchase information, stores the data in an Excel file, and answers questions about the user's spending.

The system progressively converts that request into executable source code. It first clarifies what should be built, then defines an explicit workflow graph, creates the initial code structure, fills in implementation details, tests generated functions when runtime testing is enabled, writes and validates prompts, and creates any additional files required by the generated agent.

A typical run produces one or more Python agents under `creations/<agent_name>/`, together with prompt modules and any supporting files required by the generated workflow.

### Design goals

The system is designed around four goals:

1. **Structured decomposition**: separate requirement engineering, workflow design, implementation, testing, and prompt construction into specialized stages.
2. **Executable output**: produce Python code that can be run and further inspected.
3. **Feedback and verification**: allow human review and model-based review at multiple stages, with optional runtime validation inside Docker containers.
4. **Reproducible evaluation**: keep generated benchmark agents, benchmark runners, outputs, and ablation variants in the repository.

---

## Architecture

![Top-level architecture of MABCA](docs/top_level_view.png)

The main workflow is implemented in `main_workflow/main.py`. It invokes the major stages in sequence and passes structured state between them.

---

## Core generation pipeline

| Stage | Component | Responsibility |
|---|---|---|
| 1 | **Input Refiner** | Corrects and clarifies the raw request, resolves missing details, and produces a more precise implementation-ready specification. |
| 2 | **Workflow Refiner** | Converts the refined request into a structured `WorkflowBundle` containing a root workflow graph and optional subgraphs. |
| 3 | **Code Scaffolding** | Deterministically converts the workflow description into LangGraph-oriented Python skeletons and prompt modules under `creations/`. Each graph is converted into a single Python file. The following agents are invoked in sequence for each file. |
| 4 | **Code Annotator** | Expands the scaffold with node documentation, schemas, helper-function stubs, tool stubs, and LLM configuration proposals. Adds design principles and implementation guidance. |
| 5 | **Software Engineer** | Coordinates function-level implementation, reviews Coder proposals, applies approved changes, and performs final source-code review. |
| 6 | **Coder** | Implements individual functions and proposes required imports, helper functions, or tools. |
| 7 | **Code Tester (optional)** | Generates representative inputs, executes candidate implementations in isolated Docker containers, and provides execution-informed review. |
| 8 | **Prompt Engineer** | Generates, reviews, formats, tests, and writes prompt templates used by LLM nodes. |
| 9 | **File Handler** | Creates additional files and directories required by the generated application. |

### Workflow representation

The Workflow Refiner emits a structured bundle with:

- one root `WorkflowGraph`
- zero or more subgraphs
- named nodes
- directed edges
- graph-level memory configuration
- node descriptions that specify expected execution type, such as code, LLM, LLM with tools, or subgraph execution

The scaffold generator in `utils/build_code.py` converts this representation into Python files containing LangGraph state, nodes, conditional routing, graph construction, and prompt references.

---

## Supporting agents

The repository currently contains the following specialized agents under `agents/`:

| Agent | Role |
|---|---|
| `clarificationOrchestrator` | Answers clarification questions from previously supplied context when possible. |
| `inputRefiner` | Refines the user's initial requirements. |
| `workflowRefiner` | Designs the executable workflow structure. |
| `codeAnnotator` | Enriches generated scaffolds before implementation. |
| `softwareEngineer` | Coordinates implementation and review. |
| `coder` | Implements individual functions. |
| `codeTester` | Executes generated function implementations in isolated Docker containers and reviews the results. |
| `promptEngineer` | Creates and tests prompt templates. |
| `fileHandler` | Creates supporting files and folders. |
| `researcher` | Performs focused web research and returns a consolidated result. |
| `deepResearcher` | Splits larger research tasks into smaller questions and combines Researcher outputs. |

The agent folders also include their own `readme.md` with the component's state schema, graph structure, tools, behavior, and direct invocation examples.

---

## Repository structure

```text
.
├── agents/
│   ├── clarificationOrchestrator/
│   │   ├── graphs/
│   │   │   └── clarification_orchestrator_app.png
│   │   ├── clarification_orchestrator.py
│   │   ├── prompts.py
│   │   └── readme.md
│   ├── codeAnnotator/
│   │   └── ...
│   ├── codeTester/
│   │   ├── ...
│   │   ├── code_tester.py
│   │   ├── isolated_execution.py
│   │   └── Dockerfile.code-tester
│   ├── coder/
│   │   └── ...
│   ├── deepResearcher/
│   │   └── ...
│   ├── fileHandler/
│   │   └── ...
│   ├── inputRefiner/
│   │   └── ...
│   ├── promptEngineer/
│   │   └── ...
│   ├── researcher/
│   │   └── ...
│   ├── softwareEngineer/
│   │   └── ...
│   └── workflowRefiner/
│       └── ...
│
├── creations/
│   ├── math_assistant/
│   └── menu_recommendation_workflow/
│
├── experiments/
│   ├── ablation_study/
│   │   ├── ablate_all/
│   │   ├── no_requirement_engineering/
│   │   ├── no_design/
│   │   ├── no_implementation/
│   │   ├── no_feedback/
│   │   └── full_framework/
│   └── benchmarks/
│       ├── problem_solution_pipeline/
│       ├── ifeval_solver_pipeline/
│       └── gaia_task_solver/
│
├── main_workflow/
│   ├── ablate_all.py
│   ├── ablate_req_eng.py
│   ├── ablate_design.py
│   ├── ablate_implementation.py
│   └── main.py
│
├── utils/
│   ├── build_code.py
│   └── utils.py
│
├── create_agent.py
├── requirements.txt
├── sitecustomize.py
├── .env.example
└── README.md
```

Generated files, execution logs, benchmark outputs, and experimental artifacts may add further directories beneath these paths.

---

## Requirements

### Base system

- **Python 3.11** is recommended. The isolated code-testing image also uses Python 3.11.
- An **OpenAI-compatible chat model provider** supported through `langchain-openai`.
- API credentials for the selected model provider.
- **Tavily API access** for components that use Tavily search.
- **LangSmith** is used for tracing in the current workflow configuration.

The base Python dependencies are pinned in `requirements.txt`, including:

- LangGraph `0.6.3`
- LangChain `0.3.27`
- LangChain OpenAI `0.3.28`
- LangChain Tavily `0.2.11`
- LangSmith `0.4.10`

### Docker

Docker is required when:

- `coder_run_code=True` is used in the main generation workflow
- the Code Tester is invoked directly
- benchmark runners use Docker-backed execution or evaluation
- GAIA executes generated Python code in its isolated runtime

---

## Installation

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/stefanosPanteli/EPL-thesis.git
cd EPL-thesis
python -m venv venv
```

Activate the environment.

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location)
```

### Linux or macOS

```bash
source venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

Copy the environment template:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then configure the required credentials in `.env`.

---

## Configuration

The model wrapper in `utils/utils.py` reads the provider, base URL, API key, and model name from environment variables.

A minimal configuration using OpenRouter is:

```dotenv
PROVIDER='OPENROUTER'
OPENROUTER_API_KEY='your-api-key'
OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'
MODEL_NAME='your-model-name'

TAVILY_API_KEY='your-tavily-key'

LANGCHAIN_API_KEY='your-langsmith-key'
LANGSMITH_API_KEY='your-langsmith-key'
LANGCHAIN_TRACING_V2=true
LANGSMITH_TRACING=true

DEBUG=1
```

Other OpenAI-compatible providers can be selected by setting:

```dotenv
PROVIDER='PROVIDER_NAME'
PROVIDER_NAME_API_KEY='...'
PROVIDER_NAME_BASE_URL='...'
MODEL_NAME='...'
```

`myChatOpenAI` automatically resolves `{PROVIDER}_API_KEY`, `{PROVIDER}_BASE_URL`, and `MODEL_NAME`.

### Debug mode

Set:

```dotenv
DEBUG=1
```

to print node-level execution information and enable workflow logging. When debug logging is active, the main workflow stores intermediate state and generated source snapshots under its `logs/<run_name>/` directory.

---

## Running the system

### Important execution-directory requirement

`utils/build_code.py` currently writes generated agents to `../creations/` using a relative path. Run the main workflow **from the `main_workflow/` directory** so generated files are written to the repository's `creations/` directory.

From the repository root:

### Windows PowerShell

```powershell
$env:PYTHONPATH = (Get-Location)
cd main_workflow
python main.py run_name # only for logging
```

### Linux or macOS

```bash
export PYTHONPATH="$PWD"
cd main_workflow
python main.py run_name # only for logging
```

The positional argument is the run name used for debug logs.

### CLI behavior

The `main.py` command-line entry point runs the receipt-management example request embedded in the file.

For a custom request, just change the `user_request` variable in the `main.py` file.

```python
user_request = """
Build an agent that answers questions over a collection of academic notes, keeps conversational memory, and cites the source file used for each answer.
"""

main(
    user_request= user_request,
    orchestrator= True or False,
    prompt_review_mode= "llm" or "user" or "both",
    coder_run_code= True or False,
)
```

### Main workflow options

| Argument | Meaning |
|---|---|
| `user_request` | Natural-language description of the agent to build. |
| `orchestrator` | When `True`, internal clarification can reuse prior context through the Clarification Orchestrator. |
| `prompt_review_mode` | `"llm"`, `"user"`, or `"both"`. Controls Prompt Engineer review. `"user"` and `"both"` are interactive. |
| `coder_run_code` | When `True`, Coder proposals can be checked by the Code Tester through isolated execution. |
| `run_name` | Directory name used for debug logs. |

Several pipeline stages support interactive confirmation. For unattended runs you can just press enter.

---

## Generated output

For a root workflow named `example_agent`, the scaffold generator creates a structure similar to:

```text
creations/
└── example_agent/
    ├── example_agent.py
    ├── example_agent_prompts.py
    ├── optional_subgraph.py
    ├── optional_subgraph_prompts.py
    └── additional_files_created_by_file_handler
```

Generated Python files contain:

- state schemas
- tool definitions
- LLM instances
- helper functions
- LangGraph nodes
- conditional routing functions
- graph construction
- optional `MemorySaver` checkpoints
- a local testing section

The Prompt Engineer writes prompt constants into the corresponding `*_prompts.py` modules.

---

## Isolated code testing

The Code Tester adds runtime evidence to the implementation loop. It receives an implementation of one function, generates meaningful keyword-argument inputs, executes each input separately, and reviews the implementation together with outputs, exceptions, tracebacks, and execution times.

### Build the Code Tester image

From the repository root:

```bash
docker build \
  -f agents/codeTester/Dockerfile.code-tester \
  -t thesis-code-tester:latest \
  .
```

PowerShell equivalent:

```powershell
docker build `
  -f agents\codeTester\Dockerfile.code-tester `
  -t thesis-code-tester:latest `
  .
```

### Isolation model

The Code Tester:

- checks that Docker is available
- builds candidate source from the original file, required imports, and the proposed function implementation
- executes each generated input in a separate container
- captures return values, errors, tracebacks, output types, and durations
- continues testing later inputs if one input fails
- returns the execution report to a reviewer model

This mechanism is intended to reduce the risk of accepting code based only on static model review. It is still a research sandbox and should not be treated as a complete security boundary for hostile code.

---

## Experiments and evaluation

The repository contains two main forms of evaluation:

1. **Generated-agent benchmarks** under `experiments/benchmarks/`
2. **Framework ablations** under `experiments/ablation_study/` and `main_workflow/ablate_*.py`

The benchmark directories include the generated benchmark agents, runners, run configurations, raw generations, evaluation outputs, and score summaries used for analysis.

### Models used in the experiments

The experimental evaluation distinguishes between the models
used by the generated agents during benchmark execution and
the backbone model used by MABCA during the controlled
ablation study.

| Experiment | Model |
|---|---|
| HumanEval / HumanEval+ | NVIDIA Nemotron 3 Ultra 550B A55B |
| MBPP / MBPP+ | NVIDIA Nemotron 3 Ultra 550B A55B |
| IFEval | NVIDIA Nemotron 3 Ultra 550B A55B |
| GAIA | DeepSeek V4 Flash 0731 |
| Ablation study (agent construction) | Z.ai GLM 5.3 |
| Full-framework reference run | Z.ai GLM 5.3 |

The benchmark agents used models with one-million-token
context windows, which was one of the factors considered
during model selection.

MABCA is not restricted to these models. Other compatible
models can be configured through the environment variables
described in the Configuration section.

### Benchmark pipelines

| Benchmark | Generated agent / runner | Purpose |
|---|---|---|
| HumanEval / HumanEval+ | `problem_solution_pipeline` | Evaluates generated Python solutions using EvalPlus base and extended tests. |
| MBPP / MBPP+ | `problem_solution_pipeline` | Evaluates generated MBPP solutions using EvalPlus base and extended tests. |
| IFEval | `ifeval_solver_pipeline` | Evaluates instruction-following under strict and loose official IFEval criteria. |
| GAIA | `gaia_task_solver` | Evaluates tool-using question answering on the GAIA validation split. |

---

## Benchmark results included in the repository

The following results were obtained by agents generated
using MABCA.

| Benchmark | Model | Metric | Result |
|---|---|---|---:|
| HumanEval | Nemotron 3 Ultra 550B A55B | Pass@1 | 98.17% |
| HumanEval+ | Nemotron 3 Ultra 550B A55B | Pass@1 | 91.46% |
| MBPP | Nemotron 3 Ultra 550B A55B | Pass@1 | 99.47% |
| MBPP+ | Nemotron 3 Ultra 550B A55B | Pass@1 | 83.33% |
| IFEval | Nemotron 3 Ultra 550B A55B | Strict prompt-level | 90.39% |
| IFEval | Nemotron 3 Ultra 550B A55B | Strict instruction-level | 92.81% |
| IFEval | Nemotron 3 Ultra 550B A55B | Loose prompt-level | 91.87% |
| IFEval | Nemotron 3 Ultra 550B A55B | Loose instruction-level | 94.00% |
| GAIA | DeepSeek V4 Flash 0731 | Overall accuracy | 80.61% |
| GAIA Level 1 | DeepSeek V4 Flash 0731 | Accuracy | 88.68% |
| GAIA Level 2 | DeepSeek V4 Flash 0731 | Accuracy | 77.91% |
| GAIA Level 3 | DeepSeek V4 Flash 0731 | Accuracy | 73.08% |

For the code-generation benchmarks, pass@1 refers to the
final submitted solution for each task. The generated agent
may perform multiple internal model calls before submitting
its solution.

---

## Reproducing the benchmarks

### Benchmark-specific adaptations

Before evaluation, limited non-task-specific adaptations
were made to the execution tools of the generated agents:

- Code-solving agent: Python execution was moved into an
  isolated Docker environment.
- GAIA agent: Python execution was moved into an isolated
  Docker environment, and a MIME-handling issue in the
  file-parsing tool was corrected to support multimodal inputs.
- IFEval agent: No code modifications were required.

No task-specific solution logic was manually added or modified
during these adaptations.

### HumanEval / HumanEval+

Install EvalPlus:

```bash
pip install --upgrade evalplus
```

Run from `experiments/benchmarks/problem_solution_pipeline/`:

```bash
python run_humaneval_plus_benchmark.py \
  --output-dir ./benchmark_results/humaneval/humaneval_plus_full \
  --evaluation-mode docker
```

A short smoke test can be run with:

```bash
python run_humaneval_plus_benchmark.py \
  --limit 3 \
  --output-dir ./benchmark_results/humaneval/humaneval_plus_test \
  --evaluation-mode none
```

### MBPP / MBPP+

With EvalPlus installed, run:

```bash
python run_mbpp_plus_benchmark.py \
  --output-dir ./benchmark_results/mbpp/mbpp_plus_full \
  --evaluation-mode docker
```

Smoke test:

```bash
python run_mbpp_plus_benchmark.py \
  --limit 3 \
  --output-dir ./benchmark_results/mbpp/mbpp_plus_test \
  --evaluation-mode docker
```

### IFEval

Install the IFEval evaluator dependencies:

```bash
pip install "lm_eval[ifeval]"
```

Run from `experiments/benchmarks/ifeval_solver_pipeline/`:

```bash
python run_ifeval_benchmark.py \
  --output-dir ./benchmark_results/ifeval/ifeval_full \
  --max-parallel-tasks 2 \
  --task-timeout-seconds 300
```

Smoke test:

```bash
python run_ifeval_benchmark.py \
  --limit 3 \
  --output-dir ./benchmark_results/ifeval/ifeval_test
```

### GAIA

Install the dataset dependencies:

```bash
pip install -U datasets huggingface_hub
```

GAIA is distributed through a gated Hugging Face dataset. Accept the dataset conditions and authenticate with Hugging Face before running the full evaluation.

Run from `experiments/benchmarks/gaia_task_solver/`:

```bash
python run_gaia_task_solver.py \
  --output-dir ./benchmark_results/gaia_task_solver/gaia_validation_full \
  --task-timeout 300
```

Smoke test:

```bash
python run_gaia_task_solver.py \
  --limit 3 \
  --output-dir ./benchmark_results/gaia_task_solver/gaia_validation_test
```

All benchmark runners support additional options for subsets, task selection, resuming interrupted runs, timeouts, and output directories. Run the relevant script with `--help` for the complete interface.

---

## Ablation study

The repository includes ablated workflow variants used to isolate the contribution of major system stages.

### Experimental methodology

The ablation study evaluates the contribution of individual construction mechanisms by generating a receipt-management agent under six configurations.

All configurations received the same initial natural-language request and used Z.ai GLM 5.3 as the common construction backbone model.

| ID | Configuration | Modification |
|---|---|---|
| 1 | One-shot baseline | Replaces the staged process with a single model invocation. |
| 2 | No Requirement Refinement | Bypasses the Input Refiner and Workflow Refiner, generating the WorkflowBundle through a single invocation. |
| 3 | No Code Annotation | Replaces the Code Annotator with a single model invocation. |
| 4 | No Implementation | Replaces the iterative implementation stage with a single model invocation. |
| 5 | No User Feedback | Removes user feedback and approval while retaining automated construction and review mechanisms. |
| 6 | Full Framework | Uses the complete MABCA construction process. |

The generated agents were evaluated using the same five
receipt images and a common set of conversational queries.

Evaluation considered both runtime behavior and source-code
quality, including workflow routing, tool execution, persistent
data, readability, and maintainability.

### Main workflow ablations

| Script | Main change |
|---|---|
| `main_workflow/ablate_req_eng.py` | Removes the full requirement-engineering path and uses the ablated workflow-refinement configuration. |
| `main_workflow/ablate_design.py` | Replaces the full Code Annotator with its ablated variant. |
| `main_workflow/ablate_implementation.py` | Replaces the full Software Engineer with its ablated implementation variant. |
| `main_workflow/ablate_all.py` | Uses a single model-driven generation path instead of the full staged framework. |

Additional ablated component implementations are stored beside the full agents, including variants of the Input Refiner, Workflow Refiner, Code Annotator, Software Engineer, Coder, and Prompt Engineer.

The generated applications and execution artifacts used for the ablation study are stored under:

```text
experiments/ablation_study/
├── full_framework/
├── no_requirement_engineering/
├── no_design/
├── no_implementation/
├── no_feedback/
└── ablate_all/
```

### Main findings

- Requirement refinement primarily improved behavioral completeness and generalization.
- Code annotation improved software structure, typing, and maintainability.
- Iterative implementation helped identify interface and parsing defects.
- Human feedback helped resolve undesirable but technically plausible design decisions.
- The full framework produced the most coherent implementation and the broadest behavioral coverage in the evaluated task.

All ablated configurations produced broadly functional agents.

---

## Construction cost and runtime

The full-framework receipt-management agent construction
was recorded as an end-to-end reference run.

| Metric | Recorded value |
|---|---:|
| Backbone model | Z.ai GLM 5.3 |
| Total runtime | 6,216.1 seconds |
| User interaction time | 1,048 seconds |
| Input tokens | 2.135 million |
| Output tokens | 771,000 |
| Model calls | 95 |
| Estimated API cost (GLM 5.3) | $3.00 |

The Software Engineer was the most computationally intensive
stage, accounting for approximately 54% of the total runtime.

Based on the recorded token workload and an assumed 30%
input cache-hit rate, the estimated API cost ranged from
$0.44 to $30.92 across the model-pricing scenarios considered
in the paper.

These estimates apply alternative model prices to the same
recorded workload. They do not represent additional
construction experiments using those models.

---

## Development

### Create a new internal agent scaffold

The repository includes a small helper for creating a new agent directory:

```bash
python create_agent.py <agent_name>
```

The generated scaffold contains the new agent module, a prompt module, a graph output directory, standard imports, model setup, and placeholders for schemas, tools, nodes, and graph construction.

### Component-level execution

Each major agent exposes a compiled LangGraph app, for example:

```python
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
```

The component-level `readme.md` files document the expected input state and output schema for direct use.

---

## Reproducibility notes and limitations

### Model dependence

Results depend on the configured provider and `MODEL_NAME`. Hosted models can change over time even when the model identifier remains constant.

### Interactive stages

The full generation workflow contains human-confirmation paths. In particular, requirement refinement, workflow refinement, code annotation, and some prompt-review modes can request terminal input. This is useful for human-in-the-loop generation.

### Relative paths

The current scaffold generator uses relative output paths. Run `main.py` from the `main_workflow/` directory unless the path handling is changed.

### External services

Some agents rely on external model APIs, Tavily, LangSmith, web access, or benchmark datasets. Reproduction therefore requires the corresponding credentials and service availability.

### Runtime testing

Docker-based execution reduces direct exposure of the host environment and provides useful dynamic evidence, but it is not presented as a hardened sandbox for adversarial code.

### Generated software

MABCA produces executable code automatically. Generated applications can still contain logic errors, insecure operations, inappropriate dependencies, or incorrect assumptions. Review generated code before using it with sensitive data or deploying it outside an experimental environment.

### Construction variability

Generated agents may differ across repeated executions because of model behavior, provider configurations, user clarifications, and approval decisions.

The benchmark results correspond to the particular generated agents included in this repository. Regenerating an agent from the same initial request does not guarantee an identical implementation or benchmark score.

### Intervention during active construction

Manual intervention during an active construction stage is limited. In particular, the Software Engineer cannot currently be paused, manually corrected, and resumed from the modified source-code state.

Manual corrections should therefore be applied after the active construction stage has completed.

---

## Author

**Stefanos Panteli**

Repository: <https://github.com/311stefanos/EPL-thesis>

## License

The original MABCA source code is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

Third-party dependencies, benchmark datasets, and external models
are subject to their respective licenses and terms of use.