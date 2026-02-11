# Automated Construction of Custom AI Agents via LLM Collaboration and Prompt Engineering

## Table of contents
1. [Project Summary](#project-summary)
2. [Description](#description)
3. [System Overview](#system-overview)
4. [How the System Works](#how-the-system-works)
5. [Features](#features)
6. [Repository Structure](#repository-structure)
7. [Requirements](#requirements)
8. [Setup](#setup)
9. [Configuration](#configuration)
10. [Running](#running)
11. [Development Notes](#development-notes)
12. [Author and License](#author-and-license)

## Project Summary
- **What:** A multi-agent system where agents collaborate to construct custom AI agents from a natural language user request.
- **Who:** 
    - Author: Stefanos Panteli
    - Repository: https://github.com/stefanosPanteli/EPL-thesis

## Description
- **Purpose:** This repository implements a set of lightweight agents (each in its own folder under `agents/`) to support experimentation with agent collaboration, prompt engineering, and tool integration. A `main_workflow` entrypoint demonstrates orchestrating agents for end-to-end runs.

## System Overview
This system takes a plain-language user request and turns it into one or more runnable agents under `creations/`.

You give it a goal in natural language, such as:
- “Build an agent that answers math questions using my lecture notes.”
- “Build an agent that recommends menu items based on preferences.”

The system outputs a working code package for the requested agent, typically:
- A main Python file (the agent workflow and logic)
- A prompts file (prompt templates used by the agent)
- Any extra files the agent needs (json, DBs, etc), depending on the workflow

## How the System Works
The system runs as a pipeline of specialized agents. Each agent does one job, passes its output forward, and keeps the process structured.

### End-to-end flow (high level)
1. **Input Refiner**
    - Takes your raw request and rewrites it into a clearer, buildable spec.
    - If enabled, it uses the internal clarification orchestrator to answer the questions rather than forwarding them to the user every time.

2. **Workflow Refiner**
    - Converts the clarified request into a workflow graph.
    - The workflow is represented as a `WorkflowBundle` (root graph plus optional subgraphs).
    - This step forces the request into an execution shape that LangGraph can run.

3. **Code Scaffolding**
    - Builds the initial project structure for the new agent under `creations/`.
    - Creates the Python file(s) and prompt template file(s) with correct sections and placeholders.

4. **Code Annotator**
    - Adds structured TODOs and guidance inside the scaffold.
    - Makes the file “implementation-ready” for the Software Engineer and coder loop.

5. **Software Engineer**
    - Orchestrates coder subagents to implement missing functions.
    - Applies patches back into the file.
    - Runs a QA-style last check loop.

6. **Prompt Engineer**
    - Reviews prompt templates for common failure modes:
        - schema mismatch
        - missing placeholders
        - fragile string formatting
        - unclear output constraints
    - Fixes prompts so they are consistent and safe to use at runtime.

7. **File Handler**
    - Creates all required files used by the agent.

### What you get at the end
After a full run, you should have:
- A runnable agent implementation under `creations/<creation_name>/`
- Prompts and any helper modules required by that agent

## Features
- **Multiple agents:** `clarificationOrchestrator`, `coder`, `researcher`, `promptEngineer`, `deepResearcher`, `fileHandler`, `inputRefiner`, `softwareEngineer`, `workflowRefiner`.
- **Example agent creations:** `creations/math_assistant`, `creations/academic_query_responder`, `creations/menu_recommendation_workflow`.
- **Utility helpers:** `utils/` contains helper scripts used across agents.

## Repository Structure
```python
    # Your project directory
    CloneDir/
    │   # The folder with all implemented agents
    ├── agents/
    │   └── agentName/  # The agent folder
    │       ├── graphs/ # The agent's workflow as a graph
    │       │   └── agent_name_graph.png
    │       ├── agent_name.py   # The langchain agent
    │       ├── prompts.py      # Its prompts
    │       └── readme.md       # Its readme file
    │
    ├── creations/  # The created agents get stored here
    │   └── creationName/ # The created agent folder
    │       ├── any_other_file_created  # Files handled by the fileHandler agent
    │       ├── creation_name.py        # The langchain agent
    │       └── prompts.py              # Its prompts
    │
    ├── main_workflow/  # The main workflow folder
    │   └── main.py     # The end-to-end workflow calling the agents
    │
    ├── utils/ # Utility helpers
    │   ├── utils.py        # Helpful classes and functions used across agents
    │   └── build_code.py   # Builds the code structure of the to-be-created agent
    │
    ├── venv/   # Your virtual environment
    │   └── ...
    │
    ├── sitecustomize.py    # Customizes the Python environment
    ├── create_agent.py     # Creates a structure for a new agent under the agents folder
    ├── requirements.txt    # List of dependencies
    ├── readme.md           # This file
    │
    ├── .env                # Your API keys
    ├── .env.example        # Example API keys
    └── .gitignore          # Ignored files (.env included)
```

## Requirements
- **Python:** 3.11+ recommended.
- **Dependencies:** Install packages from `requirements.txt` via:
```bash
pip install -r requirements.txt
```
- **API keys:** Fill the required API keys in the `.env` file, by following the instructions in the `.env.example` file.
  - `.env` has the following format:
```bash
PROVIDER="A PROVIDER" # e.g., OPENROUTER, GROQ, ...
# Then you must have:
{PROVIDER}_API_KEY="Your API key"
{PROVIDER}_BASE_URL="The provider's base url"
```

## Setup
1. Create and activate a virtual environment and install dependencies:
```powershell
python -m venv venv # Creates the virtual environment, if it doesn't exist
.\venv\Scripts\Activate.ps1 # Activates the virtual environment
pip install -r requirements.txt # Installs the requirements, if not already installed
```

2. Copy `.env.example` to `.env` and fill in the required API keys and configuration values as shown in `.env.example`.

3. Set `PYTHONPATH` so Python can import project modules when running from subfolders:

```powershell
# Windows
$env:PYTHONPATH = (Get-Location)
```

```bash
# Linux
export PYTHONPATH=$PWD
```

## Configuration
- The `.env.example` file lists environment variables used by the code.
- Typical entries are API keys for LLM providers like OpenRouter, service providers such as Tavily, and optional LangSmith settings.

## Running
Run a single agent (edit the agent file to provide inputs first):

```powershell
cd .\to the clone directory # Navigate to the clone directory
.\venv\Scripts\Activate.ps1 # Activates the virtual environment

$env:PYTHONPATH = (Get-Location)    # Windows
#! OR
export PYTHONPATH=$PWD              # Linux

cd agents\<agent_name>      # Navigate to the agent folder
python <agent_name>.py      # Run the agent
```

Run the full orchestrated system from `main_workflow`:

```powershell
cd .\to the clone directory # Navigate to the clone directory
.\venv\Scripts\Activate.ps1 # Activates the virtual environment

$env:PYTHONPATH = (Get-Location)    # Windows
#! OR
export PYTHONPATH=$PWD              # Linux

cd main_workflow    # Navigate to the main workflow folder
python main.py      # Run the system end-to-end
```

## Development Notes
To add a new agent, use `create_agent.py`:
```bash
python create_agent.py <agent_name> # agent_name should be snake_case
```

## Author
- **Author**: Stefanos Panteli  
- **Repo**: https://github.com/stefanosPanteli/EPL-thesis
