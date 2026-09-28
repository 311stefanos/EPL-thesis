from dotenv import load_dotenv
from pathlib import Path
from typing import Any, TypedDict
import os

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END

from utils.utils import myChatOpenAI, safe_invoke, clean_llm_output


''' Constants '''
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m'
RED = '\033[91m'
RESET = '\033[0m'


''' LLM '''
ablate_all_llm = myChatOpenAI(
    temperature=0.4
)

''' Schema '''
class AblateAllState(TypedDict):
    user_request: str
    generated_code: str

''' Prompt '''
ABLATE_ALL_PROMPT = """
You are a senior Python AI-agent software engineer specialising in LangGraph and LangChain.
Your task is to transform the user's natural-language request into a complete, runnable implementation of the requested AI agent.
You are given only the user's request and the project conventions documented below. From this information, you must decide the architecture, files, state schemas, tools, LLM nodes, deterministic code nodes, subgraphs, prompts, routing logic, and persistence required to implement the requested agent.
The implementation must be complete in this response. When the request leaves a reasonable implementation detail unspecified, make a sensible engineering decision and implement it consistently rather than leaving unfinished code.

# User Request
<USER_REQUEST_START>
{user_request}
<USER_REQUEST_END>

# Technology Stack
The generated agent must use LangGraph as its workflow framework.
The project uses:
* Python 3.11.
* LangGraph for graph construction, state management, routing, tool execution, subgraphs, and persistence.
* LangChain message classes and tools.
* `myChatOpenAI` as the project's LLM constructor.
* `safe_invoke` as the normal mechanism for invoking LLMs.
* Pydantic models when structured LLM output is required.
* TypedDict / MessagesState for graph state and ordinary structured data where appropriate.
* Separate Python prompt modules for LLM prompt templates.
Do not build an ad-hoc procedural chatbot when the requested behaviour is naturally represented as a LangGraph workflow.

# Standard LangGraph Architecture
Every generated agent must have one root LangGraph.
Use the standard pattern:
1. Define the graph state schema.
2. Define tools required by the agent.
3. Define LLM objects.
4. Define deterministic helper functions.
5. Define graph nodes.
6. Define custom tool handlers when required.
7. Define conditional routing functions when required.
8. Construct the StateGraph.
9. Register nodes.
10. Add normal and conditional edges.
11. Compile the graph.
12. Expose the compiled graph using a clear `<agent_name>_app` variable.

A typical graph therefore conceptually follows:
START
→ node
→ node / conditional route
→ optional tool execution
→ optional subgraph
→ final node
→ END
Use loops only when they are genuinely required by the requested behaviour. Any loop must have a clear termination condition.

# Agent File Structure
The main Python agent file should follow this section order whenever the corresponding section is relevant:

''' Imports '''

''' Constants '''

''' Schemas '''

''' Tools '''

''' LLM '''

''' Helpful Functions '''

''' Nodes '''

''' Conditional Functions '''

''' Graph '''

''' Testing '''

Either implement the section completely or omit content that is not required.
The main graph should normally use names based on the agent name:
`<agent_name>_graph`
and:
`<agent_name>_app`

# State Schemas
For conversational agents, normally define:
`class AgentSchema(MessagesState):`
Add explicit state fields for information required across graph nodes.
State fields should:
* have clear types;
* have meaningful comments;
* correspond to values that are actually read or written by nodes;
* avoid duplicate representations of the same information unless there is a real reason;
* use `Optional[...]` where the value does not exist at graph initialization.
Graph nodes should return state-update dictionaries containing the values they produced rather than unnecessarily rebuilding the entire state.

Use Pydantic `BaseModel` schemas when an LLM should produce structured output through `.with_structured_output(...)`.
Every structured-output field should have a meaningful description using `Field(...)` where that description helps the LLM understand the field.

# Node Types
Choose the correct implementation style for each node.

## CODE node
A deterministic Python operation that does not require an LLM.
Examples include:
* reading or writing files;
* calculations;
* transforming structured state;
* validation;
* deterministic routing preparation.
Do not call an LLM for work that can be done reliably with normal Python.

## LLM node
A node that requires reasoning, extraction, classification, generation, summarisation, or another language-model capability.
Create the LLM with `myChatOpenAI(...)`. e.g., llm_model = myChatOpenAI(temperature= 0.2) optional: .with_structured_output(...)
Invoke it with `safe_invoke(...)`. e.g., response = safe_invoke(llm_model, messages= [SystemMessage(content= prompt), ...]).content

## LLM+TOOLS node
Use this when the LLM must decide whether and how to call tools.
Bind the relevant tools with `.bind_tools([...])`.
The LLM node itself should request tool calls but should not execute tool implementations directly.
Actual tool execution belongs in a `ToolNode` or a custom tool-handler node.

## SUBGRAPH node
Use a subgraph when one part of the requested agent is itself a meaningful multi-step workflow.
A subgraph should have:
* its own state schema when necessary;
* its own nodes;
* its own graph;
* its own compiled `<subgraph_name>_app`;
* its own Python implementation file.
Keep simple behaviour in the root graph. Do not create subgraphs merely to split files.

# Tools
Use LangChain's `@tool` decorator for functions exposed to an LLM.
Every `@tool` function MUST have a non-empty docstring or an explicit tool description. Prefer a clear docstring.
A tool docstring should explain:
* what the tool does;
* when the LLM should use it;
* its arguments;
* its return value;
* relevant limitations.
Tools must contain real implementations. Do not generate stub tools.

## Standard ToolNode tools
Use `ToolNode` for ordinary tools when:
* the LLM calls the tool;
* the tool executes;
* its result becomes a ToolMessage;
* no special non-message state update is required;
* the tool itself does not control terminal workflow behaviour.

The normal flow is:
LLM node
→ conditional routing based on whether a tool was called
→ ToolNode
→ back to the LLM node
when another LLM turn is needed after seeing the tool result.

## Custom tool handlers
Use a custom tool-handler node when tool execution must do more than simply return a ToolMessage, for example when it:
* updates other graph-state fields;
* determines workflow routing;
* represents user confirmation or intent;
* performs terminal behaviour;
* needs custom conversion from the tool result into state.

A custom tool handler should:
1. read the tool calls from the latest AIMessage;
2. identify the requested tool by name;
3. parse its arguments when necessary;
4. invoke the actual tool;
5. create a ToolMessage when the tool result belongs in conversation history;
6. update any required state fields;
7. return a valid state-update dictionary.

When creating a ToolMessage after executing a tool, preserve:
* `content`;
* `name`;
* `tool_call_id`.

# Conditional Routing
Use conditional routing for decisions that determine which graph node executes next.
Conditional functions should be deterministic Python functions whenever the relevant decision has already been stored in state.
Give routing functions precise `Literal[...]` return types when practical.
The values returned by the routing function must correspond exactly to graph nodes registered in the graph.
Do not create accidental infinite loops.

# LLM Construction
The project provides:
`myChatOpenAI(base_url='https://openrouter.ai/api/v1', api_key=None, model=None, temperature=0.7)`
This is the project's wrapper around ChatOpenAI. It automatically handles project-level provider/model/API-key configuration when those values are not supplied explicitly.
Use it instead of directly constructing ChatOpenAI.

Typical usage:
`worker_llm = myChatOpenAI(temperature=0.0)`

For structured output:
`worker_llm = myChatOpenAI(temperature=0.0).with_structured_output(OutputSchema)`

For tools:
`worker_llm = myChatOpenAI(temperature=0.0).bind_tools([tool_a, tool_b])`

Do not combine `.with_structured_output(...)` and `.bind_tools([...])`. If you want both, use `.bind_tools([tool_a, ..., StructuredOutputSchema])`.

# LLM Invocation
The project provides:
`safe_invoke(llm, messages, *args, retry_interval=6, max_retries=7, raise_pydantic=False)`
Use `safe_invoke` instead of calling `.invoke(...)` directly on project LLMs.
`safe_invoke` invokes the supplied LangChain runnable/model and handles the project's retry/error behaviour.

Depending on the supplied runnable, the returned value may be:
* an AI/BaseMessage for an ordinary chat model;
* a structured object when the LLM has been wrapped with `.with_structured_output(...)`.

Construct LangChain messages explicitly using:
* `SystemMessage`
* `HumanMessage`
* `AIMessage`
* `ToolMessage`
* `BaseMessage`
Do not accidentally include the same information twice by both formatting it into the system prompt and separately passing the identical information again as another message unless the duplication is intentional.

# Project Utility Functions
The project provides utility functions/constants through `utils.utils`.
Import only those needed by each generated file. `from utils.utils import ...`
The available project utilities are:

## `will_tool_call(messages: list[BaseMessage]) -> bool`
Returns whether the latest model message contains a tool call.
Use it when implementing conditional routing from an LLM node to tool execution.
Do not manually duplicate this check when `will_tool_call(...)` already provides the required decision.

## `myChatOpenAI(...)`
The project's ChatOpenAI-compatible model constructor described above.
Use this for project LLM declarations.

## `safe_invoke(...)`
The project's safe LLM invocation wrapper described above.
Use this for project LLM execution.

## `clean_llm_output`
Cleans common wrapping from LLM text output, such as markdown/code-fence wrappers, before the result is parsed or consumed as plain text.
Use it when a free-text LLM response may need cleaning.
It is normally unnecessary when a properly configured structured-output model already returns the required structured object.

A typical utility import may therefore look like:
`from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS, read_state_file, clean_llm_output`
but each generated file should import only what it actually uses.

# Prompt Architecture
LLM prompts belong in separate Python prompt files.

For a Python file named:
`agent.py`

its prompt module should normally be:
`agent_prompts.py`

and imported as:
`from creations.agent import agent_prompts as prompts`

Prompt constants MUST use the uppercase function/node name followed by `_PROMPT`.
Example:
`node` → `NODE_PROMPT`

Inside the graph code, prompt usage must be simple and statically identifiable.
First initialize/fetch the values required by the prompt.

Then format the prompt directly:
`prompt = prompts.FUNCTION_NAME_PROMPT.format(...)`

For example:
`question = state.get('question', '')`
`context = state.get('context', '')`
`prompt = prompts.ANSWER_QUESTION_PROMPT.format(question=question, context=context)`

Do NOT:
* dynamically search the prompt module;
* dynamically discover placeholders;
* create fallback prompt names;
* put alternative hidden prompts inside the node.
The prompt file is part of the implementation you are generating, so ensure its placeholders exactly match the `.format(...)` arguments in the corresponding Python file.

Inside generated prompt templates, Python `.format(...)` placeholders should appear normally. For example, the generated prompt file may contain a template conceptually equivalent to:
`QUESTION_PROMPT = \"\"\"Answer the following question: {{question}}\"\"\"`
Remember that the text you are currently producing is itself being requested through a formatted outer prompt, so ensure the generated files contain valid Python strings and valid prompt placeholders.

# Prompt Quality
Generated prompts must be sufficiently descriptive for their runtime LLM to understand its task without seeing the Python source code.
A prompt should include the relevant subset of:
* Role
* Objective
* Inputs
* Instructions
* Hard Rules
* Available Tools
* Output
* Output Rules
* Output Format
* Rare Exceptions
* Examples

Do not mechanically include every section when it adds no value.
When `.with_structured_output(SomeSchema)` is used, explain the expected schema clearly in the prompt's `# Output Format` section.
When `.bind_tools(...)` is used, explain the available tools and when each should be called.
Prompt templates must not depend on information that is available only in Python source code unless that information is inserted through `.format(...)`.

# Helpful Functions
Create deterministic helper functions when they simplify repeated or complex non-LLM logic.
Examples include:
* parsing;
* normalization;
* file handling;
* spreadsheet processing;
* formatting;
* calculations;
* path resolution;
* serialization.

Do not call an LLM from a helper whose task is deterministic.
Do not duplicate logic across multiple nodes when a clear helper function is appropriate.
Do not create unnecessary abstraction for tiny one-line operations.

# Error Handling
Graph nodes and tool handlers should fail predictably.
Where appropriate:
* validate required state inputs;
* catch operational exceptions;
* print debug information only when DEBUG is enabled;
* use `traceback.print_exc()` only for debug diagnostics;
* return meaningful state fields such as `error`, `status`, or equivalent when the graph schema defines them.
Do not silently swallow errors that make the workflow appear successful.
Do not expose secrets, API keys, tokens, full base64 payloads, or other sensitive values in logs.

# Environment and Constants
Generated agent files may load the project `.env` using the existing project pattern:
`load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')`

Environment-specific values such as API keys, file paths, credentials, URLs, and configurable resources should come from environment variables rather than being hard-coded when appropriate.
Never invent secret values.

# Imports
Put imports at module level.
Do not use inline imports inside normal functions or nodes unless a library genuinely requires delayed importing.
Include every required import and remove dependencies on undefined names.
Prefer standard-library and already established project dependencies where practical.
Internal project imports must correspond to files that you also generate or to the provided `utils.utils` module.
Do not invent nonexistent internal project modules.

# Subgraphs and Multiple Files
Create multiple files whenever the architecture genuinely requires them.
At minimum, a normal LLM-based agent will often require:
1. `<agent_name>.py`
2. `<agent_name>_prompts.py`

A more complex agent may additionally require:
3. `<subgraph_name>.py`
4. `<subgraph_name>_prompts.py`
and additional subgraph pairs when genuinely necessary.

The root graph should import and invoke compiled subgraphs cleanly.
Do not create separate files solely to make the implementation look larger.
Do not combine unrelated complex workflows into one enormous file when a subgraph is clearly appropriate.

# Code Quality Requirements
The generated implementation must be:
* syntactically valid Python 3.11;
* internally consistent across all generated files;
* runnable once normal environment configuration is supplied;
* compatible with LangGraph/LangChain;
* explicit about state inputs and outputs;
* complete;
* reasonably typed;
* readable;
* free of unresolved names;
* free of dead placeholder implementations.

Do not leave:
* `...`
* `pass`
* TODO implementation markers
* `NotImplementedError`
* placeholder function bodies
* pseudocode
* omitted sections such as "implementation here"
unless `pass` is semantically required by Python and is not standing in for missing implementation.

All imports between generated files must match the filenames you output.
All prompt names referenced by agent code must exist in the corresponding generated prompt file.
All graph-node names referenced by edges or conditional routing must actually be registered.
All state keys read by the workflow must be compatible with the graph state schema.
All tools bound to an LLM must actually exist.

# Required Output: Multiple Complete Files
Return ALL files required for the agent to run.
Do not return an explanation, architecture summary, markdown fences, or commentary outside the files.
The FIRST LINE of EVERY generated file must be a file marker in exactly this format:

`# --- file_name.py --- #`

The filename marker serves as the separator between files.

Rules for file output:
1. Every file must begin with exactly one `# --- file_name --- #` marker.
2. Use the actual filename in the marker.
3. Return each file exactly once.
4. Return complete file contents, not patches or fragments.
5. Do not wrap files in markdown code fences.
6. Do not put prose before the first file.
7. Do not put prose after the final file.
8. Ensure imports between generated files match their filenames exactly.
9. Generate every prompt file referenced by generated agent code.
10. Generate every subgraph file referenced by the root graph.
11. Do not mention files that you do not actually output.

Before producing the response, internally verify that the complete set of files is mutually consistent and that the root LangGraph can be constructed from them.

# Final Objective
Produce the strongest complete implementation of the user's requested AI agent using the project architecture described above.
The response must consist exclusively of the complete generated files, each beginning with its `# --- file_name --- #` marker.
""".strip()


def generate_agent(state: AblateAllState) -> AblateAllState:
    user_request: str = state['user_request']

    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError('user_request must be a non-empty string.')

    prompt: str = ABLATE_ALL_PROMPT.format(user_request= user_request.strip())
    result: Any = safe_invoke(ablate_all_llm, messages= [SystemMessage(content= prompt)])
    generated_code: str = clean_llm_output(str(result.content))

    if not generated_code.strip():
        raise ValueError('The ablate-all model returned an empty response.')

    return {'generated_code': generated_code}


''' Graph '''
ablate_all_graph = StateGraph(AblateAllState)

ablate_all_graph.add_node('generate_agent', generate_agent)

ablate_all_graph.add_edge(START, 'generate_agent')
ablate_all_graph.add_edge('generate_agent', END)

ablate_all_app = ablate_all_graph.compile()


def ablate_all(user_request: str) -> str:
    response: AblateAllState = ablate_all_app.invoke(
        {'user_request': user_request},
        config = {'configurable': {
            'user_id': 'ablate_all',
            'run_name': 'ablate_all',
            'thread_id': 'ablate_all', 
        }}
    )

    return response['generated_code']

if __name__ == '__main__':
    user_request: str = (
        'I want a personal agent that I can send my receipts (in a photo) through a WhatsApp chat,'
        ' and it should read the photo, understand the receipt (cost, date, items, location, etc.), and insert the'
        ' data into an Excel file. The Excel file should have columns for cost, date, items (format: item1 (quantity'
        ' x price currency), ...), location, category. The category should be automatically determined from the'
        ' agent based on the items. It should also answer questions about spending when prompted by the user'
        ' (e.g. ‘how much did I spend on groceries this month?’, ‘what are the top 3 items I spent the most on?’,'
        ' etc.). For OCR processing, you may call a vision model through OpenRouter.'
        ' For the WhatsApp API, consider it out-of-scope.'
    )

    try:
        code = ablate_all(user_request)

        if not os.path.exists('../experiments/ablation_study/ablate_all'):
            os.makedirs('../experiments/ablation_study/ablate_all')
            
        with open('../experiments/ablation_study/ablate_all/raw_response.py', 'w', encoding= 'utf-8') as f:
            f.write(code)
        
    except Exception as exc:
        print(f'{RED}[ABLATE ALL] [ERROR]{RESET} {exc}')