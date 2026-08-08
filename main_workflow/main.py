# Langchain imports
from langchain_core.messages import BaseMessage
from langsmith import Client

# General imports
from typing import List, Literal, Dict, Callable
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import uuid
import os

# My imports (ordered by call order)
from agents.inputRefiner.input_refiner import input_refiner_app
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
from utils.build_code import create_file
from agents.codeAnnotator.code_annotator import code_annotator_app
from agents.softwareEngineer.software_engineer import software_engineer_app
from agents.promptEngineer.prompt_engineer import prompt_engineer_app
from agents.fileHandler.file_handler import file_handler_app



load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent / '.env')

''' Constants '''
DEBUG = os.getenv('DEBUG')
BRIGHT_WHITE = '\n\033[1m\033[4m\033[97m'
RESET = '\033[0m'

def print_agent(agent_name: str) -> None:
    '''
    `print_agent` prints the name of the next agent to be called.
    
    `Args:`
        `agent_name` (str): The name of the next agent to be called.
    '''
    print(f'{BRIGHT_WHITE}[NEXT AGENT] {agent_name}{RESET}')

def print_to_file(agent_name: str, result: dict, date: str) -> None:
    '''
    `print_to_file` prints the result of the agent to a file.
    
    `Args:`
        `agent_name` (str): The name of the agent.
        `result` (dict): The result of the agent.
        `date` (str): The date of the run.
    '''
    if not DEBUG:
        return
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    
    if not os.path.exists(f'./logs/logs_{date}'):
        os.makedirs(f'./logs/logs_{date}')

    with open(f'./logs/logs_{date}/{agent_name}.txt', 'w', encoding= 'utf-8') as f:
        for key, value in result.items():
            if key == 'messages':
                f.write('Messages:\n')
                for message in value:
                    message: BaseMessage
                    f.write(f'{message.pretty_repr()}\n')
                continue
            try:
                f.write(f'{key}: {str(value)}\n\n')
            except Exception as e:
                f.write(e)

def copy_file(after_agent_name: str, file_path: str, date: str) -> None:
    '''
    `copy_file` copies the file to a new file.
    
    `Args:`
        `after_agent_name` (str): The name of the agent before this function gets called.
        `file_path` (str): The path to the file to copy.
        `date` (str): The date of the run.
    '''
    if not DEBUG:
        return
    
    with open(file_path, 'r') as f:
        contents = f.read()

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    if not os.path.exists(f'./logs/logs_{date}'):
        os.makedirs(f'./logs/logs_{date}')

    with open(f'./logs/logs_{date}/after_{after_agent_name}_{Path(file_path).name}', 'w', encoding= 'utf-8') as f:
        f.write(contents)

def main(user_request: str, orchestrator: bool= True, prompt_review_mode: Literal['llm', 'user', 'both']= 'both', coder_run_code: bool= False) -> None:
    '''
    `main` is the main function of the program.
    It invokes the input refiner, workflow refiner, code annotator, software engineer, prompt engineer and file handler agents.
    
    `Args:`
        `user_request` (str): The user request.
        `orchestrator` (bool): Whether to use the orchestrator.
        `prompt_review_mode` (Literal['llm', 'user', 'both']): The prompt review mode.
    '''
    # Config
    uuid_: str = str(uuid.uuid4())
    date: str = datetime.now().strftime('%d-%m-%y')
    config: Callable[[str], Dict] = lambda agent_name: {
        'recursion_limit': 150,
        'configurable': {
            'user_id': 'main',
            'run_name': f'main:{uuid_}',
            'thread_id': f'main:{agent_name}:{date}:{uuid_}',
        }
    }

    # Connect to langsmith
    os.environ['LANGCHAIN_PROJECT'] = 'main_workflow'
    os.environ['LANGSMITH_PROJECT'] = 'main_workflow'
    client = Client()

    # Input Refiner
    print_agent('Input Refiner (internal: Clarification  Orchestrator)')
    input_refiner_response = input_refiner_app.invoke({
        'orchestrator': orchestrator,
        'user_input': user_request
    }, config= config('input_refiner')) # corrected_original, refined_text
    print_to_file('input_refiner', input_refiner_response, date)
    clarified_user_input = input_refiner_response['refined_text']

    # Workflow Refiner
    print_agent('Workflow Refiner (internal: Clarification  Orchestrator)')
    workflow_refiner_response = workflow_refiner_app.invoke({
        'messages': [],
        'orchestrator': orchestrator,
        'clarified_user_input': clarified_user_input
    }, config= config('workflow_refiner')) # workflow
    print_to_file('workflow_refiner', workflow_refiner_response, date)
    workflow_bundle = workflow_refiner_response['workflow']

    # Create code structures
    files: List[str] = create_file(workflow_bundle)
    for file in files:
        agent_name = file.split('/')[-1].split('\\')[-1].replace('.py', '')

        copy_file('code_structure', file, date)

        # Code Annotator
        print_agent(f'Code Annotator (file: {file})')
        code_annotator_response = code_annotator_app.invoke({
            'messages': [],
            'file_path': file,
            'clarified_user_input': clarified_user_input,
            'workflow': workflow_bundle,
        }, config= config(f'code_annotator:{agent_name}'))
        print_to_file('code_annotator', code_annotator_response, date)
        copy_file('code_annotator', file, date)

        # Software Engineer
        print_agent(f'Software Engineer (file: {file}) (internal: Coder)')
        software_engineer_response = software_engineer_app.invoke({
            'messages': [],
            'file_path': file,
            'times_reviewed': 0,
            'skip_tool_sections': False, 
            'coder_run_code': coder_run_code
        }, config= config(f'software_engineer:{agent_name}'))
        print_to_file('software_engineer', software_engineer_response, date)
        copy_file('software_engineer', file, date)

        # Prompt Engineer
        print_agent(f'Prompt Engineer (file: {file})')
        prompt_engineer_response = prompt_engineer_app.invoke({
            'file_path': file,
            'mode': prompt_review_mode
        }, config= config(f'prompt_engineer:{agent_name}'))
        print_to_file('prompt_engineer', prompt_engineer_response, date)
        copy_file('prompt_engineer', file, date)

        # File Handler
        print_agent(f'File Handler (file: {file})')
        file_handler_response = file_handler_app.invoke({
            'messages': [],
            'file_path': file
        }, config= config(f'file_handler:{agent_name}'))
        print_to_file('file_handler', file_handler_response, date)
        copy_file('file_handler', file, date)



if __name__ == '__main__':
    # def test(model):
    #     from utils.utils import safe_invoke, myChatOpenAI
    #     from langchain.schema import SystemMessage
    #     from typing import TypedDict, Tuple
    #     def tool(arg1: str, arg2: int) -> str:
    #         '''`tool` is a function that takes two arguments and returns a string.'''
    #         return f'{arg1} {arg2}'
    #     class SubSchema(TypedDict):
    #         arg1: str
    #         arg2: int
    #         arg3: Tuple[str, str]
            
    #     class Schema(TypedDict):
    #         name: str
    #         age: int
    #         subSchema: SubSchema

    #     llm = myChatOpenAI(
    #         model= model
    #     ).bind_tools([tool])
    #     print(safe_invoke(llm, messages= [SystemMessage(content= 'Call the provided tool with random values')]))

    #     llm = myChatOpenAI(
    #         model= model
    #     ).with_structured_output(Schema)
    #     print(safe_invoke(llm, messages= [SystemMessage(content= 'Return the provided schema with random values')]))


    # test("deepseek/deepseek-v4-flash-0731")

    user_request: str = (
        '''Create a simple LangGraph-based AI agent that solves tasks from the GAIA benchmark.

The agent must use a single solver agent and a small workflow. Do not create separate researcher, planner, browser, calculator, or file-analysis agents.

The workflow must contain these four nodes:

1. `analyze_task`
2. `solve_task`
3. `review_answer`
4. `format_output`

## Workflow behaviour

### `analyze_task`

This node receives the original GAIA question and any attached files.

It must produce a high-level, step-by-step plan for solving the task.

The plan should:

* identify the exact objective;
* identify the required final-answer format;
* identify any attached files that need inspection;
* list the main steps needed to reach the answer;
* identify which available tools are likely to be useful;
* remain high-level and avoid performing the task itself.

The plan is guidance for the solver. The solver may adjust it when new evidence makes a step unnecessary or reveals an additional requirement.

### `solve_task`

This node executes the task.

It receives:

* the original user question;
* the attached file paths;
* the plan created by `analyze_task`;
* all previous messages and tool results;
* any feedback returned by `review_answer`;
* the current review-loop count.

The solver must work through the task by making a sequence of tool calls.

It should:

* execute one clear action at a time;
* inspect each tool result before selecting the next action;
* keep exact names, dates, values, units, and supporting evidence;
* avoid repeating tool calls that already succeeded;
* revise its approach when a tool fails;
* directly address feedback from `review_answer`;
* stop investigating when it has enough evidence to answer the question;
* call `submit_final_answer` when it has produced a supported candidate answer.

Calling `submit_final_answer` must end the current solver execution and route the workflow to `review_answer`.

The solver must have access to the following tools:

#### `file_parser`

Use this tool to inspect attached or downloaded files.

It must support all input modalities accepted by Gemini, including:

* images;
* PDF files;
* audio;
* video;
* text files;
* Word documents;
* PowerPoint presentations;
* spreadsheets;
* HTML files;
* other supported document formats.

The tool should internally call a Gemini multimodal model.

Gemini must receive:

1. a fixed system prompt describing how files should be analysed;
2. a dynamic instruction supplied by `solve_task`;
3. the relevant file or files.

The dynamic instruction should clearly state what information the solver needs from the file.

The parser must return:

* the relevant extracted information;
* structured data where useful;
* exact names, dates, numbers, and units;
* evidence locations such as page numbers, timestamps, sheet names, cell ranges, or image regions;
* any uncertainty;
* any unreadable or missing content.

The parser must not invent content and should not solve the entire GAIA task unless the solver explicitly asks it to do so.

Suggested arguments:

```python
file_parser(
    file_paths: list[str],
    instruction: str,
    task_context: str | None = None
)
```

#### `web_search`

Use this tool to discover relevant webpages and online sources.

It should return:

* result title;
* URL;
* short snippet;
* publication date when available.

Search-result snippets must not be treated as final evidence. Important information should normally be verified by opening the source.

Suggested arguments:

```python
web_search(
    query: str,
    max_results: int = 5
)
```

#### `open_url`

Use this tool to open and inspect a specific webpage.

It should accept a focused extraction instruction and return:

* the requested information;
* supporting passages;
* page title;
* final URL;
* relevant links;
* downloadable files;
* any uncertainty.

When the URL points to a file rather than a normal webpage, the tool should download the file to the task workspace and return its local path so that `file_parser` or `run_python` can inspect it.

Suggested arguments:

```python
open_url(
    url: str,
    instruction: str
)
```

#### `run_python`

Use this tool for:

* calculations;
* date and time operations;
* unit conversions;
* spreadsheet and CSV processing;
* sorting and filtering;
* structured-data analysis;
* validation of candidate answers;
* reproducible transformations.

The execution environment must:

* run in an isolated sandbox;
* have no unrestricted network access;
* enforce a time limit;
* enforce memory and output limits;
* capture standard output and errors;
* allow access only to the task workspace.

Suggested arguments:

```python
run_python(
    code: str,
    input_files: list[str] | None = None
)
```

#### `submit_final_answer`

Use this tool when the solver believes it has completed the task.

The tool should record:

* the candidate answer;
* the expected answer format;
* the steps completed;
* the supporting evidence;
* any calculations;
* any unresolved issues.

Suggested arguments:

```python
submit_final_answer(
    answer: str,
    answer_format: str,
    completed_steps: list[str],
    supporting_evidence: list[str],
    calculations: list[str],
    unresolved_issues: list[str]
)
```

Calling this tool must route the workflow directly to `review_answer`.

### `review_answer`

This node reviews the full execution history after `submit_final_answer` is called.

It must inspect:

* the original question;
* the high-level plan;
* all solver messages;
* all tool calls and tool results;
* the candidate answer;
* the evidence and calculations submitted by the solver.

It must check whether:

* every part of the question was answered;
* the solver followed or reasonably adjusted the plan;
* the evidence supports the candidate answer;
* any important search result was verified from its source;
* calculations are correct and reproducible;
* dates, names, values, units, and requested precision are correct;
* attached files were interpreted correctly;
* tool outputs contradict each other;
* the solver ignored uncertainty reported by a tool;
* the candidate answer uses the required format.

The review result must be either:

* `approve`
* `revise`

When mistakes are found, the reviewer must provide specific and actionable feedback to `solve_task`.

Example:

```text
The population value is from 2021, but the task asks for 2020. Find and verify an authoritative 2020 value, then resubmit the answer.
```

The reviewer must not provide vague feedback such as:

```text
Research the task more carefully.
```

The workflow may return from `review_answer` to `solve_task` a maximum of two times.

After two review-feedback loops, the workflow must continue to `format_output`, even when the reviewer still detects a problem.

The review counter should increase only when the reviewer returns `revise`.

### `format_output`

This node receives the full message history through the `messages` state key.

It must produce the exact answer required by the original GAIA task.

It should:

* identify the latest candidate answer;
* consider the latest reviewer result;
* follow the answer-format instructions from the original question;
* return only the information required by the task;
* remove plans, explanations, tool logs, evidence notes, and reviewer feedback unless the task explicitly requests them;
* preserve exact spelling, punctuation, units, precision, ordering, and formatting.

It must not:

* call tools;
* perform new research;
* introduce new facts;
* restart the solving process;
* expose internal reasoning.

## Routing

Use this routing structure:

```text
START
  ↓
analyze_task
  ↓
solve_task
  ↓ submit_final_answer
review_answer
  ├─ approve → format_output
  └─ revise and review_count < 2 → solve_task
  └─ revise and review_count >= 2 → format_output
  ↓
END
```

## State

Use a compact state structure based mainly on the `messages` key.

Include at least:

```python
class GAIAState(TypedDict):
    messages: Annotated[list, add_messages]

    question: str
    attachments: list[str]

    task_analysis: dict | None

    candidate_answer: str | None
    review_decision: str | None
    review_feedback: list[str]
    review_count: int

    final_output: str | None
```

All tool calls and tool results should remain available in `messages` so that the reviewer and formatter can inspect the full task history.

## General requirements

* Keep the implementation simple.
* Use one solver agent.
* Do not add unnecessary nodes or agents.
* Use structured outputs for task analysis and answer review.
* Use tool calling inside `solve_task`.
* Make the package directly runnable.
* Include all required source files, prompts, schemas, tool definitions, routing functions, and dependency declarations.
* Use clear logging for node transitions, tool calls, review decisions, and feedback-loop counts.
* Ensure that one GAIA task can be executed by passing a question and an optional list of attachment paths.
'''
    )

    main(
        user_request,
        orchestrator= True,
        prompt_review_mode= 'user',
        coder_run_code= True
    )