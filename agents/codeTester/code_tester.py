r"""
- `author:` Stefanos Panteli
- `date:` 2026-07-24
- `description:` Generates test inputs for a Python function, executes the proposed implementation in an isolated Docker environment, and reviews the implementation together with the execution results.

## How to use
1. Import the app. (`from agents.codeTester.code_tester import code_tester_app`)
2. Input a dict with the following keys:
    - `function_name: str`: The exact name of the function to test.
    - `file_path: str`: The path to the Python file containing the original source code and surrounding context.
    - `implementation: str`: The latest implementation of the target function returned by the Coder.
    - `imports: Optional[List[str]]`: Additional import statements required by the proposed implementation. May be `None` or an empty list.
    - `se_instructions: str`: The instructions originally given to the Coder by the Software Engineer.
3. Invoke the app.
4. Get the output dict with the following key:
    - `reviewer_comments: str`: Returns `yes` when no material issue requires another Coder revision. Otherwise, returns a numbered list describing the identified issues and required corrections.

## Usage
```python
from agents.codeTester.code_tester import code_tester_app

graph_input = {
    'function_name': 'split_prompt',
    'file_path': r'C:\UCY\EPL-thesis\EPL-thesis\Clone\agents\promptEngineer\prompt_engineer.py',
    'implementation': '''<implemented code>''',
    'imports': [],
    'se_instructions': (
        'Implement split_prompt so it separates an LLM response into the thinking process, '
        'the proposed prompt, and a list of old-code/new-code change tuples. It must support '
        'responses with no code changes, clean every returned section using clean_llm_output, '
        'preserve the order of code changes, and exclude changes where the cleaned old and new '
        'code are identical.'
    )
}

response = code_tester_app.invoke(graph_input)

# response = {
#     'reviewer_comments': 'yes'
# }
"""
''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode

# Schema imports
from typing import TypedDict, Literal, List, Optional, Annotated, Dict, Any
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, read_state_file
from agents.codeTester.isolated_execution import run_in_isolated_env
from agents.codeTester import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'

CODE_TESTER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_TESTER_DIR.parent.parent
UTILS_PATH = PROJECT_ROOT / 'utils'
AGENTS_PATH = PROJECT_ROOT / 'agents'
CREATIONS_PATH = PROJECT_ROOT / 'creations'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Code Tester') if DEBUG else None



""" Schemas """
''' General Schemas '''
class FunctionInputs(BaseModel):
    kwargs: List[Dict[str, Any]] = Field(
        description= 'A list of test inputs, in kwargs format.',
        default_factory= list
    )
    pip_packages: List[str] = Field(
        description= 'External PyPI packages required to execute the source code.',
        default_factory= list
    )

class FunctionExecutionResult(BaseModel):
    kwargs: Dict[str, Any] = Field(default_factory=dict)
    completed: bool
    output: Optional[Any] = None
    output_type: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    execution_time_seconds: Optional[float] = None

    def __str__(self):
        arguments = '\n'.join([f'{key}= {value}' for key, value in self.kwargs.items()])
        if not arguments:
            arguments = 'None'

        execution_time = (
            f'{self.execution_time_seconds:.6f} seconds'
            if self.execution_time_seconds is not None
            else 'Not available'
        )

        if self.completed:
            return (
                f'Input Arguments:\n{arguments}\n'
                f'Status: Completed\n'
                f'Output: {self.output!r}\n'
                f'Output Type: {self.output_type}\n'
                f'Execution Time: {execution_time}\n'
            )

        return (
            f'Input Arguments:\n{arguments}\n'
            f'Status: Failed\n'
            f'Error Type: {self.error_type or "Unknown"}\n'
            f'Error Message: {self.error_message or "No error message returned."}\n'
            f'Traceback:\n{self.traceback or "No traceback returned."}\n'
            f'Execution Time: {execution_time}\n'
        )

class CodeExecutionReport(BaseModel):
    function_name: str
    total_inputs: int
    completed_executions: int
    failed_executions: int
    results: List[FunctionExecutionResult]

    def __str__(self):
        def _format_results(results: List[str]) -> str:
            if not results: 
                return '\tNone'

            formatted_results = []
            for index, result in enumerate(results,start=1):
                indented_result = '\n'.join(f'\t{line}' for line in result.strip().splitlines())
                formatted_results.append(f'\tExecution {index}:\n{indented_result}')

            return '\n\n'.join(formatted_results)
    
        completed = _format_results([str(result) for result in self.results if result.completed])
        failed = _format_results([str(result) for result in self.results if not result.completed])

        return (
            f'Function Name: {self.function_name}\n'
            f'Total Inputs: {self.total_inputs}\n'
            f'Completed Executions: {self.completed_executions}\n'
            f'Failed Executions: {self.failed_executions}\n'
            f'\nCompleted Results:\n{completed}\n'
            f'\nFailed Results:\n{failed}\n'
        )

''' Input Schema '''
class CodeTesterInput(BaseModel):
    function_name: str
    file_path: str
    implementation: str
    imports: Optional[List[str]]
    se_instructions: str

''' Intermediate Schema '''
class CodeTesterIntermediate(BaseModel):
    function_name: str
    file_path: str
    implementation: str
    imports: Optional[List[str]] = None
    se_instructions: str

    function_inputs: Optional[FunctionInputs] = None
    function_execution_report: Optional[CodeExecutionReport] = None

    execution_error: Optional[str] = None

    def has_a_report(self) -> bool:
        return self.function_execution_report is not None

''' Output Schema '''
class CodeTesterOutput(BaseModel):
    reviewer_comments: str



''' LLM '''
input_generator = myChatOpenAI(
    temperature= 0.7
).with_structured_output(FunctionInputs)#, method='function_calling')

reviewer = myChatOpenAI(
    temperature= 0.3
)



''' Nodes '''
def generate_inputs(state: CodeTesterInput) -> CodeTesterIntermediate:
    '''
    This node generates the inputs for the test cases.
    '''
    print_function_name()

    try:
        prompt = prompts.GENERATE_INPUTS_PROMPT.format(
            function_name= state.function_name,
            code= read_state_file(state),
            imports= state.imports,
            implementation= state.implementation,
            special_instructions= state.se_instructions
        )

        # call the LLM
        response: FunctionInputs = safe_invoke(input_generator, messages= [SystemMessage(content= prompt)])
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        return CodeTesterIntermediate(
            function_name= state.function_name,
            file_path= state.file_path,
            implementation= state.implementation,
            imports= state.imports,
            se_instructions= state.se_instructions,
            function_inputs= response,
            function_execution_report= None
        )
    
    except Exception as e:
        print(f'\n{RED}[AGENT] [ERROR] [GENERATE INPUTS]{RESET} {e}')
        traceback.print_exc()
        return CodeTesterIntermediate(
            function_name= state.function_name,
            file_path= state.file_path,
            implementation= state.implementation,
            imports= state.imports,
            se_instructions= state.se_instructions,
            function_inputs= None,
            function_execution_report= None
        )

def run_code(state: CodeTesterIntermediate) -> CodeTesterIntermediate:
    '''
    Executes the latest implementation with every generated input.
    '''
    print_function_name() if DEBUG else None

    try:
        report_data = run_in_isolated_env(
            function_name= state.function_name,
            source_code= read_state_file(state),
            source_file_path= state.file_path,
            implementation= state.implementation,
            imports= state.imports,
            function_inputs= state.function_inputs.kwargs,
            pip_packages= state.function_inputs.pip_packages,
            utils_path= str(UTILS_PATH),
            agents_path= str(AGENTS_PATH),
            creations_path= str(CREATIONS_PATH),
            timeout_seconds= 8,
            docker_image= os.getenv(
                'CODE_TESTER_DOCKER_IMAGE',
                'thesis-code-tester:latest',
            ) or 'thesis-code-tester:latest',
        )

        function_execution_report = CodeExecutionReport.model_validate(report_data)
        print(f'{BLUE}[NODE] [INFO] [EXECUTION REPORT]{RESET} {function_execution_report}') if DEBUG else None
        return {'function_execution_report': function_execution_report, 'execution_error': None}

    except Exception as e:
        print(f'\n{RED}[AGENT] [ERROR] [RUN CODE]{RESET} {e}')
        traceback.print_exc() if DEBUG else None
        return {'function_execution_report': None, 'execution_error': f'{type(e).__name__}: {e}'}

def review_code(state: CodeTesterIntermediate) -> CodeTesterOutput:
    print_function_name() if DEBUG else None

    try:
        prompt = prompts.REVIEW_PROMPT.format(
            function_name= state.function_name,
            code= read_state_file(state),
            imports= state.imports,
            implementation= state.implementation,
            special_instructions= state.se_instructions,
            report= state.function_execution_report            
        )

        # call the LLM
        response = safe_invoke(reviewer, messages= [SystemMessage(content= prompt)]).content
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        return CodeTesterOutput(reviewer_comments= response)

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        if state.has_a_report():
            return CodeTesterOutput(reviewer_comments= str(state.function_execution_report))

        return CodeTesterOutput(reviewer_comments= '')



''' Conditional Functions '''
def go_to_run_code(state: CodeTesterIntermediate) -> Literal['run_code', 'review_code']:
    if state.function_inputs is not None and len(state.function_inputs.kwargs) > 0:
        return 'run_code'
    
    return 'review_code'



''' Graph '''
code_tester_graph = StateGraph(CodeTesterIntermediate, input_schema= CodeTesterInput, output_schema= CodeTesterOutput)

code_tester_graph.add_node('generate_inputs', generate_inputs)
code_tester_graph.add_node('run_code', run_code)
code_tester_graph.add_node('review_code', review_code)

code_tester_graph.add_edge(START, 'generate_inputs')
code_tester_graph.add_conditional_edges(
    'generate_inputs',
    go_to_run_code,
    {   # For clarity, not needed
        'run_code': 'run_code',
        'review_code': 'review_code'
    }
)
code_tester_graph.add_edge('run_code', 'review_code')
code_tester_graph.add_edge('review_code', END)

code_tester_app = code_tester_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(code_tester_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/code_tester_app.png', 'wb') as f:
        f.write(code_tester_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'codeTester'
    os.environ['LANGSMITH_PROJECT'] = 'codeTester'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'codeTester',
            'run_name': 'codeTester',
            'thread_id': 'codeTester', 
        }
    }

    user = {
        'function_name': 'split_prompt',
        'file_path': r'C:\UCY\EPL-thesis\EPL-thesis\Clone\agents\promptEngineer\prompt_engineer.py',
        'implementation': '''
def split_prompt(content: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    """
    `split_prompt` splits the LLMs response into sections.
    
    `Args:`
        content (str): The content to split. Should be formatted as:
        ```
        #> Thinking Process
        ...
        #> Prompt
        ...
        #> Code Changes
        ## Change {{index}}
        ### Old Code
        ...
        ### New Code
        ...
        ```

    `Returns:`
        Tuple[str, str, List[str]]: The prompt, response, and comments.
    """
    thinking_process, other = content.split('#> Prompt\\n')
    prompt, changes = other.split('#> Code Changes\\n')

    if '### old code' not in clean_llm_output(changes.lower()):
        changes = ''
    
    if changes:
        changes = [c for c in changes.split('## Change') if c]

        changes = [
            (
                c.split('### Old Code\\n')[1].split('### New Code\\n')[0],
                c.split('### New Code\\n')[1]
            )
            for c in changes
        ]

    return (
        clean_llm_output(thinking_process),
        clean_llm_output(prompt),
        [
            (clean_llm_output(old_code), clean_llm_output(new_code)) 
            for old_code, new_code in changes
            if old_code != new_code
        ] if changes else []
    )''',
        'imports': [],
        'se_instructions': (
            'Implement split_prompt so it separates an LLM response into the thinking process, '
            'the proposed prompt, and a list of old-code/new-code change tuples. It must support '
            'responses with no code changes, clean every returned section using clean_llm_output, '
            'preserve the order of code changes, and exclude changes where the cleaned old and new '
            'code are identical.'
        )
    }
    response = code_tester_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
