"""
- `author:` Stefanos Panteli
- `date:` 2025-11-03
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.softwareEngineer.softwareEngineer import software_engineer_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.softwareEngineer.software_engineer import software_engineer_app
graph_input = { # TODO: add }

response = software_engineer_app.invoke(graph_input)

# response = { # TODO: add }
```
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
from typing import TypedDict, Literal, List, Optional, Annotated, Dict
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments
from agents.softwareEngineer import prompts
from agents.coder.coder import (
    InputSchema as CoderInputSchema,
    FunctionProposal,
    OutputSchema as CoderOutputSchema,
    coder_app
)



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Software Engineer') if DEBUG else None



""" Schemas """
''' General Schemas '''
class CoderComment(BaseModel):
    comment: Optional[str] = Field(description= 'The comments from the software engineer.', default= None)

class Assignements(BaseModel):
    function_name: str = Field(description= 'The name of the function for the coder to implement.')
    special_instructions: str = Field(description= 'The comments from the software engineer.')    

''' Input Schema '''
class InputSchema(MessagesState):
    file_path: str = Field(description= 'The path to the file.')
    code_issues: Optional[str] = Field(description= 'The code issues the lead software engineer found.')



''' Intermediate Schemas '''

''' Output Schema '''



''' Global Variables '''
coders: Dict[str, CoderOutputSchema] = {}
comments: Dict[str, CoderComment] = {}



''' Tools '''
@tool
def write_code_to_file(file_path: str, code: str) -> str:
    '''
    `write_code_to_file` writes the contents of `code` to the file `file_path`
    This tool should be called a coder returned a code you approved.
    
    `Args:`
        file_path (str): The path to the file to overwrite.
        code (str): The code to write to the file.

    `Returns:`
        (str) If the file overwrite was successful
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    try:
        while code[0] == '<':
            index = code.find('>\n')
            code = code[index + 1:].strip()

        while code.strip()[-1] == '>':
            index = code.rfind('<')
            code = code[:index].strip()

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        return f'[GOOD] The file {file_path} was overwritten successfully.'

    except Exception as e:
        return f'[ERROR] The file {file_path} could not be overwritten due to the error: {e}'

@tool
def call_coder(function_name: str, special_instructions: str, file_path: str) -> Dict[str, CoderOutputSchema]:
    '''
    `call_coder` calls a coder to implement the given function.

    `Args:`
        function_name (str): The name of the function to implement.
        special_instructions (str): The special instructions from the software engineer to the coder.
        file_path (str): The path to the file to implement the function in.

    `Returns:`
        (Dict[str, CoderOutputSchema]): {function_name: CoderOutputSchema}
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    if function_name not in coders:
        coders[function_name] = CoderOutputSchema(code= '', proposals= None)
        comments[function_name] = CoderComment()

    args: CoderInputSchema = {
        'messages': [],
        'file_path': file_path,
        'function_name': function_name,
        'software_engineer_instructions': special_instructions,
        'previous_outputs': [coders[function_name].code] if coders[function_name].code else [],
        'comments': [comments[function_name].comment] if comments[function_name].comment else [],
    }
    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'softwareEngineer',
            'run_name': 'softwareEngineer',
            'thread_id': function_name, 
        }
    }

    try:
        response: CoderOutputSchema = coder_app.invoke(args, config)
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
    
    print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

    coders[function_name] = CoderOutputSchema(code= response['code'], proposals= response['proposals'])

    return {function_name: coders[function_name]}

@tool
def disapprove_and_comment_on_coder_code(function_name: str, comment: str) -> str:
    '''
    `disapprove_and_comment_on_coder_code` use it to comment on the coder's output code. Only use it when you think the coder's output is incorrect and you did not approve it.
    Should be used after `call_coder` if the coder's output is incorrect and you did not approve it.

    `Args:`
        function_name (str): The name of the function to comment on.
        comment (str): The comment to add to the coder's output.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    all_keys = set(list(coders.keys()) + list(comments.keys()))
    if function_name not in all_keys:
        return f'[ERROR] The coder for function {function_name} does not exist.'

    comments[function_name].comment = comment

    return f'[SUCCESS] Commented on the coder\'s output for function {function_name}: {comment}\n\nNow ready to be called again and understand the incorrect code.'
    
@tool
def approve_function_code(file_path: str, function_name: str) -> str:
    '''
    `approve_coder_output` approves the coder's output code. Only use it when you think the coder's output is correct.
    Should be used after `call_coder`, if the coder's output is correct and approved.

    `Args:`
        file_path (str): The name of the file to overwrite.
        function_name (str): The name of the function to approve.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    all_keys = set(list(coders.keys()) + list(comments.keys()))
    if function_name not in all_keys:
        return f'[ERROR] The coder for function {function_name} does not exist.'
    
    previous_code: str = coders[function_name].code
    if not previous_code:
        return f'[ERROR] The coder for function {function_name} does not have a previous implementation.'
    
    # Replace the code in the file
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Get the code sections from the function onwards
    code_section: str = f'def {function_name}(' + f'def {function_name}('.join(code.split(f'def {function_name}(')[1:])
    for line in code_section.split('\n')[1:]:
        if line.startswith('def ') and line.endswith(':'):
            code_section = code_section.split(line)[0].strip()
            break

    if not code_section:
        return f'[ERROR] The coder for function {function_name} does not have a code section in the file.'
    code = code.replace(code_section, previous_code)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Remove the function from the coders and comments
    del coders[function_name]
    del comments[function_name]

    return f'[SUCCESS] Approved the coder\'s output for function {function_name}.\n\nNow ready to be called again and understand the code.'

@tool
def approve_function_proposals(approved_function_proposals: List[FunctionProposal], file_path: str) -> str:
    # TODO: check proposal_coder_function_name: str, approved_function_proposal_names: List[str]
    '''
    `approve_function_proposals` approves a subset of a coder's function proposals. Only use it when you think the coder's function proposals are correct.
    Should be used after `call_coder`, if the coder's function proposals are correct and approved.
    - Note: You may approve multiple function proposals at once.
    - You may approve a function proposal even if the coder's code output is not approved.

    `Args:`
        function_proposals (List[FunctionProposal]): The function proposals to approve. As given from the coder.
        file_path (str): The path to the file to implement the function in.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    proposed_tools = [str(function_proposal) for function_proposal in approved_function_proposals if function_proposal.function_type == 'tool']
    proposed_functions = [str(function_proposal) for function_proposal in approved_function_proposals if function_proposal.function_type == 'helper_function']

    code = read_state_file({'file_path': file_path})
    code = code.replace('# TODO: Add Tools', '\n\n'.join(proposed_tools) + '\n\n# TODO: Add Tools')
    code = code.replace('# TODO: Add Helpful Functions', '\n\n'.join(proposed_functions) + '\n\n# TODO: Add Helpful Functions')
    write_code_to_file.invoke({"file_path": file_path, "code": code}) 

    return f'[SUCCESS] Approved the coder\'s function proposals: {[afp.function_name for afp in approved_function_proposals]}.\n\nThe file contents have been updated.'

@tool
def submit_final_code(file_path: str) -> None:
    '''
    `submit_final_code` submits the final implementation of the file. You must implement all the functions before calling this.

    `Args:`
        file_path (str): The path to the file to implement the function in.
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None
    print(f'{BLUE}[NODE] [INFO] [SUBMIT]{RESET} {file_path} implemented successfully.') if DEBUG else None

tools = [write_code_to_file, call_coder, disapprove_and_comment_on_coder_code, approve_function_code, approve_function_proposals, submit_final_code]



''' LLM '''
mistake_correcter = myChatOpenAI(
    temperature= 0.2,
    model= 'qwen/qwen3-235b-a22b:free'
    # 'qwen/qwen3-coder:free'
)

software_engineer = myChatOpenAI(
    temperature= 0.4
).bind_tools(tools)

code_validator = myChatOpenAI(
    temperature= 0.4
)



''' Helpful Functions '''
# Reads the contents of state['file_path']
def read_state_file(state: InputSchema) -> str:
    '''
    `read_state_file` reads the contents of state['file_path']
    
    `Args:`
        state (InputSchema): The state of the agent. Must have the key 'file_path'.

    `Returns:`
        code: str
    '''
    with open(state['file_path'], 'r', encoding='utf-8') as f:
        code = f.read()
    return code

def write_state_file(state: InputSchema, code: str) -> None:
    '''
    `write_state_file` writes the contents of code to state['file_path']
    
    `Args:`
        state (InputSchema): The state of the agent. Must have the key 'file_path'.
        code (str): The code to write to the file.
    '''
    with open(state['file_path'], 'w', encoding='utf-8') as f:
        f.write(code)



''' Nodes '''
def fix_mistakes(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        # prompt
        prompt = prompts.FIX_PROMPT.format(
            code= read_state_file(state)
        )

        # call the LLM
        response = safe_invoke(mistake_correcter, [SystemMessage(content= prompt)]).content
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        if response == '':
            return state

        write_state_file(state, response)

        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state

def software_engineer_node(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        # prompt
        last_message = state['messages'][-1] if state['messages'] else None
        last_prompt = ''
        if hasattr(last_message, 'name'):
            if last_message.name == 'call_coder':
                last_prompt = '\n# Next Step:\nApprove requests, Approve code, Disapprove code using the respective tools.\n\n'

        code_issues = state.get('code_issues', '')
        code_issues_prompt = prompts.CODE_ISSUES_SECTION.format(
            code_issues= code_issues
        )

        tool_history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', []) if isinstance(mes, ToolMessage)])
        functions = '\n\n---\n\n'.join([
            f'{function}:\n{coder.code}' 
            for function, coder in coders.items() if function not in comments
        ])
        prompt = prompts.SOFTWARE_ENGINEER_PROMPT.format(
            file_path= state['file_path'],
            code= read_state_file(state),
            tool_history= tool_history,
            code_issues= code_issues_prompt,
            functions= functions,
        ) + last_prompt

        # call the LLM
        response = safe_invoke(software_engineer, [SystemMessage(content= prompt)])
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        return {'messages': [response]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state



def last_check(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        # prompt
        prompt = prompts.LAST_CHECK_PROMPT.format(
            code= read_state_file(state)
        )

        # call the LLM
        response = safe_invoke(code_validator, [SystemMessage(content= prompt)]).content
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        return {'code_issues': response}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state



''' Conditional Functions '''
def after_software_engineer(state: InputSchema) -> Literal['last_check', 'software_engineer_node', 'tools']:
    print_function_name() if DEBUG else None

    # Plain language response
    if not will_tool_call(state['messages']):
        return 'software_engineer_node'
    
    # Get the last message and extract the tool calls
    last_message = state['messages'][-1]
    tool_call = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', [])
    if tool_call:
        tool_call = tool_call[0]
    # If the last message is not a tool call, go back to the coder node
    else:
        return 'software_engineer_node'
    
    # Get the tool function and name
    if 'function' in tool_call:
        tool_call = tool_call['function']

    # If the tool is the output tool, go to the output node
    if tool_call['name'] == 'submit_final_code':
        return 'last_check'
    
    # Else, go to the tool node
    return 'tools'


def passed_last_check(state: InputSchema) -> Literal['software_engineer_node', '__end__']:
    print_function_name() if DEBUG else None

    code_issues = state['code_issues'] or ''

    if code_issues.strip().lower() in ['yes', 'good', 'end', '__end__', '', 'okay']:
        return '__end__'
    
    return 'software_engineer_node' 



''' Graph '''
software_engineer_graph = StateGraph(InputSchema) # TODO: change

software_engineer_graph.add_node('fix_mistakes', fix_mistakes)
software_engineer_graph.add_node('software_engineer_node', software_engineer_node)
software_engineer_graph.add_node('tools', ToolNode(tools))#, handle_tool_errors= False))
software_engineer_graph.add_node('last_check', last_check)

# software_engineer_graph.add_edge(START, 'fix_mistakes')
# software_engineer_graph.add_edge('fix_mistakes', 'software_engineer_node')
software_engineer_graph.add_edge(START, 'software_engineer_node')
software_engineer_graph.add_conditional_edges(
    'software_engineer_node', 
    after_software_engineer,
    {   # Not needed, for clarity
        'last_check': 'last_check',
        'software_engineer_node': 'software_engineer_node',
        'tools': 'tools'
    }
)
software_engineer_graph.add_edge('tools', 'software_engineer_node')
software_engineer_graph.add_conditional_edges(
    'last_check', 
    passed_last_check,
    { # Not needed, for clarity
        'software_engineer_node': 'software_engineer_node',
        '__end__': END
    }
)

software_engineer_app = software_engineer_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(software_engineer_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/software_engineer_app.png', 'wb') as f:
        f.write(software_engineer_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'softwareEngineer'
    os.environ['LANGSMITH_PROJECT'] = 'softwareEngineer'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'softwareEngineer',
            'run_name': 'softwareEngineer',
            'thread_id': 'softwareEngineer', 
        }
    }

    user = InputSchema(file_path= '../../creations/fitness_program_generator/fitness_program_generator.py', code_issues= None)
    response = software_engineer_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
