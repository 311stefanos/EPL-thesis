"""
- `author:` Stefanos Panteli
- `date:` 2025-10-26
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.codeClarifier.codeClarifier import code_clarifier_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.codeClarifier.code_clarifier import code_clarifier_app
graph_input = { # TODO: add }

response = code_clarifier_app.invoke(graph_input)

# response = { # TODO: add }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage, HumanMessage, RemoveMessage
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

# Schema imports
from typing import TypedDict, Literal, List, Optional, Annotated, Union
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import traceback
import json
import os
import re

# My imports
from agents.workflowRefiner.workflow_refiner import WorkflowBundle
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, USER_APPROVALS
from agents.codeClarifier import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Code Clarifier') if DEBUG else None



""" Schemas """
''' General Schemas '''
# General
class Argument(BaseModel):
    name: str = Field(description= 'The name of the argument.')
    type: str = Field(description= 'The type of the argument.')

    def __str__(self):
        return f'{self.name}: {self.type}'
        
# Docstring agent
class Docstring(BaseModel):
    function: str = Field(description= 'The function name as given.')
    docstring: str = Field(description= 'The docstring as given.')

    def __str__(self):
        return f'Function: {self.function}\nDocstring: {self.docstring}'

class Docstrings(BaseModel):
    docstrings: List[Docstring] = Field(description= 'The docstrings as given from the user.')

    def __str__(self):
        return '\n'.join([f'\n{i}) {docstring}' for i, docstring in enumerate(self.docstrings, start= 1)])

# Helpful functions/Tool agent
class Function(BaseModel):
    function_name: str = Field(description= 'The proposed helper function name.')
    arguments: List[Argument] = Field(description= 'The arguments of the helper function.')
    output: str = Field(description= 'The output of the helper function.')
    docstring: str = Field(description= 'The docstring of the helper function.')
    justification: str = Field(description= 'The justification of the helper function. Why is needed.')

    def __str__(self):
        docstring = self.docstring.replace('\n', '\n\t')
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return f'def {self.function_name}({arguments}) -> {self.output}:\n\t"""\n\t{docstring}\n\t"""\n\t...'

class HelpfulFunctions(BaseModel):
    helpful_functions: List[Function] = Field(description= 'The helpful functions.')

    def __str__(self):
        return '\n'.join([f'\n{i}) {function.justification}\n{function}' for i, function in enumerate(self.helpful_functions, start= 1)])
    
class ToolFunctions(BaseModel):
    tool_functions: List[Function] = Field(description= 'The tool functions.')

    def __str__(self):
        return '\n'.join([f'\n{i}) {function.justification}\n@tool\n{function}' for i, function in enumerate(self.tool_functions, start= 1)])

# Schema agent
class Schema(BaseModel):
    schema_name: str = Field(description= 'The name of the schema in PascalCase.')
    docstring: str = Field(description= 'The docstring of the schema.')
    base_class: Literal['BaseModel', 'TypedDict'] = Field(description= 'The base class of the schema.')
    arguments: List[Argument] = Field(description= 'The arguments of the schema.')

    def __str__(self):
        arguments = '\n\t'.join([str(arg) for arg in self.arguments])
        docstring = self.docstring.replace('\n', '\n\t')
        return f'class {self.schema_name}({self.base_class}):\n\t"""\n\t{docstring}\n\t"""\n\t{arguments}\n'

class Schemas(BaseModel):
    schemas: List[Schema] = Field(description= 'The schemas.')

    def __str__(self):
        return '\n'.join([f'\n{i}) {schema}' for i, schema in enumerate(self.schemas, start= 1)])
    
''' Input Schema '''
class InputSchema(MessagesState):
    file_path: str = Field(description= 'The current file path as given from the user.')
    clarified_user_input: str = Field(description= 'The clarified user input as given from the clarifier.')
    workflow: WorkflowBundle = Field(description= 'The proposed workflow as given from the workflow engineer.')
    code_structure: str = Field(description= 'The code structure as given from the software engineer.')

    step_changes: Union[    
        Docstrings, 
        HelpfulFunctions,
        ToolFunctions,
        Schemas
    ] = Field(description= 'The step changes as given from the LLM at each step.')



''' Tools '''



''' LLM '''
docstring_generator = myChatOpenAI(
    temperature= 0.7
).with_structured_output(Docstrings)

helpful_function_generator = myChatOpenAI(
    temperature= 0.4
).with_structured_output(HelpfulFunctions)

tool_function_generator = myChatOpenAI(
    temperature= 0.4
).with_structured_output(ToolFunctions)

schema_generator = myChatOpenAI(
    temperature= 0.4
).with_structured_output(Schemas,)



''' Helpful Functions '''



""" Nodes """
''' Docstring Nodes '''
# The node that understands the code and comments the node functions
def generate_docstrings(state: InputSchema) -> InputSchema:
    ''' # TODO: add
    '''
    print_function_name() if DEBUG else None
    
    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.ANNOTATE_NODES_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= state['code_structure'],
            history= history
        )

        # call the LLM
        docstring_proposal: Docstrings = safe_invoke(docstring_generator, [SystemMessage(content= prompt)])

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [DOCSTRING PROPOSAL]{RESET} {docstring_proposal if docstring_proposal.docstrings else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(docstring_proposal)), HumanMessage(content= user_input)]

        return {'messages': new_messages, 'step_changes': docstring_proposal}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
# Updates the docstring of the code
def update_docstrings(state: InputSchema) -> InputSchema:
    ''' # TODO: add'''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    docstrings = state['step_changes'].model_dump()['docstrings']

    with open(file_path, 'r') as f:
        code = f.read()

    nodes = code.split("''' Nodes '''")[-1].split("''' Conditional Functions '''")[0].split("''' Graph '''")[0]

    new_nodes = []
    new_node = function_name = ''
    for line in nodes.split('\n'):
        # Signature line
        if line.startswith('def '):
            # Close previous node
            new_nodes.append(new_node) if new_node else None
            # Start new node
            new_node = line + '\n'
            # Get function name and generated docstring
            function_name = line.split(' ')[1].split('(')[0]
            for docstring in docstrings: 
                if docstring['function'] == function_name:
                    function_docstring = docstring['docstring'].replace('\n', '\n\t')
        # Docstring line
        elif line.strip().startswith('""" Execution: '):
            new_node += line[:-3] + f'\n\t{function_docstring}\n\t"""\n\n'
        else:
            new_node += line + '\n'

    # Append the last node
    new_nodes.append(new_node)
    # Get the whole section as a string
    new_nodes = '\n'.join(new_nodes)
    # Replace the section
    code = code.replace(nodes, new_nodes)

    with open(file_path, 'w') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}

''' Helpful Functions Nodes '''
# The node that reads the annotated code, and adds instructions for helpful functions
def propose_helpful_functions(state: InputSchema) -> InputSchema:
    ''' # TODO: add'''
    print_function_name() if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.ADD_HELPFUL_FUNCTIONS_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= state['code_structure'],
            history= history
        )

        # call the LLM
        helpful_functions_proposal: HelpfulFunctions = safe_invoke(helpful_function_generator, [SystemMessage(content= prompt)])

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [HELPFUL FUNCTIONS PROPOSAL]{RESET} {helpful_functions_proposal if helpful_functions_proposal.helpful_functions else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(helpful_functions_proposal)), HumanMessage(content= user_input)]

        return {'messages': new_messages, 'step_changes': helpful_functions_proposal}
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

def update_helpful_functions(state: InputSchema) -> InputSchema:
    ''' # TODO: add'''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    helpful_functions = state['step_changes'].helpful_functions

    with open(file_path, 'r') as f:
        code = f.read()

    new_functions = [str(function) for function in helpful_functions]
    new_functions = '\n\n'.join(new_functions)
    code = code.replace("''' Helpful Functions '''", f"''' Helpful Functions '''\n{new_functions}")

    with open(file_path, 'w') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}

''' Tool Functions Nodes '''
# TODO: add
def propose_tool_functions(state: InputSchema) -> InputSchema:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.ADD_TOOL_FUNCTIONS_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= state['code_structure'],
            history= history
        )

        # call the LLM
        tool_functions_proposal: ToolFunctions = safe_invoke(tool_function_generator, [SystemMessage(content= prompt)])

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [TOOL FUNCTIONS PROPOSAL]{RESET} {tool_functions_proposal if tool_functions_proposal.tool_functions else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(tool_functions_proposal)), HumanMessage(content= user_input)]

        return {'messages': new_messages, 'step_changes': tool_functions_proposal}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

def update_tool_functions(state: InputSchema) -> InputSchema:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    tool_functions = state['step_changes'].tool_functions

    with open(file_path, 'r') as f:
        code = f.read()

    new_functions = [str(function) for function in tool_functions]
    new_functions = '\n\n'.join(new_functions)
    code = code.replace("''' Tool Functions '''", f"''' Tool Functions '''\n{new_functions}")

    with open(file_path, 'w') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}

''' Schema Nodes '''
# TODO: add
def propose_schemas(state: InputSchema) -> InputSchema:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.PROPOSE_SCHEMAS_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= state['code_structure'],
            history= history
        )

        # call the LLM
        schemas_proposal: Schemas = safe_invoke(schema_generator, [SystemMessage(content= prompt)])

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [SCHEMAS PROPOSAL]{RESET} {schemas_proposal if schemas_proposal.schemas else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(schemas_proposal)), HumanMessage(content= user_input)]

        return {'messages': new_messages, 'step_changes': schemas_proposal}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

def update_schemas(state: InputSchema) -> InputSchema:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    schemas = state['step_changes'].schemas

    with open(file_path, 'r') as f:
        code = f.read()

    old_schemas = code.split('""" Schemas """')[-1].split("''' Tools '''")[0]

    agent_schema = [str(schema) for schema in schemas if schema.schema_name == 'AgentSchema']
    rest_schemas = [str(schema) for schema in schemas if schema.schema_name != 'AgentSchema']
    new_schemas = [''] + rest_schemas + agent_schema + ['']
    new_schemas = '\n'.join(new_schemas)
    
    code = code.replace(old_schemas, new_schemas)

    with open(file_path, 'w') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}

''' Conditional Functions '''
# TODO: add
def after_docstrings(state: InputSchema) -> Literal['generate_docstrings', 'update_docstrings']:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    last_user_message = state['messages'][-1].content
    if last_user_message in USER_APPROVALS:
        return 'update_docstrings'
    
    return 'generate_docstrings'

def after_helpful_functions(state: InputSchema) -> Literal['propose_helpful_functions', 'update_helpful_functions']:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    last_user_message = state['messages'][-1].content
    if last_user_message in USER_APPROVALS:
        return 'update_helpful_functions'
    
    return 'propose_helpful_functions'

def after_tool_functions(state: InputSchema) -> Literal['propose_tool_functions', 'update_tool_functions']:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    last_user_message = state['messages'][-1].content
    if last_user_message in USER_APPROVALS:
        return 'update_tool_functions'
    
    return 'propose_tool_functions'

def after_schemas(state: InputSchema) -> Literal['propose_schemas', 'update_schemas']:
    '''# TODO: add'''
    print_function_name() if DEBUG else None

    last_user_message = state['messages'][-1].content
    if last_user_message in USER_APPROVALS:
        return 'update_schemas'
    
    return 'propose_schemas'



''' Graph '''
code_clarifier_graph = StateGraph(InputSchema) # TODO: change

# code_clarifier_graph.add_node('generate_docstrings', generate_docstrings)
# code_clarifier_graph.add_node('update_docstrings', update_docstrings)
# code_clarifier_graph.add_node('propose_helpful_functions', propose_helpful_functions)
# code_clarifier_graph.add_node('update_helpful_functions', update_helpful_functions)
# code_clarifier_graph.add_node('propose_tool_functions', propose_tool_functions)
# code_clarifier_graph.add_node('update_tool_functions', update_tool_functions)
code_clarifier_graph.add_node('propose_schemas', propose_schemas)
code_clarifier_graph.add_node('update_schemas', update_schemas)

# code_clarifier_graph.add_edge(START, 'generate_docstrings')
# code_clarifier_graph.add_conditional_edges(
#     'generate_docstrings',
#     after_docstrings,
#     {   # Not needed, for clarity
#         'generate_docstrings': 'generate_docstrings',
#         'update_docstrings': 'update_docstrings'
#     }
# )
# code_clarifier_graph.add_edge('update_docstrings', 'propose_helpful_functions')
# code_clarifier_graph.add_conditional_edges(
#     'propose_helpful_functions',
#     after_helpful_functions,
#     {   # Not needed, for clarity
#         'propose_helpful_functions': 'propose_helpful_functions',
#         'update_helpful_functions': 'update_helpful_functions'
#     }
# )
# code_clarifier_graph.add_edge('update_helpful_functions', 'propose_tool_functions')
# code_clarifier_graph.add_conditional_edges(
#     'propose_tool_functions',
#     after_tool_functions,
#     {   # Not needed, for clarity
#         'propose_tool_functions': 'propose_tool_functions',
#         'update_tool_functions': 'update_tool_functions'
#     }
# )

code_clarifier_graph.add_edge(START, 'propose_schemas')

code_clarifier_graph.add_conditional_edges(
    'propose_schemas',
    after_schemas,
    {   # Not needed, for clarity
        'propose_schemas': 'propose_schemas',
        'update_schemas': 'update_schemas'
    }
)
code_clarifier_graph.add_edge('update_schemas', END)

code_clarifier_app = code_clarifier_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(code_clarifier_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/code_clarifier_app.png', 'wb') as f:
        f.write(code_clarifier_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'codeClarifier'
    os.environ['LANGSMITH_PROJECT'] = 'codeClarifier'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'codeClarifier',
            'run_name': 'codeClarifier',
            'thread_id': 'codeClarifier', 
        }
    }

    from test_inputs import file_path, clarified_user_input, workflow, code_structure
    user = InputSchema(
        file_path= file_path,
        clarified_user_input= clarified_user_input,
        workflow= workflow,
        code_structure= code_structure
    )

    response = code_clarifier_app.invoke(user, config= config)

    # print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    # if DEBUG:
    #     for key, value in response.items():
    #         print(f'    {key}: {value}')
