"""
- `author:` Stefanos Panteli
- `date:` 2026-01-17
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.fileHandler.file_handler import file_handler_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.fileHandler.file_handler import file_handler_app
graph_input = { # TODO: add }

response = file_handler_app.invoke(graph_input)

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
from typing import Tuple, TypedDict, Literal, List, Optional, Annotated
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS, read_state_file, clean_llm_output
from agents.fileHandler import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} File Handler') if DEBUG else None



""" Schemas """
''' General Schemas '''

''' Input Schema '''
class InputSchema(MessagesState):
    file_path: str

''' Intermediate Schemas '''

''' Output Schema '''


''' Global Variables '''
# The project directory. Only under this directory can files be changed.
project_dir: Path = None
immutable_files: List[str] = []



''' Tools '''
# Creates a directory
@tool
def create_directory(directory_path: str) -> str:
    '''
    `create_directory` creates a directory with the given path.

    `Args:`
        directory_path (str): The relative path to the directory to create.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    try:
        global project_dir
        target = resolve_under_project(directory_path)
        after_creations = target.parts[target.parts.index('creations'):]

        if target is None:
            print(f'{RED}[TOOL] [ERR]{RESET} The directory {directory_path} must be under {project_dir}.') if DEBUG else None
            return f'[ERROR] The directory {directory_path} must be a child of {project_dir}.'

        # Ensure parent directory exists
        target.parent.mkdir(parents= True, exist_ok= True)

        if target.exists():
            print(f'{RED}[TOOL] [ERR]{RESET} The directory {after_creations} already exists.') if DEBUG else None
            return f'[ERROR] The directory {after_creations} already exists.'

        target.mkdir()

        print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Created the directory {after_creations} successfully.') if DEBUG else None
        return f'[TOOL] [INFO] [SUCCESS] Created the directory {after_creations} successfully.'

    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
        return f'[ERROR] {e}'

# Creates an external file
@tool
def create_file(file_path: str, contents: str) -> str:
    '''
    `create_file` creates a file with the given contents.

    `Args:`
        file_path (str): The relative path to the file to create.
        contents (str): The contents of the file to create.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None
    try:
        global project_dir
        target = resolve_under_project(file_path)
        after_creations = target.parts[target.parts.index('creations'):]

        if target is None:
            print(f'{RED}[TOOL] [ERR]{RESET} The file {file_path} must be under {project_dir}.') if DEBUG else None
            return f'[ERROR] The file {file_path} must be a child of {project_dir}.'

        # Ensure parent directory exists
        target.parent.mkdir(parents= True, exist_ok= True)

        if target.exists():
            print(f'{RED}[TOOL] [ERR]{RESET} The file {after_creations} already exists.') if DEBUG else None
            return f'[ERROR] The file {after_creations} already exists.'

        with open(target, 'w', encoding='utf-8') as f:
            f.write(contents)

        print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Created the file {after_creations} successfully.') if DEBUG else None
        return f'[SUCCESS] Created the file {after_creations} successfully.\n{format_contents(file_path, contents)}'

    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
        return f'[ERROR] {e}'

@tool
def modify_file(file_path: str, file_changes: List[Tuple[str, str]]) -> str:
    '''
    `modify_file` modifies the file at the given path.

    `Args:`
        file_path (str): The path to the file to modify.
        file_changes (List[Tuple[str, str]]): A list of tuples containing the old lines and the new lines. 
            For each tuple, the first element is the old lines and the second element is the new lines. The file_contents.replace(old_lines, new_lines) will be called on the file.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    try:
        global project_dir, immutable_files

        target = resolve_under_project(file_path)
        after_creations = target.parts[target.parts.index('creations'):]
        
        if target is None:
            print(f'{RED}[TOOL] [ERR]{RESET} The file {file_path} must be under {project_dir}.') if DEBUG else None
            return f'[ERROR] The file {file_path} must be a child of {project_dir}.'
        
        if target.name in immutable_files:
            print(f'{RED}[TOOL] [ERR]{RESET} The file {after_creations} is immutable.') if DEBUG else None
            return f'[ERROR] The file {after_creations} is immutable.'
        
        with open(target, 'r', encoding='utf-8') as f:
            contents = f.read()

        for (old_lines, new_lines) in file_changes:
            if old_lines not in contents:
                print(f'{RED}[TOOL] [ERR]{RESET} The old lines ```\n{old_lines}\n``` does not exist in the file {file_path}.') if DEBUG else None
                continue
            
            contents = contents.replace(old_lines, new_lines)

        with open(target, 'w', encoding='utf-8') as f:
            f.write(contents)

        print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Modified the file {after_creations} successfully.') if DEBUG else None
        return f'[SUCCESS] Modified the file {after_creations} successfully.\n{format_contents(file_path, contents)}'

    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
        return f'[ERROR] {e}'

@tool
def read_file(file_path: str) -> str:
    '''
    `read_file` reads the file at the given path.

    `Args:`
        file_path (str): The path to the file to read.

    `Returns:`
        (str) The contents of the file
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    try:
        global project_dir

        target = resolve_under_project(file_path)
        after_creations = target.parts[target.parts.index('creations'):]
        
        if target is None:
            print(f'{RED}[TOOL] [ERR]{RESET} The file {file_path} must be under {project_dir}.') if DEBUG else None
            return f'[ERROR] The file {file_path} must be a child of {project_dir}.'
        
        with open(target, 'r', encoding='utf-8') as f:
            contents = f.read()

        print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Read the file {after_creations} successfully.') if DEBUG else None
        return f'[TOOL] [INFO] [SUCCESS] Read the file {after_creations} successfully.\n{format_contents(file_path, contents)}'

    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
        return f'[ERROR] {e}'

@tool
def done(comment: str) -> str:
    '''
    `done` is a tool used to indicate that the agent is done and the workflow is complete. Should be called alone.
    
    `Args:`
        comment (str): A comment for the system and user.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None
    print(f'{BLUE}[TOOL] [INFO] [DONE]{RESET} Done with the workflow. Comment: {comment}') if DEBUG else None

    return f'[TOOL] [DONE] Done with the workflow. Comment: {comment}'

tools = [
    create_directory,
    create_file,
    modify_file,
    read_file,
    done
] # For delete_file, it should output within the done tool to let the user know



''' LLM '''
file_handler = myChatOpenAI(
    temperature= 0.3
).bind_tools(tools)



''' Helpful Functions '''
def resolve_under_project(path: str) -> Optional[Path]:
    '''
    `resolve_under_project` resolves the path to be under the project directory.

    `Args:`
        path (str): The path to resolve.

    `Returns:`
        (Optional[Path]) The resolved path. None if the path is not under the project directory.
    '''
    global project_dir
    project = project_dir.resolve()
    path: Path = Path(path)
    
    # if path is an absolute path, check if it is under the project directory
    if path.is_absolute():
        abs_path = path.resolve()

        if abs_path.is_relative_to(project):
            return abs_path     

        return None
    
    parts = path.parts

    # Strip any leading overlap with the tail of project_dir
    for k in range(min(len(parts), len(project.parts)), 0, -1):
        if parts[:k] == project.parts[-k:]:
            parts = parts[k:]
            break
    
    target = (project / Path(*parts)).resolve()
    if target.is_relative_to(project):
        return target
    
    return None

def format_contents(file_path: str, contents: Optional[str]= None) -> str:
    '''
    `format_contents` formats the contents of the file at the given path and returns them as a string.

    `Args:`
        file_path (str): The path to the file to format.
        content (Optional[str]): The contents of the file to format.

    `Returns:`
        (str) The formatted contents of the file
    '''
    if not contents:
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = f.read()

    return f'File {file_path}:\n{contents}\n\n'

def read_sibling_files(file_path: str) -> str:
    '''
    `read_sibling_files` reads the sibling files of the file at the given path.

    `Args:`
        file_path (str): The path to the file to read.

    `Returns:`
        (str) The contents of the sibling files
    '''
    def children_of(path: Path) -> list[Path]:
        try:
            kids = list(path.iterdir())
            # Sort directories first, then files.
            # Sorted alphabetically
            kids.sort(key= lambda x: (not x.is_dir(), x.name.lower()))
            return kids
        
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return []
    
    p = Path(file_path)
    root = p.parent.resolve()
    lines: list[str] = [f'{root.name}/']

    def walk(dirpath: Path, prefix: str) -> None:
        kids = children_of(dirpath)
        for i, child in enumerate(kids):
            last = (i == len(kids) - 1)
            branch = '└── ' if last else '├── '
            name = f'{child.name}/' if child.is_dir() else child.name
            lines.append(prefix + branch + name)

            if child.is_dir():
                ext_prefix = '    ' if last else '│   '
                walk(child, prefix + ext_prefix)
            
    walk(root, '')
    return '\n'.join(lines)



''' Nodes '''
def get_project_dir(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    global project_dir
    project_dir = Path(state['file_path']).parent.resolve()

    print(f'{BLUE}[NODE] [INFO] [PROJECT_DIR]{RESET} {project_dir}') if DEBUG else None

    return state

def get_immutable_files(state: InputSchema) -> InputSchema:
    global project_dir, immutable_files
    def collect_files(dir: Path) -> None:
        try:
            for child in dir.iterdir():
                if child.is_file():
                    # store paths relative to project_dir for uniqueness and clarity
                    try:
                        rel = child.relative_to(project_dir)
                    except Exception:
                        rel = child.name
                    immutable_files.append(str(rel))
                elif child.is_dir():
                    collect_files(child)
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return

    print_function_name() if DEBUG else None
    collect_files(project_dir)
    i_f: str = '\n'.join(immutable_files)
    print(f'{BLUE}[NODE] [INFO] [IMMUTABLE_FILES]{RESET} {i_f}') if DEBUG else None

    return state

def file_handler_node(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        # prompt
        code_file: str = state['file_path'].split('/')[-1]
        prompt_file: str = code_file.replace('.py', '_prompts.py')

        with open(state['file_path'].replace('.py', '_prompts.py'), 'r', encoding='utf-8') as f:
            prompt: str = f.read()

        prompt = prompts.FILE_HANDLER_PROMPT.format(
            code_file= code_file,
            code= read_state_file(state),
            prompt_file= prompt_file,
            prompt= prompt,
            files= read_sibling_files(state['file_path'])
        )

        # Call the LLM
        response: str = safe_invoke(file_handler, messages= [SystemMessage(content= prompt)] + state['messages'])
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        return {'messages': [response]}

    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
        return state
    
def error_call(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    latest_message: AIMessage = state['messages'][-1]
    tool_calls = latest_message.tool_calls or latest_message.additional_kwargs.get('tool_calls', [])

    error_tool_messages: List[ToolMessage] = []
    for tool_call in tool_calls:
        if 'function' in tool_call:
            tool_call = tool_call['function']

        error_tool_messages.append(ToolMessage(
            name= tool_call['name'],
            tool_call_id= tool_call['id'],
            content= f'[TOOL] [ERR] Did not call tool named `{tool_call["name"]}` because multiple tools were called together with the `done` tool.'
        ))

    error_call_msg: str = prompts.ERROR_TOOL_CALL_PROMPT
    return {'messages': [SystemMessage(content= error_call_msg)] + error_tool_messages}



''' Conditional Functions '''
def should_end(state: InputSchema) -> Literal['file_handler_node', 'tools', 'error_call', '__end__']:
    print_function_name() if DEBUG else None

    last_message: AIMessage = state['messages'][-1]
    tool_calls = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', [])

    if (
        len(tool_calls) == 1 and
        tool_calls[0]['name'] == 'done'    
    ):
        return '__end__'
    
    if (
        len(tool_calls) != 1 and
        any(tool_call['name'] == 'done' for tool_call in tool_calls)
    ):
        return 'error_call'
    
    if will_tool_call(state['messages']):
        return 'tools'
    
    return 'file_handler_node'



''' Graph '''
file_handler_graph = StateGraph(InputSchema)

file_handler_graph.add_node('get_project_dir', get_project_dir)
file_handler_graph.add_node('get_immutable_files', get_immutable_files)
file_handler_graph.add_node('file_handler_node', file_handler_node)
file_handler_graph.add_node('tools', ToolNode(tools))
file_handler_graph.add_node('error_call', error_call)

file_handler_graph.add_edge(START, 'get_project_dir')
file_handler_graph.add_edge('get_project_dir', 'get_immutable_files')
file_handler_graph.add_edge('get_immutable_files', 'file_handler_node')
file_handler_graph.add_conditional_edges(
    'file_handler_node',
    should_end,
    {   # Not needed, for clarity
        'file_handler_node': 'file_handler_node',
        'tools': 'tools',
        'error_call': 'error_call',
        '__end__': END
    }
)
file_handler_graph.add_edge('tools', 'file_handler_node')
file_handler_graph.add_edge('error_call', 'file_handler_node')

file_handler_app = file_handler_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(file_handler_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/file_handler_app.png', 'wb') as f:
        f.write(file_handler_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'fileHandler'
    os.environ['LANGSMITH_PROJECT'] = 'fileHandler'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'fileHandler',
            'run_name': 'fileHandler',
            'thread_id': 'fileHandler', 
        }
    }

    user = {
        'file_path': '../../creations/menu_recommendation_workflow/menu_recommendation_workflow.py'
    }
    response = file_handler_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
