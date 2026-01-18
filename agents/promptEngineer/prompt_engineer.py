"""
- `author:` Stefanos Panteli
- `date:` 2025-12-24
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.promptEngineer.prompt_engineer import prompt_engineer_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.promptEngineer.prompt_engineer import prompt_engineer_app
graph_input = { # TODO: add }

response = prompt_engineer_app.invoke(graph_input)

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
from typing import TypedDict, Literal, List, Optional, Annotated, Dict, Union
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import json
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS
from agents.promptEngineer import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

# MEMORY_PATH = Path(__file__).resolve().parent / 'memory.json'

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Prompt Engineer') if DEBUG else None



""" Schemas """
''' General Schemas '''
class Format(BaseModel):
    format_dict: Dict = Field(description= 'The dictionary used to format the prompt.')

class Prompt(BaseModel):
    prompt_name: str = Field(description= 'The name of the prompt.')
    suggested_prompt: str = Field(description= 'The suggested prompt.', default= '')

    format: Format = Field(description= 'The format of the prompt.', default= Format(format_dict= {}))

    user_comments: List[str] = Field(description= 'The user comments.', default= [])
    latest_response: str = Field(description= 'The latest response.', default= '')

    def set_format(self, format: Format) -> None:
        self.format = format

    def add_comments(self, comments: str) -> None:
        self.user_comments.append(comments)

    def add_response(self, response: str) -> None:
        self.latest_response = response

    def is_approved(self) -> bool:
        last_comment = self.user_comments[-1] if self.user_comments else ''
        last_comment = last_comment.replace('Review by Expert Reviewer: ', '')
        last_comment = last_comment.replace('Review by User: ', '')

        return not any(c.strip() not in USER_APPROVALS for c in last_comment.split())
    
    def filter_comments(self) -> List[str]:
        comments: List[str] = []
        for comment in self.user_comments:
            clear_comment: str = comment.replace('Review by Expert Reviewer: ', '')
            clear_comment = clear_comment.replace('Review by User: ', '')
            if any(c.strip() not in USER_APPROVALS for c in clear_comment.split()):
                comments.append(comment)

        return comments

''' Input Schema '''
class InputSchema(TypedDict):
    file_path: str = Field(description= 'The path to the file.')
    prompt_list: Optional[List[Prompt]] = Field(description= 'The list of prompts.')
    active_prompt_index: Optional[int] = Field(description= 'The active prompt index of the prompt list.')

    mode: Literal['llm', 'user', 'both'] = Field(description= 'The mode of the agent.')
    # tool_call: Optional[List[BaseMessage]] = Field(description= 'The tool call message.')

''' Intermediate Schemas '''

''' Output Schema '''



# ''' Tools '''
# # A tool to create a new entry in the long term memory of the agent, by the agent
# def new_memory(new_memory: str) -> str:
#     '''
#     `new_memory` inserts a new memory into the long term memory. 
#         This memory corresponds to the correct techniques to create a prompt.
    
#     `Args:`
#         new_memory (str): The new memory.

#     `Returns:`
#         (str): A message.
#     '''
#     print_function_name(colour= MAGENTA) if DEBUG else None

#     try:
#         with open(MEMORY_PATH, 'w+', encoding='utf-8') as f:
#             memory: dict = json.load(f)
#             max_key = max([int(k) for k in memory.keys()])
#             memory[str(max_key + 1)] = new_memory
#             json.dump(memory, f, indent= 4)

#         print(f'{BLUE}[NODE] [INFO] [UPDATE MEMORY]{RESET} Memory created successfully.') if DEBUG else None
#         return f'{BLUE}[NODE] [INFO] [UPDATE MEMORY]{RESET} Memory created successfully.'

#     except Exception as e:
#         print(f'{RED}[NODE] [ERROR] [UPDATE MEMORY]{RESET} Failed to create memory: {e}') if DEBUG else None
#         return f'{RED}[NODE] [ERROR] [UPDATE MEMORY]{RESET} Failed to create memory: {e}'

# # A tool to update the long term memory of the agent, by the agent
# def change_a_memory(memory_index: Union[str,int], new_memory: str) -> str:
#     '''
#     `change_a_memory` changes a memory entry of the long term memory. 
#         This memory corresponds to the correct techniques to create a prompt.
    
#     `Args:`
#         memory_index (Union[str,int]): The memory index. Has to be a number in int or str format.
#         new_memory (str): The new memory.

#     `Returns:`
#         (str): A message.
#     '''
#     print_function_name(colour= MAGENTA) if DEBUG else None

#     try:
#         with open(MEMORY_PATH, 'w+', encoding='utf-8') as f:
#             memory: dict = json.load(f)
#             memory[str(memory_index)] = new_memory
#             json.dump(memory, f, indent= 4)

#         print(f'{BLUE}[NODE] [INFO] [UPDATE MEMORY]{RESET} Memory updated successfully.') if DEBUG else None
#         return f'{BLUE}[NODE] [INFO] [UPDATE MEMORY]{RESET} Memory updated successfully.'

#     except Exception as e:
#         print(f'{RED}[NODE] [ERROR] [UPDATE MEMORY]{RESET} Failed to update memory: {e}') if DEBUG else None
#         return f'{RED}[NODE] [ERROR] [UPDATE MEMORY]{RESET} Failed to update memory: {e}'

# # A tool to delete an entry in the long term memory of the agent, by the agent
# def delete_a_memory(memory_index: Union[str,int]) -> str:
#     '''
#     `delete_a_memory` deletes a memory entry of the long term memory. 
#         This memory corresponds to the correct techniques to create a prompt.
    
#     `Args:`
#         memory_index (Union[str,int]): The memory index. Has to be a number in int or str format.

#     `Returns:`
#         (str): A message.
#     '''
#     print_function_name(colour= MAGENTA) if DEBUG else None

#     try:
#         with open(MEMORY_PATH, 'w+', encoding='utf-8') as f:
#             memory: dict = json.load(f)
#             del memory[str(memory_index)]
#             json.dump(memory, f, indent= 4)

#         print(f'{BLUE}[NODE] [INFO] [UPDATE MEMORY]{RESET} Memory deleted successfully.') if DEBUG else None
#         return f'{BLUE}[NODE] [INFO] [UPDATE MEMORY]{RESET} Memory deleted successfully.'

#     except Exception as e:
#         print(f'{RED}[NODE] [ERROR] [UPDATE MEMORY]{RESET} Failed to delete memory: {e}') if DEBUG else None
#         return f'{RED}[NODE] [ERROR] [UPDATE MEMORY]{RESET} Failed to delete memory: {e}'

# prompt_engineer_tools = [new_memory, change_a_memory, delete_a_memory]



''' LLM '''
prompt_engineer = myChatOpenAI(
    temperature= 0.4
)#.bind_tools(prompt_engineer_tools)

prompt_reviewer = myChatOpenAI(
    temperature= 0.25
)

formater = myChatOpenAI(
    temperature= 0.1
).with_structured_output(Format)

tester = myChatOpenAI(
    temperature= 0.7
)

response_reviewer = myChatOpenAI(
    temperature= 0.25
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

# Reads the contents of file in 'file_path' and returns the prompts names
def get_prompt_names(file_path: str) -> List[str]:
    '''
    `get_prompt_names` reads the contents of state['file_path'] and returns the prompts names
    
    `Args:`
        file_path (str): The file path of the file to extract.

    `Returns:`
        List[str]: The prompts names
    '''
    # Capture the prompt name after `prompts.`
    # * = prompts.* # Can be in multiple lines.
    pattern = re.compile(r'^\s*\w+\s*=\s*prompts\.([A-Z][A-Z0-9_]*)\b', re.MULTILINE)

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    names_in_order = [m.group(1) for m in pattern.finditer(code)]

    # Keep unique names, preserve first-seen order
    seen = set()
    unique_prompt_names: List[str] = []
    for name in names_in_order:
        if name not in seen:
            seen.add(name)
            unique_prompt_names.append(name)

    return unique_prompt_names

# Returns a boolean value depending on the mode of the state
def mode(mode: Literal['llm', 'user'], state: InputSchema) -> bool:
    '''
    `mode` returns a boolean value depending on the mode of the state.
    
    `Args:`
        mode (Literal['llm', 'user']): The mode of the agent.
        state (InputSchema): The state of the agent. Must have the key 'mode'.

    `Returns:`
        bool: Returns true if the mode is compatoble with the state.
    '''
    if state['mode'] in [mode, 'both']:
        return True
    
    return False

# Returns the active prompt
def get_active_prompt(state: InputSchema) -> Optional[Prompt]:
    '''
    `get_active_prompt` returns the active prompt.
    
    `Args:`
        state (InputSchema): The state of the agent. Must have the key 'prompt_list' and 'active_prompt_index'.

    `Returns:`
        Prompt: The active prompt.
    '''
    if state['active_prompt_index'] >= len(state['prompt_list']):
        return None
    
    return state['prompt_list'][state['active_prompt_index']]

# # Returns the memory of the agent
# def get_memory() -> str:
#     '''
#     `get_memory` returns the memory of the agent.
    
#     `Returns:`
#         str: The memory of the agent formatted.
#     '''
#     with open(MEMORY_PATH, 'r') as f:
#         memory: dict = json.load(f)

#     return '\n'.join([f'{k}. {v}\n' for k, v in memory.items()])



''' Nodes '''
def extract_prompts(state: InputSchema) -> InputSchema:
    '''
    This node extracts the prompts from the code that should be created.
    '''
    print_function_name() if DEBUG else None

    try:
        prompts: List[str] = get_prompt_names(state['file_path'])

        print(f'{BLUE}[NODE] [INFO] [PROMPTS]{RESET} {prompts}') if DEBUG else None

        prompt_list: List[Prompt] = [Prompt(prompt_name= p) for p in prompts]
        return {
            'prompt_list': prompt_list,
            'active_prompt_index': 0 
        }
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state

def generate_prompt(state: InputSchema) -> InputSchema:
    '''
    This node generates the prompts.
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        comments: List[str] = get_active_prompt(state).filter_comments()
        
        prev = prompts.PREV_PROMPT.format(
            previous_prompt= get_active_prompt(state).suggested_prompt,
            user_comments= '\n'.join([f'- {c}' for c in comments]),
        ) if get_active_prompt(state).suggested_prompt else ''

        prompt = prompts.GENERATE_PROMPT_PROMPT.format(
            prompt_name= get_active_prompt(state).prompt_name,
            prev= prev,
            # memory= get_memory(),
            code= read_state_file(state)
        )

        response = safe_invoke(prompt_engineer, [SystemMessage(content= prompt)])

        # if will_tool_call([response]):
        #     return {'tool_call': [response]}

        generated_prompt = response.content
        draft_prompt: str = generated_prompt.split('<DRAFT_PROMPT_START>')[1].split('</DRAFT_PROMPT_END>')[0].strip()
        final_prompt: str = generated_prompt.split('<FINAL_PROMPT_START>')[1].split('</FINAL_PROMPT_END>')[0].strip()

        print(f'{GREEN}[NODE] [INFO] [DRAFT PROMPT]{RESET} {get_active_prompt(state).prompt_name} - DRAFT:\n{draft_prompt}') if DEBUG else None
        print(f'{GREEN}[NODE] [INFO] [FINAL PROMPT]{RESET} {get_active_prompt(state).prompt_name} - FINAL:\n{final_prompt}') if DEBUG else None

        get_active_prompt(state).suggested_prompt = final_prompt

        return {'tool_call': None}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return {'tool_call': None}

def review_prompt(state: InputSchema) -> InputSchema:
    '''
    This node reviews the prompt.
    '''
    print_function_name() if DEBUG else None

    try:
        print(f'{GREEN}[NODE] [INFO] [LATEST PROMPT]{RESET} {get_active_prompt(state).suggested_prompt}')
        comments: str = ''

        # Get the LLM to review the prompt
        if mode('llm', state):
            # prompt
            all_comments: List[str] = get_active_prompt(state).filter_comments()
            prompt = prompts.REVIEW_PROMPT_PROMPT.format(
                issues= '\n'.join([f'- {c}' for c in all_comments]),
                prompt_name= get_active_prompt(state).prompt_name,
                code= read_state_file(state),
                prompt= get_active_prompt(state).suggested_prompt,
            )
            # Ask the LLM to give a sample input for the prompt
            llm_answer: str = safe_invoke(prompt_reviewer, [SystemMessage(content= prompt)]).content
            print(f'{BLUE}[NODE] [INFO] [PROMPT REVIEW]{RESET} {get_active_prompt(state).prompt_name}:\n{llm_answer}') if DEBUG else None

            if llm_answer.split('# Issues')[-1].strip():
                comments += f'Review by Expert Reviewer: {llm_answer}\n\n'

        # Get the user to review the prompt
        if mode('user', state):
            user_answer: str = input(f'{GREEN}[NODE] [INFO] [INPUT] Please either accept or reject the prompt by giving a reason (y/reason) >{RESET} ')
        
            if user_answer:
                comments += f'Review by User: {user_answer}\n\n'
        
        # Add the comments
        get_active_prompt(state).add_comments(comments)

        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state
    
def get_response(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        # prompt
        prompt = prompts.FORMAT_PROMPT.format(
            prompt= get_active_prompt(state).suggested_prompt,
        )
        # Ask the LLM to give a sample input for the prompt
        to_format: Format = safe_invoke(formater, [SystemMessage(content= prompt)])
        get_active_prompt(state).set_format(to_format)
        print(f'{GREEN}[NODE] [INFO] [FORMAT]{RESET} {get_active_prompt(state).prompt_name}:\n{to_format}')

        # Format the suggested prompt and invoke a response
        llm_prompt: str = get_active_prompt(state).suggested_prompt.format(**to_format.format_dict)
        llm_prompt += prompts.TESTER_PROMPT
        llm_response: str = safe_invoke(tester, [SystemMessage(content= llm_prompt)]).content

        get_active_prompt(state).add_response(llm_response)

        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state
    
def review_response(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        print(f'{GREEN}[NODE] [INFO] [RESPONSE]{RESET} {get_active_prompt(state).latest_response}')
        comments: str = ''

        # Get the LLM to review the prompt
        if mode('llm', state):
            # prompt
            prompt = prompts.REVIEW_RESPONSE_PROMPT.format(
                code= read_state_file(state),
                prompt= get_active_prompt(state).suggested_prompt,
                format= '\n'.join([f'- {k}: {v}'for k, v in get_active_prompt(state).format.format_dict.items()]),
                llm_response= get_active_prompt(state).latest_response,
            )
            # Ask the LLM to give a sample input for the prompt
            llm_answer: str = safe_invoke(response_reviewer, [SystemMessage(content= prompt)]).content
            print(f'{BLUE}[NODE] [INFO] [RESPONSE REVIEW]{RESET} {get_active_prompt(state).prompt_name}:\n{llm_answer}') if DEBUG else None
        
            if llm_answer.split('# Issues')[-1].strip():
                comments += f'Review by Expert Reviewer: {llm_answer}\n\n'

        # Get the user to review the prompt
        if mode('user', state):
            user_answer = input(f'{GREEN}[NODE] [INFO] [INPUT] Please either accept or reject the response by giving a reason (y/reason) >{RESET} ')
            if user_answer:
                comments += f'Review by User: {user_answer}\n\n'

        # Add the comments
        get_active_prompt(state).add_comments(comments)

        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state

def next_prompt(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        # Go to the next prompt
        state['active_prompt_index'] += 1
        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state



''' Conditional Functions '''
# def tool_or_review(state: InputSchema) -> Literal['tool', 'review_prompt']:
#     print_function_name() if DEBUG else None

#     if state['tool_call']:
#         return 'tool'

#     return 'review_prompt'

def after_prompt_review(state: InputSchema) -> Literal['generate_prompt', 'get_response']:
    print_function_name() if DEBUG else None

    # If the prompt is not approved, go back and generate a new prompt
    if not get_active_prompt(state).is_approved():
        return 'generate_prompt'

    # Otherwise, get a response
    return 'get_response'
    
def after_response_review(state: InputSchema) -> Literal['generate_prompt', 'next_prompt', '__end__']:
    print_function_name() if DEBUG else None

    # If the response is not approved, go back and generate a new prompt
    if not get_active_prompt(state).is_approved():
        return 'generate_prompt'

    # If all good with the response, go to the next prompt
    # If there are no more prompts, end
    if state['active_prompt_index'] == len(state['prompt_list']) - 1:
        return '__end__'

    return 'next_prompt'

''' Graph '''
prompt_engineer_graph = StateGraph(InputSchema) # TODO: change

prompt_engineer_graph.add_node('extract_prompts', extract_prompts)
prompt_engineer_graph.add_node('generate_prompt', generate_prompt)
# prompt_engineer_graph.add_node('tool', ToolNode(prompt_engineer_tools, messages_key= 'tool_call'))
prompt_engineer_graph.add_node('review_prompt', review_prompt)
prompt_engineer_graph.add_node('get_response', get_response)
prompt_engineer_graph.add_node('review_response', review_response)
prompt_engineer_graph.add_node('next_prompt', next_prompt)

prompt_engineer_graph.add_edge(START, 'extract_prompts')
prompt_engineer_graph.add_edge('extract_prompts', 'generate_prompt')
prompt_engineer_graph.add_edge('generate_prompt', 'review_prompt')
# prompt_engineer_graph.add_conditional_edges(
#     'generate_prompt',
#     tool_or_review,
#     {
#         'tool': 'tool',
#         'review_prompt': 'review_prompt'
#     }
# )
# prompt_engineer_graph.add_edge('tool', 'generate_prompt')
prompt_engineer_graph.add_conditional_edges(
    'review_prompt',
    after_prompt_review,
    {   # Not needed, just for clarity
        'generate_prompt': 'generate_prompt',
        'get_response': 'get_response'
    }
)
prompt_engineer_graph.add_edge('get_response', 'review_response')
prompt_engineer_graph.add_conditional_edges(
    'review_response',
    after_response_review,
    {   # Not needed, just for clarity
        'generate_prompt': 'generate_prompt',
        'next_prompt': 'next_prompt',
        '__end__': END
    }
)
prompt_engineer_graph.add_edge('next_prompt', 'generate_prompt')

prompt_engineer_app = prompt_engineer_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(prompt_engineer_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/prompt_engineer_app.png', 'wb') as f:
        f.write(prompt_engineer_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'promptEngineer'
    os.environ['LANGSMITH_PROJECT'] = 'promptEngineer'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'promptEngineer',
            'run_name': 'promptEngineer',
            # 'thread_id': 'promptEngineer', 
        }
    }

    user = {
        'file_path': '../../creations/fitness_program_generator/fitness_program_generator.py',
        'mode': 'both'
    }
    response = prompt_engineer_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
