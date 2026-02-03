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
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool

# Langgraph imports
from langgraph.constants import END, START
from langgraph.graph import StateGraph

# Schema imports
from typing import Tuple, TypedDict, Literal, List, Optional, Dict
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import json
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, USER_APPROVALS, read_state_file, clean_llm_output
from agents.promptEngineer import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')



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
    non_format_messages_list: List[BaseMessage] = Field(description= 'The list of non-format messages.', default= [])

class Prompt(BaseModel):
    prompt_name: str = Field(description= 'The name of the prompt.')
    suggested_prompt: str = Field(description= 'The suggested prompt.', default= '')
    necessary_code_changes: List[Tuple[str, str]] = Field(description= 'The necessary code changes.', default= [])

    format: Format = Field(description= 'The format of the prompt.', default= Format(format_dict= {}))

    user_comments: List[str] = Field(description= 'The user comments.', default= [])
    latest_response: str = Field(description= 'The latest response.', default= '')

    prompt_reviews: int = Field(description= 'The number of LLM reviews for the prompt.', default= 0)
    response_reviews: int = Field(description= 'The number of LLM reviews for the response.', default= 0)

    def reviewed(self, what: Literal['prompt', 'response']) -> None:
        if what == 'prompt':
            self.prompt_reviews += 1
        elif what == 'response':
            self.response_reviews += 1

    def reset_reviews(self, what: Literal['prompt', 'response']) -> None:
        if what == 'prompt':
            self.prompt_reviews = 0
        elif what == 'response':
            self.response_reviews = 0

    def can_review(self, what: Literal['prompt', 'response']) -> bool:
        prompt_max: int = 2
        response_max: int = 3
        if what == 'prompt':
            return self.prompt_reviews < prompt_max
        elif what == 'response':
            return self.response_reviews < response_max

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

        return not any(c.strip() not in USER_APPROVALS for c in last_comment.split('\n'))
    
    def filter_comments(self) -> List[str]:
        comments: List[str] = []
        for comment in self.user_comments:
            clear_comment: str = comment.replace('Review by Expert Reviewer: ', '')
            clear_comment = clear_comment.replace('Review by User: ', '')
            if any(c.strip() not in USER_APPROVALS for c in clear_comment.split('\n')):
                comments.append(comment)

        return comments

''' Input Schema '''
class InputSchema(TypedDict):
    file_path: str # The path to the file.
    prompt_list: Optional[List[Prompt]] # The list of prompts.
    active_prompt_index: Optional[int] # The active prompt index of the prompt list.

    error: Optional[bool] # If there was an error.

    mode: Literal['llm', 'user', 'both'] # The mode of the agent.



''' LLM '''
prompt_engineer = myChatOpenAI(
    temperature= 0.4
)

prompt_reviewer = myChatOpenAI(
    temperature= 0.25
)

formater = myChatOpenAI(
    temperature= 0.8,
    model= 'meta-llama/llama-3.3-70b-instruct:free'
).with_structured_output(Format)

tester = myChatOpenAI(
    temperature= 0.7
)

response_reviewer = myChatOpenAI(
    temperature= 0.25
)



''' Helpful Functions '''
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
    pattern = re.compile(r'^\s*\w+:?\s*\w+?\s*=\s*prompts\.([A-Z][A-Z0-9_]*)\b', re.MULTILINE)

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

# splits the LLMs response into sections
def split_prompt(content: str) -> Tuple[str, str, List[Tuple[str, str]]]:
    '''
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
    '''
    # Split it into sections
    thinking_process, other = content.split('#> Prompt\n')
    prompt, changes = other.split('#> Code Changes\n')

    # If no changes are needed, remove the changes section
    if '### old code' not in clean_llm_output(changes.lower()):
        changes = ''
    
    # If there are changes, split them
    if changes:
        changes = [c for c in changes.split('## Change') if c]

        # Parse them into old, new tuples
        changes = [
            (
                c.split('### Old Code\n')[1].split('### New Code\n')[0],
                c.split('### New Code\n')[1]
            )
            for c in changes
        ]

    # Clean the output and return
    return (
        clean_llm_output(thinking_process),
        clean_llm_output(prompt),
        [
            (clean_llm_output(old_code), clean_llm_output(new_code)) 
            for old_code, new_code in changes
            if old_code != new_code
        ] if changes else []
    )



''' Nodes '''
def extract_prompts(state: InputSchema) -> InputSchema:
    '''
    This node extracts the prompts from the code that should be created.
    '''
    print_function_name() if DEBUG else None

    try:
        # Extract the prompts from the code
        prompts: List[str] = get_prompt_names(state['file_path'])

        print(f'{BLUE}[NODE] [INFO] [PROMPTS]{RESET} {prompts}') if DEBUG else None

        # Create the prompts
        prompt_list: List[Prompt] = [Prompt(prompt_name= p) for p in prompts]
        return {
            'prompt_list': prompt_list,
            'active_prompt_index': 0,
            'error': False
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
        prompt = prompts.GENERATE_PROMPT_PROMPT.format(
            prompt_name= get_active_prompt(state).prompt_name,
            code= read_state_file(state)
        )

        # Create messages to send
        # An AIMess with the latest prompt
        # A HumanMessage with the comments on the latest prompt
        comments: List[str] = get_active_prompt(state).filter_comments()
        messages: List[BaseMessage] = [SystemMessage(content= prompt)]
        # Add the prompt
        if get_active_prompt(state).suggested_prompt:
            messages.append(AIMessage(content= get_active_prompt(state).suggested_prompt))
        # Add the comments
        if comments:
            # Parse the comments into a readbale string
            comment_content = '\n\n---\n\n'.join([
                f'- What lacks with the prompt (follow strictly):\n<COMMENT_START>\n{c}\n</COMMENT_END>\n' 
                for c in comments
            ])
            messages.append(HumanMessage(content= comment_content))
        # Add latest instructions
        if len(messages) > 1:
            messages[0].content +=  prompts.NEXT_MESSAGES_PROMPT

        # Invoke and parse
        response = safe_invoke(prompt_engineer, messages= messages).content
        thinking_process, suggested_prompt, code_changes = split_prompt(response)

        # Print
        code_changes_to_print = '\n'.join([
            f'## Change {i}\n### Old Code\n{old_code}\n### New Code\n{new_code}'
            for i, (old_code, new_code) in enumerate(code_changes)
        ]) if DEBUG else None
        print(f'{BLUE}[NODE] [INFO] [THINKING PROCESS]{RESET} {thinking_process}') if DEBUG else None
        print(f'{BLUE}[NODE] [INFO] [PROMPT]{RESET} {suggested_prompt}') if DEBUG else None
        print(f'{BLUE}[NODE] [INFO] [CODE CHANGES]{RESET} {code_changes_to_print}') if DEBUG else None

        # Update the state with the new prompt
        get_active_prompt(state).suggested_prompt = suggested_prompt
        get_active_prompt(state).necessary_code_changes = code_changes

        # No error
        state['error'] = False
        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        # Error
        return {'error': True}

def review_prompt(state: InputSchema) -> InputSchema:
    '''
    This node reviews the prompt.
    '''
    print_function_name() if DEBUG else None

    try:
        print(f'{GREEN}[NODE] [INFO] [LATEST PROMPT]{RESET} {get_active_prompt(state).suggested_prompt}')
        comments: str = ''

        # Get the LLM to review the prompt
        if mode('llm', state) and get_active_prompt(state).can_review('prompt'):
            # prompt
            all_comments: List[str] = get_active_prompt(state).filter_comments()
            prompt = prompts.REVIEW_PROMPT_PROMPT.format(
                issues= '\n'.join([f'- {c}' for c in all_comments]),
                prompt_name= get_active_prompt(state).prompt_name,
                code= read_state_file(state),
                prompt= get_active_prompt(state).suggested_prompt,
            )
            # Ask the LLM to give a sample input for the prompt
            llm_answer: str = safe_invoke(prompt_reviewer, messages= [SystemMessage(content= prompt)]).content
            print(f'{BLUE}[NODE] [INFO] [PROMPT REVIEW]{RESET} {get_active_prompt(state).prompt_name}:\n{llm_answer}') if DEBUG else None

            # Add the comments of the LLM
            if llm_answer.split('# Issues')[-1].strip():
                comments += f'Review by Expert Reviewer: {llm_answer}\n\n'

            # Increase the prompt review counter
            get_active_prompt(state).reviewed('prompt')

        # Get the user to review the prompt
        if mode('user', state):
            # Just ask the user
            user_answer: str = input(f'{GREEN}[NODE] [INFO] [INPUT] Please either accept or reject the prompt by giving a reason (y/reason) >{RESET} ')
            # Add the comments
            if user_answer:
                comments += f'Review by User: {user_answer}\n\n'
        
        # Add the comments to the state
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
            code= read_state_file(state),
            prompt= get_active_prompt(state).suggested_prompt
        )
        # Ask the LLM to give a sample input for the prompt
        to_format: Format = safe_invoke(formater, messages= [SystemMessage(content= prompt)])
        get_active_prompt(state).set_format(to_format)
        print(f'{GREEN}[NODE] [INFO] [FORMAT]{RESET} {get_active_prompt(state).prompt_name}:\n{json.dumps(to_format.format_dict, indent= 4)}')

        # Format the suggested prompt and invoke a response
        llm_prompt: str = get_active_prompt(state).suggested_prompt.format(**to_format.format_dict)
        llm_prompt += prompts.TESTER_PROMPT
        llm_response: str = safe_invoke(tester, messages= [SystemMessage(content= llm_prompt)]).content

        # Add the response
        get_active_prompt(state).add_response(llm_response)

        # Reset the prompt reviews
        get_active_prompt(state).reset_reviews('prompt')

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
        if mode('llm', state) and get_active_prompt(state).can_review('response'):
            # prompt
            prompt = prompts.REVIEW_RESPONSE_PROMPT.format(
                code= read_state_file(state),
                prompt= get_active_prompt(state).suggested_prompt,
                format= '\n'.join([f'- {k}: {v}'for k, v in get_active_prompt(state).format.format_dict.items()]),
                llm_response= get_active_prompt(state).latest_response,
            )
            # Ask the LLM to give a sample input for the prompt
            llm_answer: str = safe_invoke(response_reviewer, messages= [SystemMessage(content= prompt)]).content
            print(f'{BLUE}[NODE] [INFO] [RESPONSE REVIEW]{RESET} {get_active_prompt(state).prompt_name}:\n{llm_answer}') if DEBUG else None
        
            # Add the comments
            if llm_answer.split('# Issues')[-1].strip():
                comments += f'Review by Expert Reviewer: {llm_answer}\n\n'

            # Increase the response review counter
            get_active_prompt(state).reviewed('response')

        # Get the user to review the prompt
        if mode('user', state):
            # Just ask the user
            user_answer = input(f'{GREEN}[NODE] [INFO] [INPUT] Please either accept or reject the response by giving a reason (y/reason) >{RESET} ')
            # Add the comments 
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

def paste_prompts(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        # Format the prompts into a string
        prompts_str: str = '\n\n\n'.join([f'{prompt.prompt_name} = """\n{prompt.suggested_prompt}\n"""' for prompt in state['prompt_list']])
        
        prompt_file_path: str = state['file_path'].replace('.py', '_prompts.py')
        with open(prompt_file_path, 'w', encoding='utf-8') as f:
            f.write(prompts_str)

        # Make the code changes
        code: str = read_state_file(state)
        for prompt in state['prompt_list']:
            for (old, new) in prompt.necessary_code_changes:
                code = code.replace(old, new)
            
        with open(state['file_path'], 'w', encoding='utf-8') as f:
            f.write(code)
        
        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state

''' Conditional Functions '''
def generate_prompt_successfully(state: InputSchema) -> Literal['generate_prompt', 'review_prompt']:
    print_function_name() if DEBUG else None

    # Prompt did not generate
    if state['error']:
        return 'generate_prompt'
    
    # Prompt generated, go to review
    return 'review_prompt'

def after_prompt_review(state: InputSchema) -> Literal['generate_prompt', 'get_response']:
    print_function_name() if DEBUG else None

    # If the prompt is not approved, go back and generate a new prompt
    if not get_active_prompt(state).is_approved():
        return 'generate_prompt'

    # Otherwise, get a response
    return 'get_response'
    
def after_response_review(state: InputSchema) -> Literal['generate_prompt', 'next_prompt', 'paste_prompts']:
    print_function_name() if DEBUG else None

    # If the response is not approved, go back and generate a new prompt
    if not get_active_prompt(state).is_approved():
        return 'generate_prompt'

    # If all good with the response, go to the next prompt
    # If there are no more prompts, end
    if state['active_prompt_index'] == len(state['prompt_list']) - 1:
        return 'paste_prompts'

    return 'next_prompt'

''' Graph '''
prompt_engineer_graph = StateGraph(InputSchema) # TODO: change

prompt_engineer_graph.add_node('extract_prompts', extract_prompts)
prompt_engineer_graph.add_node('generate_prompt', generate_prompt)
prompt_engineer_graph.add_node('review_prompt', review_prompt)
prompt_engineer_graph.add_node('get_response', get_response)
prompt_engineer_graph.add_node('review_response', review_response)
prompt_engineer_graph.add_node('next_prompt', next_prompt)
prompt_engineer_graph.add_node('paste_prompts', paste_prompts)

prompt_engineer_graph.add_edge(START, 'extract_prompts')
prompt_engineer_graph.add_edge('extract_prompts', 'generate_prompt')
prompt_engineer_graph.add_conditional_edges(
    'generate_prompt',
    generate_prompt_successfully,
    {   # Not needed, just for clarity
        'generate_prompt': 'generate_prompt',
        'review_prompt': 'review_prompt'
    }
)
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
        'paste_prompts': 'paste_prompts'
    }
)
prompt_engineer_graph.add_edge('next_prompt', 'generate_prompt')
prompt_engineer_graph.add_edge('paste_prompts', END)

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
        'file_path': '../../creations/menu_recommendation_workflow/menu_recommendation_workflow.py',
        'mode': 'both'
    }
    response = prompt_engineer_app.invoke(user, config= config)

    # print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    # if DEBUG:
    #     for key, value in response.items():
    #         print(f'    {key}: {value}')
