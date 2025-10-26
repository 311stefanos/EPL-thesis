"""
- `author:` Stefanos Panteli
- `date:` 2025-10-25
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.codeInterpreter.codeInterpreter import code_interpreter_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.codeInterpreter.code_interpreter import code_interpreter_app
graph_input = { # TODO: add }

response = code_interpreter_app.invoke(graph_input)

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

# Schema imports
from typing import TypedDict, Literal, List, Optional, Annotated
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name
from agents.codeInterpreter import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Code Interpreter') if DEBUG else None



""" Schemas """
''' General Schemas '''

''' Input Schema '''
class CodeInterpreterInput(BaseModel):
    code: str = Field(
        description= 'The code to interpret.',
    )

''' Intermediate Schemas '''

''' Output Schema '''



''' Tools '''



''' LLM '''
interpreter = myChatOpenAI(
    temperature= 0
)



''' Helpful Functions '''



''' Nodes'''
def code_interpreter(code: str) -> str:
    ...


''' Conditional Functions '''



''' Graph '''
code_interpreter_graph = StateGraph() # TODO: change


code_interpreter_app = code_interpreter_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(code_interpreter_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/code_interpreter_app.png', 'wb') as f:
        f.write(code_interpreter_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'codeInterpreter'
    os.environ['LANGSMITH_PROJECT'] = 'codeInterpreter'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'codeInterpreter',
            'run_name': 'codeInterpreter',
            'thread_id': 'codeInterpreter', 
        }
    }

    user = '' # TODO: add
    response = code_interpreter_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
