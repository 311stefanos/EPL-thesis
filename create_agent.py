import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure the correct number of arguments are provided
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <agent_name>")
    print('    <agent_name> must be in snake_case.')
    sys.exit(1)

agent_name = sys.argv[1]

parent_path = Path(__file__).resolve().parent

# Convert agent_name to a suitable directory name
directory_name = agent_name.split('_')[0] + ''.join(x.title() for x in agent_name.split('_')[1:])

# Create the directory and a 'graphs' subdirectory
os.makedirs(parent_path / 'agents' / directory_name / 'graphs', exist_ok=True)

# List of files to create
FILE_NAMES = ['prompts.py', f'{agent_name}.py']

agent_file_text = f"""\"\"\"
- `author:` Stefanos Panteli
- `date:` {datetime.today().strftime('%Y-%m-%d')}
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.{directory_name}.{agent_name} import {agent_name}_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.{directory_name}.{agent_name} import {agent_name}_app
graph_input = {{ # TODO: add }}

response = {agent_name}_app.invoke(graph_input)

# response = {{ # TODO: add }}
```
\"\"\"



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
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments
from agents.{directory_name} import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\\033[94m' # INFO
RED = '\\033[91m' # ERR
MAGENTA = '\\033[95m' # TOOLS
GREEN = '\\033[92m' # REST
RESET = '\\033[0m'



print(f'\\n{{BLUE}}[AGENT] [INFO] [STARTUP]{{RESET}} {' '.join(x.title() for x in agent_name.split('_'))}') if DEBUG else None



\"\"\" Schemas \"\"\"
''' General Schemas '''

''' Input Schema '''

''' Intermediate Schemas '''

''' Output Schema '''



''' Tools '''



''' LLM '''
llm = myChatOpenAI(
    temperature= 0
)



''' Helpful Functions '''



''' Nodes '''



''' Conditional Functions '''



''' Graph '''
{agent_name}_graph = StateGraph() # TODO: change


{agent_name}_app = {agent_name}_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage({agent_name}_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/{agent_name}_app.png', 'wb') as f:
        f.write({agent_name}_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = '{directory_name}'
    os.environ['LANGSMITH_PROJECT'] = '{directory_name}'
    client = Client()

    config = {{
        'recursion_limit': 100, # TODO: change
        'configurable': {{
            'user_id': '{directory_name}',
            'run_name': '{directory_name}',
            'thread_id': '{directory_name}', 
        }}
    }}

    user = '' # TODO: add
    response = {agent_name}_app.invoke(user, config= config)

    print(f'{{BLUE}}[MAIN] [INFO]{{RESET}} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {{key}}: {{value}}')
"""

# Create each file in the new directory
for file in FILE_NAMES:
    with open(parent_path / 'agents' / directory_name / file, 'w') as f:
        if file == f'{agent_name}.py':
            f.write(agent_file_text)

        print(f'Created {file} succesfully')