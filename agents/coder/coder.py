"""
- `author:` Stefanos Panteli
- `date:` 2025-11-04
- `description:` The coder agent is used to implement any function. It can return the implemented code, along side possible function proposals to aid the implementation process.

## How to use
1. Import the app. (`from agents.coder.coder import coder_app`)
2. Input a dict with the following keys:
    - `messages: List[Message]`: A list of messages. Should provide an empty list.
    - `file_path: str`: The path of the file to implement the function.
    - `function_name: str`: The name of the function to implement.
    - `software_engineer_instructions: str`: The instructions from the software engineer.
    - `previous_outputs: List[OutputSchema]`: A list of previous outputs. Used by the Software Engineer.
    - `comments: List[str]`: A list of comments. Used by the Software Engineer.
3. Invoke the app.
4. Get the output dict with the following keys:
    - `code: str`: The implemented code of the function.
    - `proposals: List[FunctionProposal]`: A list of function proposals.
        - `function_type: Literal["helper_function", "tool"]`
        - `function_name: str`
        - `docstring: str`
        - `function_arguments: List[Argument]`
            - `name: str`
            - `type: str`
        - `output: str`
        - `justification: str`

## Usage
```python
from agents.coder.coder import coder_app
graph_input = {
    'messages': [],
    'file_path': '../../creations/fitness_program_generator/fitness_program_generator.py',
    'function_name': 'macro_calculation',
    'software_engineer_instructions': 'Implement the macro_calculation function.',
    'previous_outputs': [],
    'comments': []
}

response = coder_app.invoke(graph_input)

# response = {
#       'code': '<implemented code>',
#       'proposals': []
# }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage, ToolMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode

# Schema imports
from typing import Literal, List, Optional, Annotated, Union
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, parse_tool_arguments
from agents.coder import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Coder') if DEBUG else None



""" Schemas """
''' General Schemas '''     
# Used at the output_tool. The agent requests from the Software Engineer a function to implement.
class FunctionProposal(BaseModel):
    class Argument(BaseModel):
        name: str = Field(description= 'The name of the argument.')
        type: str = Field(description= 'The type of the argument.')

        def __str__(self):
            return f'{self.name}: {self.type}'
   
    function_type: Literal['helper_function', 'tool'] = Field(description= 'The type of the function you need.')
    function_name: str = Field(description= 'The name of the function.')
    docstring: str = Field(description= 'The docstring of the function.')
    function_arguments: List[Argument] = Field(description= 'The arguments of the function.')
    output: str = Field(description= 'The return type of the function.')
    justification: str = Field(description= 'The justification of the function.')

    def __str__(self):
        if self.function_type == 'tool': 
            tool = '@tool\n'
        else:
            tool = ''
        docstring = self.docstring.replace('\n', '\n\t')
        arguments = ', '.join([str(arg) for arg in self.function_arguments])
        return f'{tool}def {self.function_name}({arguments}) -> {self.output}:\n\t"""\n\t{docstring}\n\t"""\n\t...'

''' Input Schema '''
class InputSchema(MessagesState):
    file_path: str = Field(description= 'The path to the file.')
    function_name: str = Field(description= 'The name of the function.')
    software_engineer_instructions: str = Field(description= 'The instructions from the software engineer.')

    previous_outputs: Annotated[List['OutputSchema'], add] = Field(description= 'The previous outputs you provided.')
    comments: Annotated[List[str], add] = Field(description= 'The comments the software engineer provided.')

''' Output Schema '''
class OutputSchema(BaseModel):
    code: str = Field(description= 'The implemented code of the function.')
    proposals: Optional[List[FunctionProposal]] = Field(description= 'The requested proposals.')



''' Tools '''
tavily_search = TavilySearch(
    tavily_api_key= os.getenv('TAVILY_API_KEY'),
    search_depth= "advanced",
    max_results= 5,
    include_answer= True
).as_tool()

# Output tool. Called when the agent is done, and ends the workflow.
@tool(description= 'Submit the final single-function implementation (and optional function/tool proposals)')
def output_tool(code: str, proposals: Optional[Union[List[FunctionProposal],str]] = None) -> OutputSchema:
    '''
    Submit the final single-function implementation (and optional function/tool proposals).
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None
    if proposals and proposals in ['None', '[]']:
        proposals = None
        
    return OutputSchema(code= code, proposals= proposals)

# List of tools
tools = [tavily_search, output_tool] # , github_tool, stackoverflow_tool
# Dictionary of tools: tool name -> tool
tools_by_name = {tool.name: tool for tool in tools}



''' LLM '''
brainstormer = myChatOpenAI(
    temperature= 0.8,
    model= 'qwen/qwen3-coder:free'
    # 'qwen/qwen3-235b-a22b:free'
)

coder = myChatOpenAI(
    temperature= 0.5,
    # model= 'qwen/qwen3-235b-a22b:free',
    # 'qwen/qwen3-235b-a22b:free'
).bind_tools(tools + [output_tool])



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



''' Nodes '''
# This nodes is called first, and the agent brainstorms some solutions.
def solution_brainstorm_node(state: InputSchema) -> InputSchema:
    '''
    This node is used to think possible solutions to the problem.
    '''
    print_function_name() if DEBUG else None
    # Remove all prior messages, not from this invokation.
    remove_messages: List[RemoveMessage] = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    
    try:
        # prompt
        history = '\n\n---\n\n'.join([
            f'Output number {i})\n{po}\n\nComment from software engineer:\n{comm}' 
            for i, (po, comm) in enumerate(zip(state['previous_outputs'], state['comments']), start= 1)
        ])

        prompt = prompts.SOLUTION_BRAINSTORM_PROMPT.format(
            code= read_state_file(state),
            function_name= state['function_name'],
            special_instructions= state['software_engineer_instructions'],
            history= history
        )

        # call the LLM
        results = safe_invoke(brainstormer, [SystemMessage(content= prompt)]).content

        print(f'{BLUE}[NODE] [INFO] [RESULTS]{RESET} {results}') if DEBUG else None

        return {'messages': remove_messages + [AIMessage(content= results)]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return {'messages': remove_messages}

# This node uses different tools to implement the function.
def coder_node(state: InputSchema) -> InputSchema:
    '''
    This node calls the coder agent to implement the function. It has access to tools, and prior knowledge.
    '''
    print_function_name() if DEBUG else None

    last_message = state['messages'][-1]
    if isinstance(last_message, ToolMessage):
        print(f'{BLUE}[NODE] [INFO] [TOOL CALL]{RESET} {last_message}') if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([
            f'Output number {i})\n{po}\n\nComment from software engineer:\n{comm}' 
            for i, (po, comm) in enumerate(zip(state['previous_outputs'], state['comments']), start= 1)
        ])

        prompt = prompts.CODE_PROMPT.format(
            code= read_state_file(state),
            function_name= state['function_name'],
            special_instructions= state['software_engineer_instructions'],
            history= history,
            tool_history= [mes.pretty_repr() for mes in state.get('messages', [])]
        )

        # call the LLM
        response = safe_invoke(coder, [SystemMessage(content= prompt)])
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        # return
        return {'messages': response}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return {'messages': AIMessage(content= '')}

# This node executes the output tool, and ends the workflow
def output_node(state: InputSchema) -> OutputSchema:
    '''
    This node executes the output tool, and ends the workflow.
    '''
    print_function_name() if DEBUG else None

    try:
        # Get the last message, and extract the tool calls
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            from_kwargs = False
            tool_calls = last_message.tool_calls
        else:
            from_kwargs = True
            tool_calls = last_message.additional_kwargs.get('tool_calls', [])

        # Get the first tool call (and probably the only one)
        tool_call = tool_calls[0]
        # Get the tool and arguments
        if from_kwargs:
            tool_call = tool_call['function']
        tool = tools_by_name[tool_call['name']]

        args = tool_call.get('args', {}) or tool_call.get('arguments', {})
        # Parse the tool arguments if needed.
        if isinstance(args, str):
            args = parse_tool_arguments(args)                

        # Add the proposals key if the LLM skipped it.
        if 'proposals' not in args:
            args['proposals'] = []

        # Execute the tool
        output: OutputSchema = tool.invoke(args)

        print(f'{BLUE}[NODE] [INFO] [OUTPUT]{RESET} {output}') if DEBUG else None
        
        return output

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return OutputSchema(code= '', proposals= None)



''' Conditional Functions '''
# This node is called to decide the next step of the coder_node
def tool_node_or_end(state: InputSchema) -> Literal['tool_node', 'output_node', 'coder_node']:
    '''
    This node is called to decide the next step of the coder_node.
    Possible return values: 
        - 'tool_node': When the agent called a tool, except the output tool.
        - 'output_node': When the agent called the output tool.
        - 'coder_node': When the agent did not call a tool, or an error occured.
    '''
    print_function_name() if DEBUG else None

    try:
        # Get the last message and extract the tool calls
        last_message = state['messages'][-1]
        tool_call = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', [])
        if tool_call:
            tool_call = tool_call[0]
        # If the last message is not a tool call, go back to the coder node
        else:
            return 'coder_node'
        
        # Get the tool function and name
        if 'function' in tool_call:
            tool_call = tool_call['function']

        # If the tool is the output tool, go to the output node
        if tool_call['name'] == 'output_tool':
            return 'output_node'
        # Else, go to the tool node
        else:
            return 'tool_node'

    # If an error occured, go to the coder node
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return 'coder_node'


''' Graph '''
coder_graph = StateGraph(InputSchema, output_schema= OutputSchema) # TODO: change

coder_graph.add_node('solution_brainstorm_node', solution_brainstorm_node)
coder_graph.add_node('coder_node', coder_node)
coder_graph.add_node('tool_node', ToolNode(tools)) # ToolNode with all the tools excluding the output tool
coder_graph.add_node('output_node', output_node)

coder_graph.add_edge(START, 'solution_brainstorm_node')
coder_graph.add_edge('solution_brainstorm_node', 'coder_node')
coder_graph.add_conditional_edges(
    'coder_node', 
    tool_node_or_end,
    {   # Not needed, for clarity
        'tool_node': 'tool_node',
        'output_node': 'output_node',
        'coder_node': 'coder_node'
    }    
)
coder_graph.add_edge('tool_node', 'coder_node')
coder_graph.add_edge('output_node', END)

coder_app = coder_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(coder_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/coder_app.png', 'wb') as f:
        f.write(coder_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'coder'
    os.environ['LANGSMITH_PROJECT'] = 'coder'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'coder',
            'run_name': 'coder',
            'thread_id': 'coder', 
        }
    }

    user = InputSchema(
        messages= [],
        file_path= '../../creations/fitness_program_generator/fitness_program_generator.py',
        function_name= 'macro_calculation',
        software_engineer_instructions= 'Implement the macro_calculation function.',
        previous_outputs= [],
        comments= []
    )
    response = coder_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
