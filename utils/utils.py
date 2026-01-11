
from openai import APIConnectionError, InternalServerError, RateLimitError, BadRequestError, AuthenticationError
from pydantic_core._pydantic_core import ValidationError as PydanticValidationError
from json.decoder import JSONDecodeError

from langchain_core.messages import BaseMessage
from langgraph.prebuilt import tools_condition
from typing import Protocol, Any, Callable
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from time import sleep
from pathlib import Path
import inspect
import json 
import os
import re



load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent / '.env')

DEBUG = os.getenv('DEBUG')



# A constant for user approvals
USER_APPROVALS = ['y', 'ye', 'yea', 'yes', 'ok', 'okay', 'k', '', 'true', 'True']


''' Helpful General Functions '''
# Print the name of the function that is being executed
def print_function_name(colour: str= '\033[93m') -> None:
    '''
    `print_function_name` is a function that prints the name of the function that is being executed

    `Args:`
        colour (str): The colour of the text
    '''
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    filename = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
    print(f'\n{colour}[NODE]\033[0m {filename}/{func_name}')

# Check if the last message will or should call a tool
def will_tool_call(messages: list[BaseMessage], instruction_texts: list[str] = [], actually_called: bool= False) -> bool:
    '''
    Check if the last message will call a tool.

    ### Args:
    - `messages`: the list of messages up to now
        - **note:** remember to add the last message if the state is not updated yet
    - `instruction_text`: the text that the LLM is instructed to respond with when calling a tool
    - `actually_call`: whether it actually called the tool (by **only** searching the additional kwargs and tool_calls)
        - **default**: False

    ### Returns:
    - True if the last message will call a tool

    ### Tool Calls:
    - 'Will use tavily_search to gather context'
        - Skipped if actually_call is True
    - last_message.tool_calls exists and not empty
    - last_message.additional_kwargs.tool_calls exists and not empty
    - tools_condition(last_message) == tools
    '''
    last_message = messages[-1]
    instruction_text_bool = any(instruction_text in last_message.content.lower() for instruction_text in instruction_texts)
    return (
        # If actually_called is set, we should check wheather the last message is a tool call, not the content
        instruction_text_bool and not actually_called or 
        hasattr(last_message, 'tool_calls') and last_message.tool_calls or
        hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls', False) or
        tools_condition({'messages': messages}) == 'tools'
    )

# Function to parse tool arguments (when they come in additional_kwargs)
def parse_tool_arguments(args):
    # If the SDK already gave you a dict, use it
    if isinstance(args, dict):
        return args

    s = str(args).strip()

    # Normalize line endings
    s = s.replace('\r\n', '\n')
    # Replace any unescaped newlines with a space (JSON doesn't allow raw newlines)
    #    (?<!\\)\n  = a newline not preceded by a backslash
    s = re.sub(r'(?<!\\)\n', ' ', s)
    # Remove other control characters that are illegal in JSON
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', ' ', s)
    # Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Optional: last-resort escape of remaining bare backslashes before quote/newline
        s2 = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', s)
        return json.loads(s2)  # will raise again if truly broken
    


''' Helpful LLM Classes/Functions '''
# A class that extends the ChatOpenAI class, that automatically inputs the base url, api keys etc
class myChatOpenAI(ChatOpenAI):
    '''
    A class that extends the ChatOpenAI class, that automatically inputs some parametres
    '''
    def __init__(
            self, 
            base_url: str = 'https://openrouter.ai/api/v1', 
            api_key: str|None = None,
            model: str|None = None,
            temperature: float = 0.7,
            *args,
            **kwargs
        ):
        kwargs['base_url'] = base_url
        kwargs['api_key'] = api_key or os.getenv('OPENROUTER_API_KEY')
        kwargs['model'] = model or os.getenv('MODEL_NAME')
        kwargs['temperature'] = temperature
        super().__init__(*args, **kwargs)



# An exception that is raised when the LLM cannot be reached after too many tries
class TooManyTriesException(Exception): ...

# A Protocol class that requires the `invoke` method
class Invokable(Protocol):
    '''
    A Protocol class that requires the `invoke` method
    '''
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...

# A function that invokes an LLM and handles errors
def safe_invoke(llm: Invokable, *args, retry_interval: int = 6, max_retries: int = 7, raise_pydantic= False) -> BaseMessage:
    '''
    `safe_invoke` is a function that invokes an LLM and handles errors

    `Args:`
        llm (Invokable): The LLM to invoke
        *args (Any): The arguments to pass to the LLM
        retry_interval (int) = 5: The number of seconds to wait between retries
        max_retries (int) = 5: The maximum number of retries
    
    `Returns:`
        (BaseMessage) The result of the LLM invocation

    `Raises:` The errors raised by the LLM
        AuthenticationError: If the API key is invalid
        other Exceptions: If the LLM returns an error that is not catched

    '''
    retry_counter = 0
    while retry_counter < max_retries:
        try:
            return llm.invoke(*args)
        
        # Nothing to do, just raise the error
        except (AuthenticationError,) as e:
            raise e
        
        # Try again
        except PydanticValidationError as e:
            if raise_pydantic:
                raise e
            else:
                print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
                retry_counter += 1
                sleep(retry_interval)
        
        # Try again
        except (BadRequestError, APIConnectionError, InternalServerError, JSONDecodeError) as e:
            print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
            retry_counter += 1
            sleep(retry_interval)

        # Try again, with a little more handling
        except RateLimitError as e:
            error: dict = e.response.json()['error']
            # Check if it's a rate limit because of the server
            if 'is temporarily rate-limited upstream' in error.get('metadata', {}).get('raw', ''):
                cause = '(Upstream rate limit)'
                sleep_for = retry_interval

            # Or becuase of free-models-per-min
            elif 'Rate limit exceeded:' in error.get('message', '') and 'free-models-per-min' in error.get('message', ''):
                cause = '(Rate limit per minute exceeded)'
                # UFor a minute
                sleep_for = 60
            # TODO: for the day
            # Or fallback
            else: 
                cause = ''
                sleep_for = retry_interval
            
            print(f'RateLimitError {cause}, retrying in {sleep_for} seconds...') if DEBUG else None
            retry_counter += 1
            sleep(sleep_for)

        except ValueError as e:
            error_dict: dict = e.args[0] if e.args else None
            if error_dict.get('code', -1) == 500 and error_dict.get('message', '') == 'Internal Server Error':
                print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
                retry_counter += 1
                sleep(retry_interval)

        except KeyboardInterrupt as e:
            print(f'{e.__class__.__name__}, exiting...') if DEBUG else None
            exit()

        # Something went wrong, raise it
        except Exception as e:
            raise e
        
    raise TooManyTriesException(f'Could not get a response from the LLM after {max_retries} tries.')



CODE = """''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode

# Schema imports
from typing import TypedDict, Literal, List, Optional, Annotated, Union, Dict, Any
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
from creations.{root} import {directory_name}_prompts as prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\\033[94m' # INFO
RED = '\\033[91m' # ERR
GREEN = '\\033[92m' # REST
RESET = '\\033[0m'



print(f'\\n{{BLUE}}[AGENT] [INFO] [STARTUP]{{RESET}} {agent_name}') if DEBUG else None



\"\"\" Schemas \"\"\"
# TODO: Add Schemas (if needed)
''' General Schemas '''

''' Agent Schema '''
class AgentSchema(...): # TODO: Add parent class (e.g. MessagesState, BaseModel, TypedDict, etc.)
    ... # TODO: Add fields



''' Tools '''
# TODO: Add Tools (if needed)



''' LLM '''
# TODO: Add/Change LLMs (one per llm calling function) (if needed)
{llms}



''' Helpful Functions '''
# TODO: Add Helpful Functions (if needed)



''' Nodes '''
{nodes}



{conditional_functions}


''' Graph '''
{agent_name}_graph = StateGraph(AgentSchema)

{add_nodes}

{add_edges}

{agent_name}_app = {agent_name}_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image({agent_name}_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
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
        'recursion_limit': 100,
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

# A function that builds the workflow using langgraph
def build_workflow(bundle) -> str:
    '''
    `build_workflow` is a function that builds the workflow using langgraph

    `Args:`
        bundle (WorkflowBundle): The bundle to build the workflow from
    
    `Returns:`
        (str) The workflow
    '''
    def create_node(node) -> str:
        '''
        `create_node` is a function that creates a node

        `Args:`
            node (WorkflowNode): The node to create
        
        `Returns:`
            (str) The node
        '''
        if 'LLM' in node['description'] or 'SUBGRAPH' in node['description']:
            invokation = f'safe_invoke({node["name"]}_llm, [SystemMessage(content= prompt), ...])' if not node['subgraph_id'] else f'{node["subgraph_id"]}_app.invoke([SystemMessage(content= prompt), ...])'
            node_function = '\n'.join([
                f'def {node["name"]}(state: AgentSchema) -> AgentSchema:',
                f'    """ {node["description"]} """',
                 '    print_function_name()',
                 '    try:',
                 '        # TODO: <preprocess>',
                f'        prompt = prompts.{node["name"].upper()}_PROMPT.format(...) # TODO: <formatting>',
                f'        result = {invokation} # TODO: <inputs>',
                 '        # TODO: <postprocess>',
                 '        return ... # TODO: <return>',
                 '    except Exception as e:',
                 '        print(f\'{RED}[NODE] [ERR]{RESET}\', e) if DEBUG else None',
                 '        traceback.print_exc() if DEBUG else None',
                 '        # TODO: <error_handling> if needed',
                 '        return state',
                 ''
            ])
        else:
            node_function = '\n'.join([
                f'def {node["name"]}(state: AgentSchema) -> AgentSchema:',
                f'    """ {node["description"]} """',
                 '    print_function_name()',
                 '    try:',
                 '        # TODO: <process>',
                 '        return ... # TODO: <return>',
                 '    except Exception as e:',
                 '        print(f\'{RED}[NODE] [ERR]{RESET}\', e) if DEBUG else None',
                 '        traceback.print_exc() if DEBUG else None',
                 '        # TODO: <error_handling> if needed',
                 '        return state',
                 ''
            ])
        return node_function
    
    def create_edge(from_node: str, to_node: str, graph_name: str) -> str:
        '''
        `create_edge` is a function that creates an edge

        `Args:`
            from_node (str): The name of the source node
            to_node (str): The name of the target node
            graph_name (str): The name of the graph
        
        `Returns:`
            (str) The edge
        '''
        if 'start' == from_node.lower():
            from_node = '__start__' 
        if 'end' == to_node.lower():
            to_node = '__end__'
        edge_function = f'{graph_name}_graph.add_edge("{from_node}", "{to_node}")\n'
        return edge_function.replace('"__start__"', 'START').replace('"__end__"', 'END')
    
    def create_conditional_edge(from_node: str, to_nodes: list[str], graph_name: str) -> tuple[str, str]:
        '''
        `create_conditional_edge` is a function that creates a conditional edge

        `Args:`
            edge (WorkflowEdge): The edge to create
        
        `Returns:`
            (str) The edge
        '''
        upper_end: Callable[[str],str] = lambda s: s.upper() if s == 'end' else s
        to_nodes = [upper_end(to_node) for to_node in to_nodes]

        # Conditional function
        literals = ', '.join([f'"{to_node}"' for to_node in to_nodes])
        conditional_function = '\n'.join([
            f'def from_{from_node}_to(state: AgentSchema) -> Literal[{literals}]:',
             '    """ TODO: <docstring> """',
             '    print_function_name()',
             '    # TODO: <conditions>',
             ''
        ])
        
        # Edge function
        edge_map = '\n'.join([f'        "{to_node}": {to_node},' for to_node in to_nodes])
        edge_function = '\n'.join([
            f'{graph_name}_graph.add_conditional_edges(',
            f'    "{from_node}",' ,
            f'    from_{from_node}_to,',
             '    {   # Not needed just for clarity',
            f'{edge_map}',
             '    }',
            f')',
             ''
        ])
        return conditional_function, edge_function
    
    def build_graph(graph) -> dict:
        '''
        `build_graph` is a function that builds the graph

        `Args:`
            graph (WorkflowGraph): The graph to build
        
        `Returns:`
            (str) The graph
        '''
        # LLMs
        llms = '\n'.join([
            f'{node["name"]}_llm = myChatOpenAI(\n\ttemperature= 0\n) # TODO: <config> and change the temperature if needed' 
            for node in graph['nodes'] if 'LLM' in node['description']
        ])

        # Nodes
        nodes = '\n'.join([create_node(node) for node in graph['nodes'] if node['name'].lower() not in ('start', 'end')])

        # Edges
        edge_graph = {} # {source: [targets]}
        for edge in graph['edges']:
            if edge['source_name'] not in edge_graph:
                edge_graph[edge['source_name']] = []
            edge_graph[edge['source_name']].append(edge['target_name'])
            
        # Non Conditional Edges
        edges = {
            from_node: create_edge(from_node, to_nodes[0], graph['name']) 
            for from_node, to_nodes in edge_graph.items() if len(to_nodes) == 1
        }

        # Conditional edges
        conditional_edges = {
            from_node: create_conditional_edge(from_node, to_nodes, graph['name']) 
            for from_node, to_nodes in edge_graph.items() if len(to_nodes) > 1
        }

        # In order to keep the order
        all_edges = []
        for edge in edge_graph.keys():
            if edge in edges:
                all_edges.append(edges[edge])
            else:
                all_edges.append(conditional_edges[edge][1])

        cond = '\n'.join([conditional_edge[0] for conditional_edge in conditional_edges.values()])

        # Return
        return {
            'llms': llms,
            'nodes': nodes,
            'conditional_functions': f"''' Conditional Functions '''\n{cond}" if cond else '',
            'edges': ''.join(all_edges)
        }

    # Build for all graphs
    graphs = [bundle['root']] + list(bundle['subgraphs'].values())
    for graph in graphs:
        graph['name'] = graph['name'].replace(' ', '_').lower()
    
    args = {}
    code = {}
    for graph in graphs:
        built = build_graph(graph)
        args[graph['name']] = {
            'llms': built['llms'],
            'nodes': built['nodes'],
            'conditional_functions': built['conditional_functions'],
            'add_nodes': '\n'.join([
                f'{graph["name"]}_graph.add_node("{node["name"]}", {node["name"]})' 
                for node in graph['nodes'] if node['name'].lower() not in ('start', 'end')
            ]),
            'edges': built['edges']
        }

        code[graph['name']] = CODE.format(
            root= bundle['root']['name'],
            directory_name= graph['name'], # To camelCase from snake_case
            agent_name= graph['name'],
            llms= args[graph['name']]['llms'],
            nodes= args[graph['name']]['nodes'],
            conditional_functions= args[graph['name']]['conditional_functions'],
            add_nodes = args[graph['name']]['add_nodes'],
            add_edges = args[graph['name']]['edges']
        )

    return code



if __name__ == '__main__':
    dict_to_test = {
    "comments": "The user requested that 'generate_suggestions' and 'collect_feedback' nodes should loop back to themselves, and 'update_preferences' should be a tool used by 'collect_feedback'.",
    "root": {
        "type": "reactive_conversational",
        "name": "whatsapp_menu_suggestion_workflow",
        "nodes": [
            {
                "name": "start",
                "description": "Execution: CODE. Start node triggered by a user sending a WhatsApp message (text, image, link, or PDF). Initializes the conversation context and prepares to receive menu input.",
                "subgraph_id": None
            },
            {
                "name": "receive_menu_input",
                "description": "Execution: LLM+TOOLS. Extract text from menu input (photo, URL, or PDF) using OpenRouter vision API, requests/BeautifulSoup, and PyMuPDF/pdfplumber. Parse the extracted text into a structured format.",
                "subgraph_id": None
            },
            {
                "name": "generate_suggestions",
                "description": "Execution: LLM+TOOLS. Engage in free-form conversation to generate dish suggestions based on parsed menu items and user preferences stored in local JSON. Uses a tool called 'suggest_list(list: List)' to transition to the next node.",
                "subgraph_id": None
            },
            {
                "name": "send_suggestions",
                "description": "Execution: TOOLS. Send the ranked dish suggestions to the user via WhatsApp using the Twilio WhatsApp API.",
                "subgraph_id": None
            },
            {
                "name": "collect_feedback",
                "description": "Execution: LLM+TOOLS. Engage in free-form conversation to collect feedback on the meal and parse preference updates from the dialogue using an LLM. Uses a tool called 'update_preferences' to update user preferences.",
                "subgraph_id": None
            },
            {
                "name": "end",
                "description": "Execution: CODE. End node that signals the termination of the workflow when the user ends the conversation or no further input is received.",
                "subgraph_id": None
            }
        ],
        "edges": [
            {
                "source_name": "start",
                "target_name": "receive_menu_input",
                "description": "Transition triggered when a valid WhatsApp message is received from the user."
            },
            {
                "source_name": "receive_menu_input",
                "target_name": "generate_suggestions",
                "description": "Proceed to generate suggestions once the menu input has been successfully parsed."
            },
            {
                "source_name": "generate_suggestions",
                "target_name": "send_suggestions",
                "description": "Transition occurs when the LLM uses the tool 'suggest_list(list: List)' to generate and send suggestions."
            },
            {
                "source_name": "send_suggestions",
                "target_name": "collect_feedback",
                "description": "Engage in conversation to collect feedback after suggestions have been sent."
            },
            {
                "source_name": "collect_feedback",
                "target_name": "end",
                "description": "Terminate the workflow after feedback has been collected and preferences have been updated."
            },
            {
                "source_name": "generate_suggestions",
                "target_name": "generate_suggestions",
                "description": "Loop back to continue generating suggestions if more interaction is needed."
            },
            {
                "source_name": "collect_feedback",
                "target_name": "collect_feedback",
                "description": "Loop back to continue collecting feedback if more interaction is needed."
            }
        ],
        "description": "A reactive conversational workflow that processes user-provided menu inputs via WhatsApp, generates dish suggestions based on user preferences, sends these suggestions back to the user, collects feedback, and updates preferences. The workflow is triggered by user messages and operates in a streaming I/O mode."
    },
    "subgraphs": {}
}

    parsed = build_workflow(dict_to_test)

    def print_text_values(d, indent= 0):
        for k, v in d.items():
            if isinstance(v, dict):
                print("  " * indent + f"{k}:")
                print_text_values(v, indent + 1)
            else:
                print("  " * indent + f"{k}:")
                print(v)
                print()
    def stringify_text_values(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                # recursively convert nested dicts to stringified text
                result[k] = stringify_text_values(v)
            elif isinstance(v, list):
                # concatenate all list items (flatten into readable text)
                combined = ""
                for item in v:
                    if isinstance(item, dict):
                        combined += stringify_text_values(item)
                    else:
                        combined += str(item).rstrip() + "\n"
                result[k] = combined.strip() + "\n"
            else:
                result[k] = str(v).rstrip() + "\n"
        return result


    # print_text_values(parsed)
    code = stringify_text_values(parsed)
    parent_dir = dict_to_test['root']['name'].replace(' ', '_').lower()
    if not os.path.exists(f'../creations/{parent_dir}'):
        os.makedirs(f'../creations/{parent_dir}')
    for k, v in code.items():
        with open(f'../creations/{parent_dir}/{k}.py', 'w') as f:
            f.write(v)
        with open(f'../creations/{parent_dir}/{k}_prompts.py', 'w') as f:
            f.write('')
