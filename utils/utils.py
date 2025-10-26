
from openai import APIConnectionError, InternalServerError, RateLimitError, BadRequestError, AuthenticationError
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from typing import Protocol, Any
from dotenv import load_dotenv
from time import sleep, time
from pathlib import Path
import inspect
import os



load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent / '.env')

DEBUG = os.getenv('DEBUG')



# A constant for user approvals
USER_APPROVALS = ['y', 'ye', 'yea', 'yes', 'ok', 'okay', 'k', '']


''' Helpful General Functions '''
# Print the name of the function that is being executed
def print_function_name():
    '''
    `print_function_name` is a function that prints the name of the function that is being executed

    `Returns:`
        (str) The name of the function that is being executed
    '''
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    filename = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
    print(f'\n\033[93m[NODE]\033[0m {filename}/{func_name}')

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
def safe_invoke(llm: Invokable, *args, retry_interval: int = 5, max_retries: int = 5) -> BaseMessage:
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
        except (BadRequestError, APIConnectionError, InternalServerError) as e:
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
                # Until the next minute
                reset_time = int(error['metadata']['headers']['X-RateLimit-Reset'])
                sleep_for = (reset_time - int(time() * 1000)) / 1000
            # TODO: for the day
            # Or fallback
            else: 
                cause = ''
                sleep_for = retry_interval
            
            print(f'RateLimitError {cause}, retrying in {sleep_for} seconds...') if DEBUG else None
            retry_counter += 1
            sleep(sleep_for)

        # Something went wrong, raise it
        except Exception as e:
            raise e
        
    raise TooManyTriesException(f'Could not get a response from the LLM after {max_retries} tries.')



CODE = """
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
# TODO: Add Schemas
''' General Schemas '''

''' Input Schema '''

''' Intermediate Schemas '''

''' Output Schema '''



''' Tools '''
# TODO: Add Tools



''' LLM '''
# TODO: Add/Change LLMs (one per llm calling function)
{llms}



''' Helpful Functions '''
# TODO: Add Helpful Functions



''' Nodes'''
{nodes}



''' Conditional Functions '''
{conditional_functions}


''' Graph '''
{agent_name}_graph = StateGraph() # TODO: change

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
            invokation = f'safe_invoke({node["name"]}_llm, [SystemMessage(content= prompt)<, anything else>])' if not node['subgraph_id'] else f'{node["subgraph_id"]}_app.invoke([SystemMessage(content= prompt)<, anything else>])'
            node_function = '\n'.join([
                f'# TODO: <comment>',
                f'def {node["name"]}(state):',
                f'    """ {node["description"]} """',
                 '    print_function_name()',
                 '    try:',
                 '        # TODO: <preprocess>',
                f'        prompt = prompts.{node["name"].upper()}_PROMPT.format(<formatting>) # TODO: <formatting>',
                f'        result = {invokation} # TODO: <inputs>',
                 '        # TODO: <postprocess>',
                 '        return <return> # TODO: <return>',
                 '    except Exception as e:',
                 '        print(f\'{RED}[NODE] [ERR]{RESET}\', e) if DEBUG else None',
                 '        traceback.print_exc() if DEBUG else None',
                 '        # TODO: <error_handling> if needed',
                 '        return state',
                 ''
            ])
        else:
            node_function = '\n'.join([
                f'# TODO: <comment>',
                f'def {node["name"]}(state):',
                f'    """ {node["description"]} """',
                 '    print_function_name()',
                 '    try:',
                 '        # TODO: <process>',
                 '        return <return> # TODO: <return>',
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
        for to_node in to_nodes:
            if to_node == 'end':
                to_node = '__end__'

        # Conditional function
        literals = ', '.join([f'"{to_node}"' for to_node in to_nodes])
        conditional_function = '\n'.join([
            f'# TODO: <comment>',
            f'def from_{from_node}_to(state) -> Literal[{literals}]:',
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
            f'{node["name"]}_llm = myChatOpenAI(\n\ttemperature= 0\n) # TODO: <config>' 
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

        # Return
        return {
            'llms': llms,
            'nodes': nodes,
            'conditional_functions': '\n'.join([conditional_edge[0] for conditional_edge in conditional_edges.values()]),
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
    "comments": "Added explicit start and end nodes to the root workflow as requested. The start node is triggered by either schedule (every day at 09:00) or incoming WhatsApp message. The end node is reached after both the periodic and conversational workflows have completed once.",   
    "root": {
        "type": "hybrid",
        "name": "WhatsApp Review Processing",
        "nodes": [
            {
                "name": "start",
                "description": "Execution: CODE. Trigger both periodic and conversational workflows. Trigger: schedule (every day at 09:00) and incoming WhatsApp message.",
                "subgraph_id": None
            },
            {
                "name": "call_periodic_workflow",
                "description": "Execution: CODE. Execute the periodic review fetching and storing workflow.",
                "subgraph_id": "periodic_workflow"
            },
            {
                "name": "call_conversational_workflow",
                "description": "Execution: CODE. Execute the conversational agent workflow for WhatsApp messages.",
                "subgraph_id": "conversational_workflow"
            },
            {
                "name": "end",
                "description": "Execution: CODE. End of the root workflow.",
                "subgraph_id": None
            }
        ],
        "edges": [
            {
                "source_name": "start",
                "target_name": "call_periodic_workflow",
                "description": "Guard: always true. Trigger periodic workflow."
            },
            {
                "source_name": "start",
                "target_name": "call_conversational_workflow",
                "description": "Guard: always true. Trigger conversational workflow."
            },
            {
                "source_name": "call_periodic_workflow",
                "target_name": "end",
                "description": "Guard: always true. End of periodic workflow done."
            },
            {
                "source_name": "call_conversational_workflow",
                "target_name": "end",
                "description": "Guard: always true. End of conversational workflow done."
            }
        ],
        "description": "Hybrid workflow with two independent flows: periodic batch task (scheduled daily at 09:00) and reactive conversational agent for incoming WhatsApp messages. The start node is triggered by either schedule or message, and the end node is reached after both workflows complete."
    },
    "subgraphs": {
        "periodic_workflow": {
            "type": "linear_pipeline",
            "name": "Periodic Review Workflow",
            "nodes": [
                {
                    "name": "start",
                    "description": "Execution: CODE. Begin the periodic review workflow.",
                    "subgraph_id": None
                },
                {
                    "name": "periodic_review_fetcher",
                    "description": "Execution: TOOLS. Fetch reviews using third-party APIs for all place IDs.",
                    "subgraph_id": None
                },
                {
                    "name": "database_updater",
                    "description": "Execution: CODE. Store reviews in SQLite.",
                    "subgraph_id": None
                },
                {
                    "name": "end",
                    "description": "Execution: CODE. End of the periodic workflow.",
                    "subgraph_id": None
                }
            ],
            "edges": [
                {
                    "source_name": "start",
                    "target_name": "periodic_review_fetcher",
                    "description": "Guard: always true."
                },
                {
                    "source_name": "periodic_review_fetcher",
                    "target_name": "database_updater",
                    "description": "Guard: reviews fetched successfully."
                },
                {
                    "source_name": "database_updater",
                    "target_name": "end",
                    "description": "Guard: reviews stored successfully."
                }
            ],
            "description": "Automated process to fetch reviews from third-party APIs and store them in SQLite, run daily at 09:00."
        },
        "conversational_workflow": {
            "type": "reactive_conversational",
            "name": "Conversational WhatsApp Agent",
            "nodes": [
                {
                    "name": "start",
                    "description": "Execution: CODE. Capture incoming WhatsApp message.",
                    "subgraph_id": None
                },
                {
                    "name": "message_parser",
                    "description": "Execution: LLM. Analyze user intent (query/recommendation request).",
                    "subgraph_id": None
                },
                {
                    "name": "database_query",
                    "description": "Execution: CODE. Retrieve relevant review data from SQLite based on parsed intent.",
                    "subgraph_id": None
                },
                {
                    "name": "response_generator",
                    "description": "Execution: LLM. Generate natural language response/recommendation.",
                    "subgraph_id": None
                },
                {
                    "name": "whatsapp_response_sender",
                    "description": "Execution: TOOLS. Send WhatsApp reply to user.",
                    "subgraph_id": None
                },
                {
                    "name": "end",
                    "description": "Execution: CODE. End of the conversational flow.",
                    "subgraph_id": None
                }
            ],
            "edges": [
                {
                    "source_name": "start",
                    "target_name": "message_parser",
                    "description": "Guard: message captured successfully. Pass message to parser."
                },
                {
                    "source_name": "message_parser",
                    "target_name": "database_query",
                    "description": "Guard: intent requires data. Pass parsed intent to query."
                },
                {
                    "source_name": "database_query",
                    "target_name": "response_generator",
                    "description": "Guard: data retrieved. Pass data to response generator."
                },
                {
                    "source_name": "response_generator",
                    "target_name": "whatsapp_response_sender",
                    "description": "Guard: response generated. Pass response to sender."
                },
                {
                    "source_name": "whatsapp_response_sender",
                    "target_name": "end",
                    "description": "Guard: response sent. End conversation flow."
                }
            ],
            "description": "Reactive conversational agent that handles incoming WhatsApp messages and provides review-based responses. Trigger: incoming WhatsApp message. I/O Mode: streaming."
        }
    }
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
