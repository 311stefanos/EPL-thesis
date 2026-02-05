from typing import Callable, List
from pydantic import BaseModel
import os


_CODE = """''' Imports '''
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
import json
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS, read_state_file, clean_llm_output
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

{agent_name}_app = {agent_name}_graph.compile({memory})



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
def _build_workflow(bundle) -> str:
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
            invokation = f'safe_invoke({node["name"]}_llm, messages= [SystemMessage(content= prompt), ...])' if not node.get('subgraph_id') else f'{node["subgraph_id"]}_app.invoke(...)'
            node_function = '\n'.join([
                f'def {node["name"]}(state: AgentSchema) -> AgentSchema:',
                f'    """ {node["description"]} """',
                 '    print_function_name()',
                 '    try:',
                 '        # TODO: <preprocess>',
                 '        ... # TODO: <make format arguments readable>',
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
        # Functions to parse the start and end node so they can be used in the conditional functions with the right names
        to_end: Callable[[str],str] = lambda s: '__end__' if s == 'end' else s
        from_start: Callable[[str], str] = lambda s: '__start__' if s == 'start' else s

        to_nodes = [to_end(to_node) for to_node in to_nodes]

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
        edge_map = '\n'.join([f'        "{to_node}": "{to_node}",' for to_node in to_nodes])
        edge_function = '\n'.join([
            f'{graph_name}_graph.add_conditional_edges(',
            f'    "{from_start(from_node)}",' ,
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

        code[graph['name']] = _CODE.format(
            root= bundle['root']['name'],
            directory_name= graph['name'], # To camelCase from snake_case
            agent_name= graph['name'],
            llms= args[graph['name']]['llms'],
            nodes= args[graph['name']]['nodes'],
            conditional_functions= args[graph['name']]['conditional_functions'],
            add_nodes = args[graph['name']]['add_nodes'],
            add_edges = args[graph['name']]['edges'],
            memory= 'checkpointer= MemorySaver()' if graph['memory'] else ''
        )

    return code




def _stringify_text_values(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                # recursively convert nested dicts to stringified text
                result[k] = _stringify_text_values(v)
            elif isinstance(v, list):
                # concatenate all list items (flatten into readable text)
                combined = ""
                for item in v:
                    if isinstance(item, dict):
                        combined += _stringify_text_values(item)
                    else:
                        combined += str(item).rstrip() + "\n"
                result[k] = combined.strip() + "\n"
            else:
                result[k] = str(v).rstrip() + "\n"
        return result

def create_file(workflow_dict: dict) -> List[str]:
    '''
    `create_file` is a function that creates the file

    `Args:`
        workflow_dict (dict): The workflow dictionary

    `Returns:`
        (List[str]) The names of the created files (non prompt files)
    '''
    if 'workflow' in workflow_dict:
        workflow_dict: dict = workflow_dict['workflow']

    if isinstance(workflow_dict, BaseModel):
        workflow_dict: dict = workflow_dict.model_dump()
        

    parsed = _build_workflow(workflow_dict)

    code = _stringify_text_values(parsed)
    parent_dir = workflow_dict['root']['name'].replace(' ', '_').lower()

    if not os.path.exists(f'../creations/{parent_dir}'):
        os.makedirs(f'../creations/{parent_dir}')

    created_agents: List[str] = []
    for k, v in code.items():
        with open(f'../creations/{parent_dir}/{k}.py', 'w') as f:
            f.write(v)
            created_agents.append(f'../creations/{parent_dir}/{k}.py')

        with open(f'../creations/{parent_dir}/{k}_prompts.py', 'w') as f:
            f.write('')

    return created_agents



if __name__ == '__main__':
    dict_to_test = {
        "comments": "The workflow has been adjusted to parse the menu at the start, with the start node connecting to both the parse and chat nodes. This ensures menu data is processed early in the workflow.",
        "root": {
            "description": "A reactive conversational workflow that processes user messages to provide menu recommendations based on user preferences and feedback. The workflow starts by parsing the menu data and then processes the user message to generate recommendations.",
            "edges": [
                {
                    "description": "Transition to parse the menu data before processing the user message.",
                    "source_name": "start",
                    "target_name": "parse_menu"
                },
                {
                    "description": "Transition to the chat node to process the user message and generate a response.",
                    "source_name": "start",
                    "target_name": "chat"
                },
                {
                    "description": "Transition to the chat node after parsing the menu data.",
                    "source_name": "parse_menu",
                    "target_name": "chat"
                },
                {
                    "description": "Transition to the end node to terminate the workflow run after processing the user message.",
                    "source_name": "chat",
                    "target_name": "end"
                }
            ],
            "memory": True,
            "name": "menu_recommendation_workflow",
            "nodes": [
                {
                    "description": "Execution: CODE. Route based on the mode/next_action stored in state and initiate menu parsing.",
                    "name": "start"
                },
                {
                    "description": "Execution: CODE. Parse the menu data provided by the user, supporting photo, link, or text formats.",
                    "name": "parse_menu"
                },
                {
                    "description": "Execution: LLM+TOOLS. Process the user message, retrieve user preferences using user_preferences_manager, generate recommendations, and handle feedback using feedback_processor.",
                    "name": "chat"
                },
                {
                    "description": "Execution: CODE. Terminate the workflow run.",
                    "name": "end"
                }
            ],
            "type": "reactive_conversational"
        },
        "subgraphs": {}
    }

    parsed = _build_workflow(dict_to_test)

    code = _stringify_text_values(parsed)
    parent_dir = dict_to_test['root']['name'].replace(' ', '_').lower()
    if not os.path.exists(f'../creations/{parent_dir}'):
        os.makedirs(f'../creations/{parent_dir}')
    for k, v in code.items():
        with open(f'../creations/{parent_dir}/{k}.py', 'w') as f:
            f.write(v)
        with open(f'../creations/{parent_dir}/{k}_prompts.py', 'w') as f:
            f.write('')
