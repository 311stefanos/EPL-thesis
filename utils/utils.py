
from openai import (OpenAIError, APIConnectionError, InternalServerError,
                    RateLimitError, BadRequestError, AuthenticationError)
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing import Protocol, Any
from dotenv import load_dotenv
from time import sleep, time
from pathlib import Path
import inspect
import os

load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent / '.env')

DEBUG = os.getenv('DEBUG')



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
class TooManyTriesException(Exception): pass

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
        
    raise TooManyTriesException('Could not get a response from the LLM after {max_retries} tries.')



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
        invokation = f'safe_invoke({node["name"]}_llm, [SystemMessage(content= prompt)<, anything else>])' if not node['subgraph_id'] else f'{node["subgraph_id"]}_app.invoke([SystemMessage(content= prompt)<, anything else>])',
        node_function = '\n'.join([
            f'# <comment>',
            f'def {node["name"]}(state):',
            f'    """ {node["description"]} """',
             '    print_function_name()',
             '    try:',
             '        <preprocess>',
            f'        prompt = prompts.{node["name"]}_PROMPT.format(<formatting>)',
            f'        {invokation[0]}',
             '        <postprocess>',
             '        return <return>',
             'except Exception as e:',
             '        print(f\'{RED}[NODE] [ERR]{RESET}\', e) if DEBUG else None',
             '        traceback.print_exc() if DEBUG else None',
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
        if 'start' in from_node.lower():
            from_node = '__start__' 
        if 'end' in to_node.lower():
            to_node = '__end__'
        edge_function = f'{graph_name}.add_edge("{from_node}", "{to_node}")\n'
        return edge_function
    
    def create_conditional_edge(from_node: str, to_nodes: list[str], graph_name: str) -> tuple[str, str]:
        '''
        `create_conditional_edge` is a function that creates a conditional edge

        `Args:`
            edge (WorkflowEdge): The edge to create
        
        `Returns:`
            (str) The edge
        '''
        # Conditional function
        literals = ', '.join([f'"{to_node}"' for to_node in to_nodes])
        conditional_function = '\n'.join([
            f'# <comment>',
            f'def from_{from_node}_to(state) -> Literal[{literals}]:',
            f'    ...',
             ''
        ])
        # Edge function
        edge_map = '\n'.join([f'        "{to_node}": {to_node},' for to_node in to_nodes])
        edge_function = '\n'.join([
            f'{graph_name}.add_conditional_edge(',
            f'    from_{from_node}_to,',
            f'    "{from_node}",' ,
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
            'nodes': nodes,
            'conditional_edge_functions': '\n'.join([conditional_edge[0] for conditional_edge in conditional_edges.values()]),
            'edges': ''.join(all_edges)
        }

    # Build for all graphs
    graphs = [bundle['root']] + list(bundle['subgraphs'].values())
    for graph in graphs:
        graph['name'] = graph['name'].replace(' ', '_').lower()
    
    to_return = {}
    for graph in graphs:
        built = build_graph(graph)
        to_return[graph['name']] = {
            'nodes': built['nodes'],
            'conditional_edge_functions': built['conditional_edge_functions'],
            'add_nodes': '\n'.join([
                f'{graph["name"]}_graph.add_node("{node["name"]}", {node["name"]})' 
                for node in graph['nodes'] if node['name'].lower() not in ('start', 'end')
            ]),
            'edges': built['edges']
        }

    return to_return



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
    for name, subgraph in parsed.items():
        print(f"Subgraph: {name}" if name != dict_to_test['root']['name'] else f"Root: {name}")
        print(subgraph['nodes'])
        print(subgraph['conditional_edge_functions'])
        print(subgraph['add_nodes'])
        print(subgraph['edges'])
        print()
