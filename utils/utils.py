
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
    qualified_name = f"{filename}/{func_name}"
    print(f'\n\033[93m[NODE]\033[0m {qualified_name}')



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
        invokation = f'        safe_invoke({node["name"]}_llm, [SystemMessage(content= prompt)<, anything else>])' if not node['subgraph_id'] else f'        {node["subgraph_id"]}_app.invoke([SystemMessage(content= prompt)<, anything else>])',
        node_function = '\n'.join([
            f'# <comment>',
            f'def {node["name"]}(state):',
            f'    """ {node["description"]} """',
             '    print_function_name()',
             '    try:',
             '        <preprocess>',
            f'        prompt = prompts.{node["name"]}_PROMPT.format(<formatting>)',
            f'        {invokation}',
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
        conditional_function = '\n'.join([
            f'# <comment>'
            f'def from_{from_node}_to(state) -> Literal[{", ".join(to_nodes)}]:'
            f'    ...'
             ''
        ])
        # Edge function
        edge_map = '\n'.join([f'        "{to_node}": "{to_node}",' for to_node in to_nodes])
        edge_function = '\n'.join([
            f'{graph_name}.add_conditional_edge('
            f'    "{from_node}",' 
            f'    "from_{from_node}_to",'
             '    { # Not needed just for clarity'
            f'{edge_map}'
             '    }'
            f')'
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
        nodes = ''.join([create_node(node) for node in graph['nodes'] if node['name'].lower() not in ('start', 'end')])

        # Edges
        edge_graph = {} # {source: [targets]}
        for edge in graph['edges']:
            if edge['source_name'] not in edge_graph:
                edge_graph[edge['source_name']] = []
            edge_graph[edge['source_name']].append(edge['target_name'])
            
        # Non Conditional Edges
        edges = {
            from_node: create_edge(from_node, to_nodes[0], graph['name']) 
            for from_node, to_nodes in edge_graph if len(to_nodes) == 1
        }

        # Conditional edges
        conditional_edges = {
            from_node: create_conditional_edge(from_node, to_nodes, graph['name']) 
            for from_node, to_nodes in edge_graph if len(to_nodes) > 1
        }

        # In order to keep the order
        all_edges = []
        for edge in edge_graph.keys():
            if edge in edges:
                all_edges.append(edges[edge])
            else:
                all_edges.append(conditional_edges[edge])

        # Return
        return {
            'nodes': nodes,
            'edges': ''.join(all_edges)
        }

    # Build for all graphs
    graphs = [bundle['root']] + list(bundle['subgraphs'].values())
    
    to_return = {}
    for graph in graphs:
        built = build_graph(graph)
        to_return[graph['name']] = {
            'nodes': built['nodes'],
            'edges': '\n'.join([
                f'{graph["name"]}_graph = StateGraph(<StateSchema>)',
                f'{built["edges"]}',
                f'{graph["name"]}_app = {graph["name"]}_graph.compile()'
            ])
        }

    return to_return



