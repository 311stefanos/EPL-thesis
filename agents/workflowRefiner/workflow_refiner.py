"""
- `author:` Stefanos Panteli
- `date:` 2025-09-17
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.workflowRefiner.workflowRefiner import workflow_refiner_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
graph_input = { # TODO: add }

response = workflow_refiner_app.invoke(graph_input)

# response = { # TODO: add }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

# Schema imports
from typing import Literal, List, Optional, Dict
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, print_function_name
from agents.workflowRefiner import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Workflow Refiner') if DEBUG else None



""" Schemas """
''' General Schemas '''
class WorkflowNode(BaseModel):
    name: str = Field(description= 'The name of the node in snake_case.')
    description: str = Field(description= 'The description of the node.')
    subgraph_id: Optional[str] = Field(
        description= 'If set, this node references a subgraph by its ID in WorkflowBundle.subgraphs.',
        default= None
    )

    def __str__(self) -> str:
        subgraph = f'(subgraph: {self.subgraph_id})' if self.subgraph_id else ''
        return f'• {self.name} {subgraph}\n│   ⤷ {self.description}'

class WorkflowEdge(BaseModel):
    source_name: str = Field(description= 'The name of the source node.')
    target_name: str = Field(description= 'The name of the target node.')
    description: str = Field(description= 'The description of the edge, and the why.')

    def __str__(self) -> str:
        # return f'{self.source_name} -> {self.target_name}: {self.description}'
        return f'• {self.source_name} ➜  {self.target_name}\n│   ⤷ {self.description}'

class WorkflowGraph(BaseModel):
    type: Literal['reactive_conversational', 'linear_pipeline', 'planner_executor', 'hybrid'] = Field(
        description= 'The type of the workflow.'
    )
    name: str = Field(description= 'The name of the workflow.')
    nodes: List[WorkflowNode] = Field(description= 'The nodes of the workflow.')
    edges: List[WorkflowEdge] = Field(description= 'The edges of the workflow.')
    description: str = Field(description= 'The description of the workflow; and the why.')

    def __str__(self) -> str:
        title_bar = f'╭─ {self.name} [{self.type}] ─────'
        desc_block = f'│ {self.description}'
        nodes_block = '│\n│ Nodes:\n' + '\n'.join(f'│   {str(node)}' for node in self.nodes) if self.nodes else '│ Nodes: (none)'
        edges_block = '│\n│ Edges:\n' + '\n'.join(f'│   {str(edge)}' for edge in self.edges) if self.edges else '│ Edges: (none)'
        bottom_bar = '╰' + '─' * (len(title_bar))
        return f'{title_bar}\n{desc_block}\n{nodes_block}\n{edges_block}\n{bottom_bar}'
    
class WorkflowBundle(BaseModel):
    comments: str = Field(
        description= 'Use this field to add any comments regarding to the users request.'
    )
    root: WorkflowGraph
    subgraphs: Dict[str, WorkflowGraph] = Field(default_factory= dict)

    def __str__(self) -> str:
        # Comments first
        bundle_output = [f'\n📝 COMMENTS: {self.comments}'] if self.comments else []
        # Root right after
        bundle_output.append(f'\n🌐 ROOT WORKFLOW\n{str(self.root)}')
        # Append subgraphs (sorted for deterministic order)
        for node in self.root.nodes:
            if node.subgraph_id and node.subgraph_id in self.subgraphs:
                sub_id = node.subgraph_id
                subgraph = self.subgraphs[sub_id]
                bundle_output.append(f'\n🧩 SUBGRAPH: {sub_id}\n{str(subgraph)}')
        return '\n'.join(bundle_output)

''' Input Schema '''
class InputSchema(MessagesState):
    user_input: str = Field(
        description= 'The user input to be studied and made into a workflow.'
    )
    clarified_user_input: Optional[str] = Field(
        description= 'The user input refined to be studied and made into a workflow.'
    )

''' Output Schema '''
class OutputSchema(BaseModel):
    workflow: WorkflowBundle = Field(
        description= 'The workflow created from the user input.'
    )



''' Tools '''
# Tavily, to search and gather information from the web
tavily_search = TavilySearch(
    tavily_api_key= TAVILY_API_KEY,
    search_depth= "advanced",
    max_results= 5,
    include_answer= True
).as_tool()

# TODO: can add a rag tool with tutorials and descriptions on each workflow type



''' LLM '''
clarifier = myChatOpenAI(
    temperature= 0.7
).bind_tools([tavily_search])

workflow_engineer = myChatOpenAI(
    temperature= 0.7
).with_structured_output(WorkflowBundle)



''' Helpful Functions '''
# Check if the last message will or should call a tool
def _will_tool_call(messages: list[BaseMessage], actually_called: bool= False) -> bool:
    '''
    Check if the last message will call a tool.

    ### Args:
    - `messages`: the list of messages up to now
        - **note:** remember to add the last message if the state is not updated yet
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
    return (
        # If actually_called is set, we should check wheather the last message is a tool call, not the content
        'will use tavily_search to gather context' in last_message.content.lower() and not actually_called or 
        hasattr(last_message, 'tool_calls') and last_message.tool_calls or
        hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls', False) or
        tools_condition({'messages': messages}) == 'tools'
    )



''' Nodes'''
# This node accepts a user input and asks clarifying questions or assumptions in a conversation in order to get the desired workflow.
def clarify(state: InputSchema) -> InputSchema:
    '''
    This node accepts a  user input and provides clarifying context
    '''
    print_function_name() if DEBUG else None
    
    if DEBUG and state['messages'] and isinstance(state['messages'][-1], ToolMessage): # The ToolNode added a message.
        print(f'{GREEN}[NODE] [TOOL RESULT]{RESET} {state["messages"][-1].content}')
    try:
        # prompt
        prompt = prompts.CLARIFICATION_PROMPT.format(
            user_input= state['user_input'],
            clarified_user_input= state.get('clarified_user_input') or '',
            clarifications= '\n\n---\n'.join([mess.content for mess in state['messages']])
        )
        
        # call the LLM
        clarification = safe_invoke(clarifier, [SystemMessage(content= prompt)])
        print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {clarification}') if DEBUG else None

        # If no further clarifications are needed, wrap it in an AIMessage
        if 'no clarification needed' in clarification.content.lower():
            print(f'{BLUE}[NODE] [INFO]{RESET} No further clarifications needed') if DEBUG else None
            return {'messages': [AIMessage(content = clarification.content)]}
        
        # If a tool call is needed/will be used, **do not** wrap it in an AIMessage, as it has to keep the context
        if _will_tool_call(state['messages'] + [clarification]):
            print(f'{BLUE}[NODE] [INFO]{RESET} Will tool call') if DEBUG else None
            return {'messages': [clarification]}

        # Otherwise (just a clarification question), wrap it in an AIMessage, and ask the user for input
        print(f'{GREEN}[NODE] [CLARIFICATION/ASSUMPTION QUESTION]{RESET} {clarification.content}')
        user_input = input(f'\n{GREEN}[NODE] [INPUT] >{RESET} ')

        return {'messages': [AIMessage(content = clarification.content), HumanMessage(content= user_input)]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state



# This node accepts a corrected version of a user input and a conversation history, and provides a refined version of it
def create_workflow(state: InputSchema) -> OutputSchema:
    '''
    This node accepts a the conversation history, and provides a refined version of it.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # prompt
        history: list[str] = []
        for mess in state['messages']:
            # Append all messages from the conversation.
            history.append(mess.pretty_repr() if isinstance(mess, BaseMessage) else str(mess))

        # call the LLM to refine
        workflow_tries_user_requests: list[str] = []
        should_continue = True
        while should_continue:
            # parse the prompt with the user requests
            prompt = prompts.CREATE_WORKFLOW_PROMPT.format(
                history= '\n---\n\n'.join(history),
                workflow_tries_user_requests= '\n---\n\n'.join(workflow_tries_user_requests)
            )
            
            try:
                workflow = safe_invoke(workflow_engineer, [SystemMessage(content= prompt)])
            except ValueError as e: 
                print(e)
                input('Press enter to continue')
                continue

            # print(f'{BLUE}[NODE] [INFO]{RESET} suggested workflow: {workflow}') if DEBUG else None

            # Ask the user if the refined version of the user input is okay
            print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {workflow}')
            answer = input(f'{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ')
            if answer in ['y', 'ye', 'yea', 'yes', 'ok', 'okay', 'k', '']:
                # Redundant because of the break, for clarity
                should_continue = False
                break

            else:
                messages = AIMessage(content= str(workflow)).pretty_repr() + HumanMessage(content= answer).pretty_repr()
                workflow_tries_user_requests.append(messages)

        print(f'{BLUE}[NODE] [INFO]{RESET} Final workflow: {workflow}') if DEBUG else None
        return OutputSchema(workflow= workflow)

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        
        # Return the original
        return state
    


''' Conditional Functions '''
# This conditional logic is used to determine what to do after clarifying: keep clarifying, use tools, or create_workflow
def keep_clarifying(state: InputSchema) -> Literal['clarify', 'tools', 'create_workflow']:
    '''
    This functions provides the next node to go after clarifying.
    - If no further clarifications are needed, go to refine
    - If a tool call is needed, go to tools
    - Otherwise, go to clarify

    Returns:
        Literal['clarify', 'tools', 'refine']
    '''
    print_function_name() if DEBUG else None

    # If no further clarifications are needed
    if 'no clarification needed' in state['messages'][-1].content.lower():
        print(f'{BLUE}[NODE] [INFO]{RESET} No further clarifications needed') if DEBUG else None
        return 'create_workflow'
    
    # If a tool call is needed
    if isinstance(state['messages'][-1], AIMessage) and _will_tool_call(state['messages']):
        print(f'{BLUE}[NODE] [INFO]{RESET} Will use a tool') if DEBUG else None
        return 'tools'

    # Otherwise, keep asking for clarifications
    print(f'{BLUE}[NODE] [INFO]{RESET} Will ask for clarifications') if DEBUG else None
    return 'clarify'



''' Graph '''
workflow_refiner_graph = StateGraph(InputSchema, output_schema= OutputSchema)

workflow_refiner_graph.add_node('clarify', clarify)
workflow_refiner_graph.add_node('tools', ToolNode([tavily_search]))
workflow_refiner_graph.add_node('create_workflow', create_workflow)

workflow_refiner_graph.add_edge(START, 'clarify')
workflow_refiner_graph.add_conditional_edges(
    'clarify', 
    keep_clarifying,
    {   # Not needed, just for clarity
        'clarify': 'clarify',
        'tools': 'tools',
        'create_workflow': 'create_workflow'
    }
)
workflow_refiner_graph.add_edge('tools', 'clarify')
workflow_refiner_graph.add_edge('create_workflow', END)

workflow_refiner_app = workflow_refiner_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(workflow_refiner_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/workflow_refiner_app.png', 'wb') as f:
        f.write(workflow_refiner_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'workflowRefiner'
    os.environ['LANGSMITH_PROJECT'] = 'workflowRefiner'
    client = Client()

    config = {
        'configurable': {
            'user_id': 'workflowRefiner',
            'run_name': 'workflowRefiner',
            'thread_id': 'workflowRefiner', 
        }
    }

    user = InputSchema(
        # user_input= 'I want an agent to help me with designing a holiday.', 
        # clarified_user_input= 'Create an interactive, English-only vacation-planning chat agent that acts as a comprehensive travel-research assistant for worldwide destinations. The agent starts every interaction with a friendly, neutral-professional greeting and then conducts a natural dialogue to gather: (1) departure location—city or IATA airport code; (2) destination(s) or criteria for destination discovery; (3) exact or flexible travel dates (e.g., “two weeks in September”); (4) total trip budget expressed in EUR plus the user’s home currency (automatically converted for comparison); (5) group size and any special requirements; (6) accommodation style (backpacker to luxury) and activity mix preferences. It immediately searches major booking APIs—flights, lodging, activities—returning 3-5 fully-priced, bookable options in real time, ranked by price–convenience balance unless the user explicitly prioritises other factors. Each option lists specific flight times, named hotels with live availability and total all-inclusive cost (taxes & mandatory fees noted). It then builds a balanced itinerary of relaxation and adventure, appending visa requirements, vaccination advice, travel-insurance suggestions, weather forecasts, and direct booking links. Transportation costs that exceed ~70 % of the stated budget are automatically hidden unless < 5 viable options exist, in which case they are shown with a clear budget-warning label. The agent maintains an automatic, continuously updated user-profile system that remembers general preferences (but never assumes they are permanent) and always re-confirms at the start of each new trip request. It works for any budget, any travel period, any global destination, but focuses on mainstream tourist cities and common routes. Responses are unlimited in length; no bookings are processed—users complete reservations externally.\n\n- role: Interactive vacation-planning research assistant\n- scope/boundaries: Worldwide travel research (flights, hotels, activities) for mainstream destinations; no payment or booking processing; English interface only\n- inputs/data sources: Real-time flight, hotel, activity, weather, visa, and health data from major booking platforms and public APIs; user-provided budget, dates, preferences, group details\n- outputs/format: 3-5 destination options with specific, bookable items, ranked by price-convenience balance, plus full itineraries, warnings, and direct booking links; unlimited conversational detail\n- constraints (cost/latency/safety/style/language): English only; filter out transport-heavy options unless < 5 choices; friendly-neutral tone; no response length cap; no minimum budget\n- key preferences: Auto-profile with preference memory (re-confirmed each session); EUR + home-currency pricing; flexible-date cost optimisation; budget-warning labels; focus on common tourist destinations'
        user_input= 'i want to create a system that recognises when i make google maps reviews, then stores them in a DB. Then i want to be able to converse with the system asking questions and maybe reccomendations.',
        clarified_user_input= '''<refined paragraph>
Develop a Unix-based agent that periodically fetches your Google Maps reviews using third-party APIs (with your provided credentials and place IDs), stores review details (text content, rating 1-5, location metadata, and timestamp) in a structured SQLite database, and provides a WhatsApp-based conversational interface 
for natural language queries about your review history and personalized recommendations based on your stored data.

<bullet list of agent-creation essentials:
- `role`: Review collection and analysis agent
- `scope/boundaries`:
  - Collects Google Maps reviews via third-party API
  - Stores data in local SQLite database
  - Provides conversational interface via WhatsApp for queries/recommendations
  - Excludes direct Google API access or web scraping
- `inputs/data sources`:
  - Third-party API credentials (SerpApi/Outscraper)
  - List of place IDs for frequented locations
  - User conversational queries/questions (sent via WhatsApp)
- `outputs/format`:
  - Structured SQLite database (reviews table with text, rating, location, timestamp, place_id)
  - Natural language responses to WhatsApp queries
  - Recommendations based on stored review patterns
- `constraints`:
  - Cost (third-party API usage fees)
  - Rate limits (API provider constraints)
  - Privacy (secure handling of API credentials)
  - Legal compliance (third-party ToS adherence)
  - No direct Google Maps API access or web scraping
  - WhatsApp integration limitations (message length, availability)
- `key preferences/deadlines`:
  - Prefer third-party API approach over scraping/detection
  - SQLite database storage
  - WhatsApp-based conversational interface
  - Recommendations based solely on user's stored history
- `additional technical requirements`:
  - Scheduled execution (cron jobs for periodic checks)
  - Natural language processing capabilities
  - Database schema design
  - API integration layer
  - WhatsApp integration (using Twilio/WhatsApp Business API)
  - Conversational interface implementation
  - Message parsing for WhatsApp queries'''
    )
    response: OutputSchema = workflow_refiner_app.invoke(user, config= config)
    
    print(f'{BLUE}[MAIN] [INFO]{RESET} Response', response) if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')

        print(response['workflow'].model_dump_json(indent= 4))
