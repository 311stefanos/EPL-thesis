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
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

# Schema imports
from typing import TypedDict, Literal, List, Optional, Annotated, Union
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import os

# My imports
from agents.workflowRefiner import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DEBUG = os.getenv('DEBUG')
MODEL_NAME = os.getenv('MODEL_NAME')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'


print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Workflow Refiner') if DEBUG else None



""" Schemas """
''' General Schemas '''
class WorkflowNode(BaseModel):
    name: str = Field(description= 'The name of the node.')
    description: str = Field(description= 'The description of the node.')

    def __str__(self) -> str:
        return f'{self.name}: {self.description}'

class WorkflowEdge(BaseModel):
    source_name: str = Field(description= 'The name of the source node.')
    target_name: str = Field(description= 'The name of the target node.')
    description: str = Field(description= 'The description of the edge, and the why.')

    def __str__(self) -> str:
        return f'{self.source_name} -> {self.target_name}: {self.description}'

class WorkflowGraph(BaseModel):
    type: Literal['reactive_conversational', 'linear_pipeline', 'planner_executor', 'hybrid'] = Field(
        description= 'The type of the workflow.'
    )
    Nodes: List[Union[WorkflowNode, 'WorkflowGraph']] = Field(description= 'The nodes of the workflow.')
    Edges: List[WorkflowEdge] = Field(description= 'The edges of the workflow.')
    description: str = Field(description= 'The description of the workflow; and the why.')

    def __str__(self) -> str:
        nodes = 'Nodes:\n' + '\n'.join([str(node) for node in self.Nodes])
        edges = 'Edges:\n' + '\n'.join([str(edge) for edge in self.Edges])
        return f'{self.type}: {self.description}\n\n---\n\n{nodes}\n\n{edges}\n'

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
    workflow: WorkflowGraph = Field(
        description= 'The workflow created from the user input.'
    )



''' Tools '''
# The think tool, is for strategic reflection of the agent
@tool(description= 'Strategic reflection tool for workflow planning')
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on workflow planning and decision-making.

    Use this tool after some messages of the conversation to analyze results and plan next steps systematically.
    This creates a deliberate pause in the workflow planning for quality decision-making.

    When to use:
    - After receiving vital information: What key information did I find?
    - Before deciding next steps: Do I have enough to answer confidently?
    - When assessing gaps: What specific information am I still missing?
    - Before concluding: Can I provide a complete and accurate answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence for a good answer?
    4. Strategic decision - Should I continue asking or provide my answer?

    Args:
        reflection: Your detailed reflection on progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f'Reflection recorded: {reflection}'

# Tavily, to search and gather information from the web
# tavily_search = TavilySearch(
#     tavily_api_key= TAVILY_API_KEY,
#     search_depth= "advanced",
#     max_results= 5,
#     include_answer= True
# ).as_tool() # TODO: add the tool

# TODO: can add a rag tool with tutorials and descriptions on each workflow type



''' LLM '''
clarifier = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= MODEL_NAME,
    temperature= 0.7
).bind_tools([think_tool]) # TODO: add the tavily tool and maybe remove the think tool

workflow_engineer = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= MODEL_NAME,
    temperature= 0.7
).with_structured_output(WorkflowGraph)



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
    - 'Will use think_tool to reflect on progress'
        - Skipped if actually_call is True
    - last_message.tool_calls exists and not empty
    - last_message.additional_kwargs.tool_calls exists and not empty
    - tools_condition(last_message) == tools
    '''
    last_message = messages[-1]
    return (
        # If actually_called is set, we should check wheather the last message is a tool call, not the content
        'Will use think_tool to reflect on progress' in last_message.content.lower() and not actually_called or 
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
    print(f'\n{BLUE}[NODE]{RESET} workflow_refiner/clarify') if DEBUG else None
    if DEBUG and state['messages'] and isinstance(state['messages'][-1], ToolMessage): # The ToolNode added a message.
        print(f'{GREEN}[NODE] [TOOL RESULT]{RESET} {state["messages"][-1].content}')
    try:
        # prompt
        prompt = prompts.CLARIFICATION_PROMPT.format(
            user_input= state['user_input'],
            clarified_user_input= state.get('clarified_user_input') or '',
            clarifications= '\n---\n'.join([mess.content for mess in state['messages']])
        )
        # call the LLM
        clarification = clarifier.invoke([SystemMessage(content= prompt)])
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

        # If an error code 429 is returned
        if 'Error code: 429' in str(e):
            print(f'{RED}[NODE] [ERR]{RESET} Too many requests. Waiting 5 seconds.') if DEBUG else None
            sleep(5)

        return state



# This node accepts a corrected version of a user input and a conversation history, and provides a refined version of it
def create_workflow(state: InputSchema) -> OutputSchema:
    '''
    This node accepts a the conversation history, and provides a refined version of it.
    '''
    print(f'\n{BLUE}[NODE]{RESET} input_refiner/refine_user_input') if DEBUG else None
    try:
        # prompt
        history: list[str] = []
        for mess in state['messages']:
            # Append all messages from the conversation.
            history.append(mess.pretty_repr() if isinstance(mess, BaseMessage) else str(mess))
            
        prompt_with_history = prompts.CREATE_WORKFLOW_PROMPT.format(
            history= '\n---\n\n'.join(history)
        )

        # call the LLM to refine
        workflow_tries_user_requests: list[str] = []
        should_continue = True
        while should_continue:
            # parse the prompt with the user requests
            prompt = prompt_with_history.format(
                workflow_tries_user_requests = '\n---\n\n'.join(workflow_tries_user_requests)
            )
            try:
                workflow = workflow_engineer.invoke([SystemMessage(content= prompt)])
            except Exception as e:
                if 'error' in e and '429' in str(e['error']['code']):
                    print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
                    print(f'{RED}[NODE] [ERR]{RESET} Too many requests. Waiting 5 seconds.') if DEBUG else None
                    sleep(5)
                    continue
                else:
                    raise e

            print(f'{BLUE}[NODE] [INFO]{RESET} suggested workflow: {workflow}') if DEBUG else None

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
                # print(f'{BLUE}[NODE] [INFO]{RESET} User request: {answer}') if DEBUG else None

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
    print(f'\n{BLUE}[NODE]{RESET} input_refiner/keep_clarifying') if DEBUG else None

    # If no further clarifications are needed
    if 'no clarification needed' in state['messages'][-1].content.lower():
        print(f'{BLUE}[NODE] [INFO]{RESET} No further clarifications needed') if DEBUG else None
        return 'create_workflow'
    
    # If a tool call is needed
    if isinstance(state['messages'][-1], AIMessage) and _will_tool_call(state['messages']):
        print(f'{BLUE}[NODE] [INFO]{RESET} Will use a tool') if DEBUG else None
        # # But no actually tool call happened
        # while not _will_tool_call(state['messages'], actually_called= True):
        #     sys_msg = prompts.FORCE_TOOL_CALL
        #     # Call the llm again to make it call the tool
        #     state['messages'] += [ # Append the LLM's response
        #         clarifier.invoke([state['messages'][-1], SystemMessage(content= sys_msg)])
        #     ]
        #     print(f'{BLUE}[NODE] [INFO]{RESET} Trying to call the tool.') if DEBUG else None
        #     input('\n> press to continue') if DEBUG else None

        return 'tools'

    # Otherwise, keep asking for clarifications
    print(f'{BLUE}[NODE] [INFO]{RESET} Will ask for clarifications') if DEBUG else None
    return 'clarify'



''' Graph '''
workflow_refiner_graph = StateGraph(InputSchema, output_schema= OutputSchema)

workflow_refiner_graph.add_node('clarify', clarify)
workflow_refiner_graph.add_node('tools', ToolNode([think_tool]))
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
    # Image(workflow_refiner_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    # parent_dir = Path(__file__).resolve().parent
    # if not os.path.exists(parent_dir / 'graphs'):
    #     os.makedirs(parent_dir / 'graphs')
    # with open(parent_dir / 'graphs/workflow_refiner_app.png', 'wb') as f:
    #     f.write(workflow_refiner_app.get_graph().draw_mermaid_png())

    
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
        user_input= 'I want an agent to help me with designing a holiday.', 
        clarified_user_input= 'Create an interactive, English-only vacation-planning chat agent that acts as a comprehensive travel-research assistant for worldwide destinations. The agent starts every interaction with a friendly, neutral-professional greeting and then conducts a natural dialogue to gather: (1) departure location—city or IATA airport code; (2) destination(s) or criteria for destination discovery; (3) exact or flexible travel dates (e.g., “two weeks in September”); (4) total trip budget expressed in EUR plus the user’s home currency (automatically converted for comparison); (5) group size and any special requirements; (6) accommodation style (backpacker to luxury) and activity mix preferences. It immediately searches major booking APIs—flights, lodging, activities—returning 3-5 fully-priced, bookable options in real time, ranked by price–convenience balance unless the user explicitly prioritises other factors. Each option lists specific flight times, named hotels with live availability and total all-inclusive cost (taxes & mandatory fees noted). It then builds a balanced itinerary of relaxation and adventure, appending visa requirements, vaccination advice, travel-insurance suggestions, weather forecasts, and direct booking links. Transportation costs that exceed ~70 % of the stated budget are automatically hidden unless < 5 viable options exist, in which case they are shown with a clear budget-warning label. The agent maintains an automatic, continuously updated user-profile system that remembers general preferences (but never assumes they are permanent) and always re-confirms at the start of each new trip request. It works for any budget, any travel period, any global destination, but focuses on mainstream tourist cities and common routes. Responses are unlimited in length; no bookings are processed—users complete reservations externally.\n\n- role: Interactive vacation-planning research assistant\n- scope/boundaries: Worldwide travel research (flights, hotels, activities) for mainstream destinations; no payment or booking processing; English interface only\n- inputs/data sources: Real-time flight, hotel, activity, weather, visa, and health data from major booking platforms and public APIs; user-provided budget, dates, preferences, group details\n- outputs/format: 3-5 destination options with specific, bookable items, ranked by price-convenience balance, plus full itineraries, warnings, and direct booking links; unlimited conversational detail\n- constraints (cost/latency/safety/style/language): English only; filter out transport-heavy options unless < 5 choices; friendly-neutral tone; no response length cap; no minimum budget\n- key preferences: Auto-profile with preference memory (re-confirmed each session); EUR + home-currency pricing; flexible-date cost optimisation; budget-warning labels; focus on common tourist destinations'
    )
    response = workflow_refiner_app.invoke(user, config= config)
    
    print(f'{BLUE}[MAIN] [INFO]{RESET} Response', response) if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')

'''
hybrid: Hybrid workflow for holiday planning. Triggered by user messages, with a conversational core (streaming I/O) for preference gathering and feedback, and linear pipeline steps (batch I/O) for research and generation. The workflow includes conditional branching and looping back to preference gathering for both research and feedback steps.

---

Nodes:
Start: Triggered when the user speaks to it (activated by user message). Begins the holiday design process.
GatherPreferences: The agent asks clarifying questions and collects user preferences, suggesting destination options if needed. (Tools: Ask user (LLM), I/O Mode: streaming). Guard: User provides preferences (budget, destination, dates, activities).
ResearchOptions: The agent uses external tools (Web Search for flights, hotels, attractions) and other agents to find relevant travel options for the chosen destination. (Tools: 
Web Search + other agents). Guard: Preferences are collected.
GenerateItinerary: The agent creates a sample holiday itinerary based on user preferences and research results. (Tools: LLM). Guard: Research options are retrieved OR if skipped 
from GatherPreferences.
PresentDesign: The agent presents holiday design with recommendations, links, and cost estimates. (Tools: LLM). Guard: Itinerary is generated.
CollectFeedback: The agent asks for feedback and changes on the design. (Tools: Ask user (LLM), I/O Mode: streaming). Guard: User reviews design. Also, the user can ask for drastic changes that lead back to preferences gathering.
FinalizeDesign: The agent finalizes the holiday design automatically. (Tools: LLM). Guard: Workflow completes (user approves or no changes needed, or user says 'done').
End: The workflow ends.

Edges:
Start -> GatherPreferences: Triggered by a user message to begin the workflow.
GatherPreferences -> ResearchOptions: Guard: if agent decides to research.
GatherPreferences -> GenerateItinerary: Guard: if agent decides to generate (either without research or after having done research and decided to generate).
ResearchOptions -> GatherPreferences: Guard: if more preferences are needed after research.
GenerateItinerary -> PresentDesign: Guard: itinerary is generated.
PresentDesign -> CollectFeedback: Guard: design is presented.
CollectFeedback -> GatherPreferences: Guard: if user indicates drastic changes that require more preferences.
CollectFeedback -> FinalizeDesign: Guard: if user approves or only minor changes.
FinalizeDesign -> End: End the workflow after finalizing the design.
'''