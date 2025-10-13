"""
- `author:` Stefanos Panteli
- `date:` 2025-08-06
- `description:` Accepts a user input and provides a corrected and refined version of it.

## How to use
1. Import the app. (`from agents.inputRefiner.input_refiner import input_refiner_app`)
2. Input a dict with the following keys:
    - `user_input: str`: The user input to be refined.
3. Invoke the app.
4. Get the output dict with the following keys:
    - `corrected_original: str`: The original request with grammar and spelling fixed, vocabulary unchanged.
    - `refined_text: str`: A more precise, clear, and search-friendly version of the request.

## Usage
```python
from agents.userInputRefiner.input_refiner import input_refiner_app
graph_input = {'user_input': 'I want a personall fitness coach.'}

response = input_refiner_app.invoke(graph_input)

# response = {
#     'corrected_original': 'I want a personal fitness coach.', 
#     'refined_text': '# TODO: ADD'
# }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

# Schema imports
from typing import TypedDict, Annotated, List, Literal
from pydantic import BaseModel, Field

# General imports
from pyaspeller import YandexSpeller
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name
from agents.inputRefiner import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DEBUG = os.getenv('DEBUG')
MODEL_NAME = os.getenv('MODEL_NAME')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'

print(f'{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Input Refiner') if DEBUG else None



""" Schemas """
''' Input Schema '''
class InputSchema(TypedDict):
    # The user's input as is
    user_input: str = Field(
        description= 'The user input to be clarified and refined.'
    )

''' Intermediate Schema '''
class IntermediateSchema(MessagesState): # A lit of the messages
    # The user's input, grammatically corrected
    corrected_original: str = Field(
        description= 'The original request with grammar and spelling fixed, vocabulary unchanged.'
    )
    # The LLM refinement tries
    refinements: Annotated[List[AIMessage], add_messages] = Field(
        description= 'The LLM refinements, if any.'
    )
    # The user's requests to the LLM's refinements
    user_requests: Annotated[List[HumanMessage], add_messages] = Field(
        description= 'The user requests, if any.'
    )

''' Output Schema '''
class OutputSchema(BaseModel):
    # The user's input, grammatically corrected
    corrected_original: str = Field(
        description= 'The original request with grammar and spelling fixed, vocabulary unchanged.'
    )
    # The LLM refinement, as agreed by the user
    refined_text: str = Field(
        description= 'A more precise, clear, and search-friendly version of the request.'
    )



''' Tools '''
# Tavily, to search and gather information from the web
tavily_search = TavilySearch(
    tavily_api_key= TAVILY_API_KEY,
    search_depth= "advanced",
    max_results= 5,
    include_answer= True
).as_tool()



''' LLM '''
# The LLM used to correct the user's input
correcter =  myChatOpenAI(
    temperature= 0
)

# The LLM used to clarify the user's input with questions and assumptions
clarifier = myChatOpenAI(
    temperature= 0.8
).bind_tools([tavily_search])

# The LLM used to refine the user's input
refiner = myChatOpenAI(
    temperature= 0.7
)



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
    - 'will use tavily web search to gather context'
        - Skipped if actually_call is True
    - last_message.tool_calls exists and not empty
    - last_message.additional_kwargs.tool_calls exists and not empty
    - tools_condition(last_message) == tools
    '''
    last_message = messages[-1]
    return (
        # If actually_called is set, we should check wheather the last message is a tool call, not the content
        'will use tavily web search to gather context' in last_message.content.lower() and not actually_called or 
        hasattr(last_message, 'tool_calls') and last_message.tool_calls or
        hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls', False) or
        tools_condition({'messages': messages}) == 'tools'
    )



''' Nodes'''
# This node accepts a user input and provides a corrected version of it
def correct_user_input(state: InputSchema) -> IntermediateSchema:
    '''
    This node accepts a user input and provides a corrected version of it.
    '''

    print_function_name() if DEBUG else None
    
    try:
        # prompt
        user_input = state['user_input']
        prompt = prompts.CORRECTION_PROMPT.format(user_input= user_input)
        # call the LLM
        corrected = correcter.invoke(prompt).content

        # TODO: More tests. The LLM corrects even the grammar and capitalises.
        # Maybe its work is good enough to use rather that an LLM invokation
        # speller = YandexSpeller()
        # corrected = speller.spelled(state['user_input'])

        print(f'{BLUE}[NODE] [INFO] [CORRECTION]{RESET} {corrected}') if DEBUG else None

        # Replace the message with the corrected one
        return {
            'messages': [HumanMessage(content= corrected)],
            'corrected_original': corrected,
            'refinements': [],
            'user_requests': []
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return {
            'messages': [HumanMessage(content= state['user_input'])],
            'corrected_original': state['user_input'],
            'refinements': [],
            'user_requests': []
        }
    
# This node accepts a corrected version of a user input and asks clarifying questions or assumptions in a conversation.
def clarify(state: IntermediateSchema) -> IntermediateSchema:
    '''
    This node accepts a corrected version of a user input and provides clarifying context
    '''
    print_function_name() if DEBUG else None
    
    if DEBUG and isinstance(state['messages'][-1], ToolMessage): # The ToolNode added a message.
        print(f'{GREEN}[NODE] [TAVILY RESULT]{RESET} {state["messages"][-1].content}')
    try:
        # prompt
        prompt = prompts.CLARIFICATION_PROMPT.format(
            user_input= state['corrected_original'],
            clarifications= '\n---\n'.join([mess.content for mess in state['messages'][1:]])
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
            print(f'{BLUE}[NODE] [INFO]{RESET} Will use tavily web search to gather context') if DEBUG else None
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
def refine_user_input(state: IntermediateSchema) -> IntermediateSchema:
    '''
    This node accepts a corrected version of a user input and a conversation history, and provides a refined version of it.
    '''
    print_function_name() if DEBUG else None
    
    try:
        
        history: list[str] = []
        for mess in state['messages']:
            # Append all messages from the conversation.
            history.append(mess.pretty_repr() if isinstance(mess, BaseMessage) else str(mess))

        # If there are refinement tries or user requests, add them to the prompt.
        refinements_and_requests: list[str] = []
        refinements = state['refinements']
        requests = state['user_requests']
        for i in range(max(len(refinements), len(requests))):
            if i < len(refinements):
                refinements_and_requests.append(refinements[i].pretty_repr())
            if i < len(requests):
                refinements_and_requests.append(requests[i].pretty_repr())
            
        prompt = prompts.REFINE_INPUT_PROMPT.format(
            history= '\n---\n\n'.join(history),
            refinements_and_requests= '\n---\n\n'.join(refinements_and_requests)
        )

        # call the LLM to refine
        refined = safe_invoke(refiner, [SystemMessage(content= prompt)]).content

        print(f'{BLUE}[NODE] [INFO]{RESET} Refined: {refined}') if DEBUG else None

        # Ask the user if the refined version of the user input is okay
        print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {refined}')
        answer = input(f'{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ')

        return {'refinements': [AIMessage(content= refined)], 'user_requests': [HumanMessage(content= answer)]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        
        # Return the original
        return state

# Just parses the output so it can be returned
def parse_output(state: IntermediateSchema) -> OutputSchema:
    '''
    This node accepts a corrected version of a user input and provides a refined version of it.
    '''
    print_function_name() if DEBUG else None
    return OutputSchema(corrected_original= state['corrected_original'], refined_text= state['refinements'][-1].content)



''' Conditional Functions '''
# This conditional logic is used to determine what to do after clarifying: keep clarifying, use tools, or refine
def keep_clarifying(state: IntermediateSchema) -> Literal['clarify', 'tools', 'refine']:
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
        return 'refine'
    
    # If a tool call is needed
    if isinstance(state['messages'][-1], AIMessage) and _will_tool_call(state['messages']):
        print(f'{BLUE}[NODE] [INFO]{RESET} Will use tavily web search to gather context') if DEBUG else None
        return 'tools'

    # Otherwise, keep asking for clarifications
    print(f'{BLUE}[NODE] [INFO]{RESET} Will ask for clarifications') if DEBUG else None
    return 'clarify'

# This node asks the user if the refined version of the user input is okay
def refinement_okay(state: IntermediateSchema) -> Literal['parse_output', 'refine']:
    '''
    This node asks the user if the refined version of the user input is okay.
    '''
    print_function_name() if DEBUG else None
    
    answer = state['user_requests'][-1].content

    # If the answer is yes, parse the output and end
    if answer.lower() in ['y', 'ye', 'yea', 'yes', 'ok', 'okay', 'k', '']:
        return 'parse_output'

    # If the answer is no, keep refining
    else:
        return 'refine'



''' Graph '''
input_refiner_graph = StateGraph(IntermediateSchema, input_schema= InputSchema, output_schema= OutputSchema)

input_refiner_graph.add_node('correct', correct_user_input)
input_refiner_graph.add_node('clarify', clarify)
input_refiner_graph.add_node('tools', ToolNode([tavily_search]))
input_refiner_graph.add_node('refine', refine_user_input)
input_refiner_graph.add_node('parse_output', parse_output)

input_refiner_graph.add_edge(START, 'correct')
input_refiner_graph.add_edge('correct', 'clarify')
input_refiner_graph.add_conditional_edges(
    'clarify',
    keep_clarifying,
    {   # Not needed, but added for clarity
        'clarify': 'clarify',
        'tools': 'tools',
        'refine': 'refine'
    }
)
input_refiner_graph.add_edge('tools', 'clarify')
input_refiner_graph.add_conditional_edges(
    'refine',
    refinement_okay,
    {   # Not needed, but added for clarity
        'parse_output': 'parse_output',
        'refine': 'refine'
    }
)
input_refiner_graph.add_edge('parse_output', END)

input_refiner_app = input_refiner_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(input_refiner_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/input_refiner_app.png', 'wb') as f:
        f.write(input_refiner_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'inputRefiner'
    os.environ['LANGSMITH_PROJECT'] = 'inputRefiner'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'inputRefiner',
            'run_name': 'inputRefiner',
            'thread_id': 'inputRefiner'
        }
    }

    user = {'user_input': 'heljo, i want to go play baskeball wih me freinds in th e usa. congrajulations!'}
    response = input_refiner_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')