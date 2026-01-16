"""
- `author:` Stefanos Panteli
- `date:` 2025-08-06
- `description:` Accepts a user input and provides a corrected and refined version of it, after asking a series of clarifying questions.

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
graph_input = {'user_input': 'I want a personall fitness coach.', 'orchestrator': False}

response = input_refiner_app.invoke(graph_input)

# response = {
#     'corrected_original': 'I want a personal fitness coach.', 
#     'refined_text': '<refined paragraph>
#                     Design a comprehensive virtual AI-powered fitness and nutrition coaching agent that creates personalized, 
#                     structured programs targeting simultaneous weight loss and muscle gain. The agent must utilize only bodyweight 
#                     exercises and running as available equipment, accommodating 5 weekly sessions of 90 minutes each. 
#                     Programs should be adaptable to either morning (dawn) or afternoon workout time slots within a free or minimal-cost model, 
#                     delivered entirely in English. The solution requires no human interaction or location dependency, with integrated dietary planning and nutritional guidance.
#
#                     - `role`: Virtual AI fitness coach and dietary assistant
#                     - `scope/boundaries`:
#                        - Designs structured workout plans using bodyweight exercises and running
#                        - Creates integrated dietary plans for weight loss and muscle gain
#                        - Provides program guidance only (no execution or equipment provision)
#                        - Operates within free/minimal-cost constraints
#                        - Delivers content solely in English
#                     - `inputs/data sources`:
#                        - User's health status (no conditions/injuries)
#                        - Available equipment: bodyweight exercises, running
#                        - Session requirements: 5 days/week, 90 minutes/session
#                        - Time flexibility: morning (dawn) or afternoon
#                        - Dietary preferences/allergies (if any)
#                     - `outputs/format`:
#                        - Weekly workout plans (structured schedules)
#                        - Exercise instructions with form guidance
#                        - Integrated weekly meal plans with portion guidance
#                        - Macronutrient targets aligned with dual goals
#                        - Progress tracking metrics for both fitness and nutrition
#                     - `constraints`:
#                        - **Cost**: Free or minimal-cost (freemium model)
#                        - **Equipment**: Bodyweight and running only
#                        - **Time**: Programs adaptable to dawn or afternoon slots
#                        - **Safety**: Safe for general healthy individuals
#                        - **Scope**: No medical diagnosis or prescription capabilities
#                        - **Language**: English-only delivery
#                        - **Dietary Limits**: Should avoid complex medical nutrition therapy
#                     - `key preferences`:
#                        - Simultaneous weight loss and muscle gain focus
#                        - Integrated exercise and nutrition approach
#                        - Strict adherence to 5x90 minute weekly structure
#                     - `additional requirements`:
#                        - Progression/scaling mechanisms for workouts and nutrition
#                        - Recovery and rest day guidelines
#                        - Form correction tips for injury prevention
#                        - Basic nutritional guidance with calorie/macro calculations
#                        - Hydration advice and food logging suggestions
#                        - Dietary flexibility options for preferences/restrictions'
# }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_tavily import TavilySearch

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode

# Schema imports
from typing import TypedDict, Annotated, List, Literal, Tuple
from pydantic import BaseModel, Field

# General imports
# from pyaspeller import YandexSpeller
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import os

# My imports
from agents.clarificationOrchestrator.clarification_orchestrator import clarification_orchestrator_app
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, USER_APPROVALS
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
    # If it should call the orchestrator to get the inputs
    orchestrator: bool = Field(
        description= 'If it should call the orchestrator to get the inputs.',
        default= False
    )
    # The user's input as is
    user_input: str = Field(
        description= 'The user input to be clarified and refined.'
    )

''' Intermediate Schema '''
class IntermediateSchema(MessagesState): # A lit of the messages
    # If it should call the orchestrator to get the inputs
    orchestrator: bool = Field(
        description= 'If it should call the orchestrator to get the inputs.',
        default= False
    )
    qna: List[Tuple[str, str]] = Field(
        description= 'The user answers to the clarifying questions.'
    )
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
    temperature= 0,
    model= 'mistralai/devstral-2512:free'
)

# The LLM used to clarify the user's input with questions and assumptions
clarifier = myChatOpenAI(
    temperature= 0.8,
    model= 'mistralai/devstral-2512:free'
).bind_tools([tavily_search])

# The LLM used to refine the user's input
refiner = myChatOpenAI(
    temperature= 0.7,
    model= 'mistralai/devstral-2512:free'
)



''' Nodes '''
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

        print(f'{BLUE}[NODE] [INFO] [CORRECTION]{RESET} {corrected}') if DEBUG else None

        # Replace the message with the corrected one
        return {
            'orchestrator': state['orchestrator'],
            'messages': [HumanMessage(content= corrected)],
            'qna': [],
            'corrected_original': corrected,
            'refinements': [],
            'user_requests': []
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return {
            'orchestrator': state['orchestrator'],
            'messages': [HumanMessage(content= state['user_input'])],
            'qna': [],
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
        print(f'{BLUE}[NODE] [TAVILY RESULT]{RESET} {state["messages"][-1].content}')
        
    try:
        # prompt
        prompt = prompts.CLARIFICATION_PROMPT.format(
            user_input= state['corrected_original'],
            tool_calls= '\n---\n'.join([mess.content for mess in state['messages'][1:]]),
            clarifications= '\n\n'.join([f'Question: {qna[0]}\nAnswer: {qna[1]}' for qna in state['qna']])
        )
        # call the LLM
        clarification = safe_invoke(clarifier, [SystemMessage(content= prompt)])

        print(f'{BLUE}[NODE] [LLM RESPONSE]{RESET} {clarification}') if DEBUG else None

        # If no further clarifications are needed, wrap it in an AIMessage
        if 'no clarification needed' in clarification.content.lower():
            print(f'{BLUE}[NODE] [INFO]{RESET} No further clarifications needed') if DEBUG else None
            return {'messages': [AIMessage(content = clarification.content)]}
        
        # If a tool call is needed/will be used, **do not** wrap it in an AIMessage, as it has to keep the context
        if will_tool_call(state['messages'] + [clarification], instruction_texts= ['will use tavily_search to gather context']):
            print(f'{BLUE}[NODE] [INFO]{RESET} Will use tavily web search to gather context') if DEBUG else None
            return {'messages': [clarification]}

        # If the orchestrator flag is up, call it
        if state['orchestrator']:
            orch_config = {
                'configurable': {
                    'user_id': 'inputRefiner',
                    'run_name': 'inputRefiner',
                    # ThreadID for the memory
                    'thread_id': 'clarificationOrchestrator'
                }
            }
            user_input = clarification_orchestrator_app.invoke({'question': clarification.content}, config= orch_config)
            # Wrap it in an AIMessage, and the answer in a HumanMessage
            new_qna = (user_input['qna'].question, user_input['qna'].answer)

        else:
            # Otherwise (just a clarification question), wrap it in an AIMessage, and ask the user for input
            print(f'{GREEN}[NODE] [CLARIFICATION/ASSUMPTION QUESTION]{RESET} {clarification.content}')
            
            user_input = input(f'\n{GREEN}[NODE] [INPUT] >{RESET} ') 
            new_qna = (clarification.content, user_input)

        return {'qna': state['qna'] + [new_qna]}

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
    if isinstance(state['messages'][-1], AIMessage) and will_tool_call(state['messages'], instruction_texts= ['will use tavily_search to gather context']):
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
    if answer.lower() in USER_APPROVALS:
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

input_refiner_app = input_refiner_graph.compile()



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

    # user = {'user_input': 'i want a math helper', 'orchestrator': True}
    user = {
        'user_input': 'i want an agent that will store my preferences on food and drink, and then when i sent a photo/link/text of a menu, it can give me suggestions. whenever i want it to be conversational and interactive.', 
        'orchestrator': False
    }
    response = input_refiner_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')