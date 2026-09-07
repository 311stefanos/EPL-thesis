''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

# Langgraph imports
from langgraph.graph import StateGraph, add_messages
from langgraph.constants import END, START

# Schema imports
from typing import TypedDict, Annotated, List, Tuple
from pydantic import BaseModel, Field

# General imports
# from pyaspeller import YandexSpeller
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name



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
    orchestrator: bool
    # The user's input as is
    user_input: str

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

    # for traceability
    messages: Annotated[List[BaseMessage], add_messages] 
    qna: List[Tuple[str, str]]
    refinements: Annotated[List[AIMessage], add_messages]
    user_requests: Annotated[List[HumanMessage], add_messages]



''' LLM '''
model = myChatOpenAI(
    temperature= 0.7
)



''' Nodes '''
def refined_paragraph(state: InputSchema) -> OutputSchema:
    print_function_name() if DEBUG else None
    
    try:
        prompt = (
            "You are an Input Refiner for a custom AI agent builder.\n"
            "\n"
            "Transform the user's request into a clear specification for the custom AI agent\n"
            "that should be created.\n"
            "\n"
            "Clarify and organize the request while preserving the user\'s intent.\n"
            "Where relevant, make clear:\n"
            "- the agent's purpose and main tasks;\n"
            "- the information or inputs it receives;\n"
            "- the outputs or actions it should produce;\n"
            "- any tools, data sources, integrations, or memory it needs;\n"
            "- important behavioural requirements, constraints, or user preferences.\n"
            "\n"
            "Resolve minor ambiguities when the intended meaning is reasonably clear.\n"
            "Do not invent requirements that are not stated or reasonably implied.\n"
            "Leave non-essential missing details unspecified.\n"
            "Do not ask questions and do not perform the task requested by the user.\n"
            "\n"
            "Return only the refined custom-agent specification in clear natural language.\n"
            "\n"
            "User request:\n"
            f"{state['user_input']}"
        )

        refined = safe_invoke(model, messages= [SystemMessage(content= prompt)]).content

        print(f'{BLUE}[NODE] [INFO]{RESET} Refined: {refined}') if DEBUG else None
        return OutputSchema(
            corrected_original= state['user_input'],
            refined_text= refined,
            messages= [HumanMessage(content= state['user_input']), AIMessage(content= refined)],
            qna= [],
            refinements= [],
            user_requests= []
        )

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        
        # Return the original
        return state



''' Graph '''
input_refiner_graph = StateGraph(InputSchema, output_schema= OutputSchema)

input_refiner_graph.add_node('refined_paragraph', refined_paragraph)

input_refiner_graph.add_edge(START, 'refined_paragraph')
input_refiner_graph.add_edge('refined_paragraph', END)

input_refiner_app = input_refiner_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(input_refiner_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
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