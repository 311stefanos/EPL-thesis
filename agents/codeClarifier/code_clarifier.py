"""
- `author:` Stefanos Panteli
- `date:` 2025-10-26
- `description:` # TODO: add

## How to use
1. Import the app. (`from agents.codeClarifier.codeClarifier import code_clarifier_app`)
2. Input a dict with the following keys:
    - # TODO: add
3. Invoke the app.
4. Get the output dict with the following keys:
    - # TODO: add

## Usage
```python
from agents.codeClarifier.code_clarifier import code_clarifier_app
graph_input = { # TODO: add }

response = code_clarifier_app.invoke(graph_input)

# response = { # TODO: add }
```
"""



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
from datetime import datetime
from pathlib import Path
import traceback
import json
import os
import re

# My imports
from agents.clarificationOrchestrator.clarification_orchestrator import clarification_orchestrator_app
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, USER_APPROVALS
from agents.codeClarifier import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Code Clarifier') if DEBUG else None



""" Schemas """
''' General Schemas '''
class ChangeComment(BaseModel):
    change: str = Field(description= 'The change as given from the agent.')
    comment: str = Field(description= 'The comment as given from the user.')

class UserFeedbackSchema(BaseModel):
    user_feedback: List[ChangeComment] = Field(description= 'The user feedback as given from the user.')

''' Input Schema '''
class InputSchema(MessagesState):
    orchestrator: bool = Field(description= 'If it should call the orchestrator to get the inputs.', default= False)
    current_file_name: str = Field(description= 'The current file name as given from the user.')
    clarified_user_input: str = Field(description= 'The clarified user input as given from the clarifier.')
    workflow: dict = Field(description= 'The proposed workflow as given from the workflow engineer.')
    code_structure: str = Field(description= 'The code structure as given from the software engineer.')



''' Tools '''
# Tool for the LLM to propose a new code structure
@tool
def propose_code_structure(file_name: str, proposed_code_structure: str, summary_of_changes: List[str]) -> str:
    '''
    `propose_code_structure`
        This function creates a new file for the proposed code structure.
        Only call this tool when you asked enough information from the user.
        Never call this tool before you asked enough information from the user.
        Never change the file_name or basic structure of the code.

    `Args:`
        file_name (str): The name of the file. Same as given.
        proposed_code_structure (str): The proposed code structure. Must be the same as the given structure with annotations.
        summary_of_changes (List[str]): The summary of changes. Each element of the list must a single change.

    `Returns:`
        (str) The file path
    '''
    print_function_name() if DEBUG else None

    base = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in file_name).strip("_") or "proposed_code"
    out_dir = Path("../../creations")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{base}_proposed_{ts}.py"

    with open(out_path, 'w', encoding= 'utf-8') as f:
        f.write(proposed_code_structure)

    print(f'\n{BLUE}[AGENT] [INFO] [PROPOSED CODE] [CREATED]{RESET} {out_path}') if DEBUG else None

    return str(out_path)

# Tool for the LLM to call when it completed the code annotations.
@tool
def complete(message: str):
    '''
    `complete`
        This function completes the code annotations.
        After the tool is called, the user is prompted to agree or not.

    `Args:`
        message (str): The message to complete.

    `Returns:`
        (str) The message
    '''
    print_function_name() if DEBUG else None

    return f'Message Recorded: {message}'

# List of tools
tools = [propose_code_structure, complete]
# Dictionary of tools: tool name -> tool
tools_by_name = {tool.name: tool for tool in tools}



''' LLM '''
annotator = myChatOpenAI(
    temperature= 0.7
).bind_tools(tools)

memory_updater = myChatOpenAI(
    temperature= 0
).with_structured_output(UserFeedbackSchema)



''' Helpful Functions '''
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
    


''' Nodes'''
# The clarification node that understands the code
def clarify(state: InputSchema) -> InputSchema:
    '''
    This node clarifies the code.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # prompt
        clarifications = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.ANNOTATE_CODE_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= state['code_structure'],
            clarifications= clarifications,
            filename= state['current_file_name']
        )

        # call the LLM
        clarification = safe_invoke(annotator, [SystemMessage(content= prompt)])

        print(f'{BLUE}[NODE] [INFO] [CLARIFICATION]{RESET} {clarification.content}') if DEBUG else None

        if will_tool_call(state['messages'] + [clarification]):
            print(f'{BLUE}[NODE] [INFO]{RESET} Tool Call') if DEBUG else None
            return {'messages': [clarification]}
        
        # If the orchestrator flag is up, call it
        if state['orchestrator']:
            orch_config = {
                'configurable': {
                    'user_id': 'codeClarifier',
                    'run_name': 'codeClarifier',
                    # ThreadID for the memory
                    'thread_id': 'clarificationOrchestrator'
                }
            }
            user_input = clarification_orchestrator_app.invoke({'question': clarification.content}, config= orch_config)
            # Wrap it in an AIMessage, and the answer in a HumanMessage
            new_messages = [AIMessage(content = user_input['qna'].question), HumanMessage(content= user_input['qna'].answer)]

        else:
            # Otherwise (just a clarification question), wrap it in an AIMessage, and ask the user for input
            print(f'{GREEN}[NODE] [CLARIFICATION/ASSUMPTION QUESTION]{RESET} {clarification.content}')
            
            user_input = input(f'\n{GREEN}[NODE] [INPUT] >{RESET} ') 
            new_messages = [AIMessage(content = clarification.content), HumanMessage(content= user_input)]

        return {'messages': new_messages}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

# The tool_node node, where the agent uses the tools
def tool_node(state: InputSchema) -> InputSchema:
    '''
    This node executes the tool.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # Get the last message, and extract the tool calls
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            from_kwargs = False
            tool_calls = last_message.tool_calls
        else:
            from_kwargs = True
            tool_calls = last_message.additional_kwargs.get('tool_calls', [])

        print(json.dumps(tool_calls, indent= 4)) if DEBUG else None

        # Execute all tool calls
        new_messages = []
        for tool_call in tool_calls:
            # Get the tool and arguments
            if from_kwargs:
                tool_call = tool_call['function']
            tool = tools_by_name[tool_call['name']]

            args = tool_call.get('args', {}) or tool_call.get('arguments', {})
            # Parse the tool arguments if needed.
            if isinstance(args, str):
                args = parse_tool_arguments(args)                

            print(f'{BLUE}[NODE] [INFO] [TOOL CALL]{RESET} {tool_call["name"]} with {args}') if DEBUG else None

            try:
                # Execute the tool
                observation = None
                observation = tool.invoke(args)

            except Exception as e:
                print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
                traceback.print_exc() if DEBUG else None
                # If the tool fails, skip it
                continue

            # If the tool called is the complete tool, ask the user if everything is okay.
            if tool in [complete] or tool_call['name'] == 'complete':
                user_approval = input(f'{GREEN}[NODE] [TOOL] [COMPLETE]{RESET} Do you approve? (y/request) > ')

                new_messages.extend([
                    ToolMessage(content= observation, name= tool_call['name'], tool_call_id= tool_call['id']),
                    HumanMessage(content= user_approval),
                ])


            # If the tool is a propose_code_structure, call an llm to update the memory
            elif tool in [propose_code_structure] or tool_call['name'] == 'propose_code_structure':
                # prompt
                summary_of_changes = args['summary_of_changes']
                proposed_code = args['proposed_code_structure']
                user_comments = input(f'{GREEN}[NODE] [TOOL] [PROPOSED CODE STRUCTURE]{RESET} Kindly check the proposed code structure in {observation}. Enter your comments:\n > ')

                prompt = prompts.MEMORY_UPDATE_PROMPT.format(
                    proposed_code_structure= proposed_code,
                    summary_of_changes= summary_of_changes,
                    user_comments= user_comments
                )
                result: UserFeedbackSchema = safe_invoke(memory_updater, [SystemMessage(content= prompt)])

                # Create a pair of AI and Human message
                all_changes = '\n'.join(f'{i}. {uf.change}' for i, uf in enumerate(result.user_feedback, start= 1))
                all_comments = '\n'.join(f'{i}. {uf.comment}' for i, uf in enumerate(result.user_feedback, start= 1))
                new_messages.extend([
                    AIMessage(content= f'Proposed changes:\n{all_changes}'),
                    HumanMessage(content= f'Comments by the user:\n{all_comments}')
                ])
        
        # Add them to the state
        return {'messages': new_messages}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state



''' Conditional Functions '''
# The next node to go after a clarification question
def after_clarify(state: InputSchema) -> Literal['clarify', 'tools']:
    '''
    This functions provides the next node to go after clarifying.
    - If a tool call is needed, go to tools
    - Otherwise, go back to clarify
    '''
    print_function_name() if DEBUG else None
    # If a tool call is needed
    if will_tool_call(state['messages']):
        return 'tools'
    
    return 'clarify'

# The next node to go after tools
def after_tools(state: InputSchema) -> Literal['clarify', '__end__']:
    '''
    This functions provides the next node to go after tools.
    - If the user approves, go to __end__
    - Otherwise, go back to clarify
    '''
    print_function_name() if DEBUG else None
    last_message = state['messages'][-1]
    second_to_last_message = state['messages'][-2]

    if (
        isinstance(second_to_last_message, ToolMessage) and # Complete tool
        isinstance(last_message, HumanMessage) and          # User approval or not
        last_message.content.lower() in USER_APPROVALS
    ):
        return '__end__'
    
    return 'clarify'


''' Graph '''
code_clarifier_graph = StateGraph(InputSchema) # TODO: change

code_clarifier_graph.add_node('clarify', clarify)
code_clarifier_graph.add_node('tools', tool_node)

code_clarifier_graph.add_edge(START, 'clarify')
code_clarifier_graph.add_conditional_edges(
    'clarify',
    after_clarify,
    {   # Not needed, for clarity
        'clarify': 'clarify',
        'tools': 'tools'
    }
)
code_clarifier_graph.add_conditional_edges(
    'tools',
    after_tools,
    {   # Not needed, for clarity
        'clarify': 'clarify',
        '__end__': END
    }
)

code_clarifier_app = code_clarifier_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(code_clarifier_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/code_clarifier_app.png', 'wb') as f:
        f.write(code_clarifier_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'codeClarifier'
    os.environ['LANGSMITH_PROJECT'] = 'codeClarifier'
    client = Client()

    config = {
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'codeClarifier',
            'run_name': 'codeClarifier',
            'thread_id': 'codeClarifier', 
        }
    }

    from test_inputs import filename, clarified_user_input, workflow, code_structure
    user = InputSchema(
        orchestrator= False,
        current_file_name= filename,
        clarified_user_input= clarified_user_input,
        workflow= workflow,
        code_structure= code_structure
    )
    response = code_clarifier_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
