"""
- `author:` Stefanos Panteli
- `date:` 2025-09-17
- `description:` Accepts a user input along with its refined version. After asking a series of clarifying questions about the technical workflow of the agent, it provides a graph version of the workflow.

## How to use
1. Import the app. (`from agents.workflowRefiner.workflowRefiner import workflow_refiner_app`)
2. Input a dict with the following keys:
    - `orchestrator: bool`: If it should call the orchestrator to get the inputs.
    - `user_input: str`: The user input as is.
    - `clarified_user_input: str`: The user input refined to be studied and made into a workflow.
3. Invoke the app.
4. Get the output dict with the following keys:
    - `workflow: WorkflowBundle`: A graph version of the workflow.
        - `comments: str`: Use this field to add any comments regarding to the users request.
        - `root: WorkflowGraph`: The root graph of the workflow.
        - `subgraphs: Dict[str, WorkflowGraph]`: The subgraphs of the workflow.
            - `sub_id: WorkflowGraph`: A subgraph of the workflow with id `sub_id`.
                - `type: Literal['reactive_conversational', 'linear_pipeline', 'planner_executor', 'hybrid']`: The type of the subgraph.
                - `name: str`: The name of the subgraph.
                - `description: str`: The description of the subgraph; and the why.
                - `nodes: List[WorkflowNode]`: The nodes of the subgraph.
                    - `name: str`: The name of the node in snake_case.
                    - `description: str`: The description of the node.
                    - `subgraph_id: Optional[str]`: If set, this node references a subgraph by its ID in WorkflowBundle.subgraphs.
                - `edges: List[WorkflowEdge]`: The edges of the subgraph.
                    - `source_name: str`: The name of the source node in snake_case.
                    - `target_name: str`: The name of the target node in snake_case.
                    - `description: str`: The description of the edge; and the why.
    
## Usage
```python
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
graph_input = {'user_input': 'I want a personall fitness coach.', 'clarified_user_input': <Output of the input refiner>}

response = workflow_refiner_app.invoke(graph_input)

# Only the str() representation of the workflow will be shown
# response = 📝 COMMENTS: Integrated iterative refinement into a planner_executor workflow to allow re-planning if the user is dissatisfied. The plan and execute nodes are designed to handle both workout and dietary plans together, and the reflect node decides whether to re-plan or proceed to output.
#            🌐 ROOT WORKFLOW
#            ╭─ Iterative Fitness and Nutrition Plan Generator [planner_executor] ─────
#            │ Iterative planner-executor workflow that collects user data, generates workout and dietary plans, then allows the user to provide feedback and request changes. The cycle of planning and execution repeats until the user is satisfied, then delivers the final output.
#            │
#            │ Nodes:
#            │   • start
#            │   ⤷ Execution: TRIGGER. Triggered by user providing input data.
#            │   • input_collection
#            │   ⤷ Execution: TOOLS. Collect required user input data: health status, equipment, session requirements, time flexibility, dietary preferences, age, weight, height, gender, activity level.
#            │   • plan
#            │   ⤷ Execution: LLM. Generate workout and dietary plans based on user input and feedback (if any).
#            │   • execute
#            │   ⤷ Execution: LLM. Present the generated workout and dietary plans to the user and handle feedback (e.g., adjustments like replacing eggs with broccoli).
#            │   • reflect
#            │   ⤷ Execution: CODE. Analyze user feedback and decide whether to re-plan or proceed to output.
#            │   • output_generation
#            │   ⤷ Execution: CODE. Format and deliver the final output including workout schedules, exercise instructions, meal plans, progress metrics, and safety tips.
#            │   • end
#            │   ⤷
#            │
#            │ Edges:
#            │   • start ➜  input_collection
#            │   ⤷ When user provides input data.
#            │   • input_collection ➜  plan
#            │   ⤷ After user input is collected.
#            │   • plan ➜  execute
#            │   ⤷ After plans are generated.
#            │   • execute ➜  reflect
#            │   ⤷ After user feedback is collected.
#            │   • reflect ➜  plan
#            │   ⤷ If feedback indicates changes are needed (user is dissatisfied).
#            │   • reflect ➜  output_generation
#            │   ⤷ If user is satisfied.
#            │   • output_generation ➜  end
#            │   ⤷ When output is delivered.
#            ╰──────────────────────────────────────────────────────────────────────────
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_tavily import TavilySearch

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode

# Schema imports
from typing import Literal, List, Optional, Dict
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

# My imports
from agents.clarificationOrchestrator.clarification_orchestrator import clarification_orchestrator_app
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, USER_APPROVALS
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
    # If it should call the orchestrator to get the inputs
    orchestrator: bool = Field(
        description= 'If it should call the orchestrator to get the inputs.',
        default= False
    )
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



''' Nodes '''
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
            # user_input= state['user_input'], # Not used
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
        if will_tool_call(state['messages'] + [clarification], instruction_texts= ['will use tavily_search to gather context']):
            print(f'{BLUE}[NODE] [INFO]{RESET} Will tool call') if DEBUG else None
            return {'messages': [clarification]}

        # If the orchestrator flag is up, call it
        if state['orchestrator']:
            orch_config = {
                'configurable': {
                    'user_id': 'inputRefiner',
                    'run_name': 'inputRefiner',
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

        return {'messages':new_messages}

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
                continue

            # Ask the user if the refined version of the user input is okay
            print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {workflow}')
            answer = input(f'{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ')
            if answer.lower() in USER_APPROVALS:
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
    if isinstance(state['messages'][-1], AIMessage) and will_tool_call(state['messages'], instruction_texts= ['will use tavily_search to gather context']):
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
        user_input= 'I want a personal fitness coach.', 
        clarified_user_input= '''<refined paragraph>
Design a comprehensive virtual AI-powered fitness and nutrition coaching agent that creates personalized, 
structured programs targeting simultaneous weight loss and muscle gain. The agent must utilize only bodyweight 
exercises and running as available equipment, accommodating 5 weekly sessions of 90 minutes each. 
Programs should be adaptable to either morning (dawn) or afternoon workout time slots within a free or minimal-cost model, 
delivered entirely in English. The solution requires no human interaction or location dependency, with integrated dietary planning and nutritional guidance.
- `role`: Virtual AI fitness coach and dietary assistant
- `scope/boundaries`:
   - Designs structured workout plans using bodyweight exercises and running
   - Creates integrated dietary plans for weight loss and muscle gain
   - Provides program guidance only (no execution or equipment provision)
   - Operates within free/minimal-cost constraints
   - Delivers content solely in English
- `inputs/data sources`:
   - User's health status (no conditions/injuries)
   - Available equipment: bodyweight exercises, running
   - Session requirements: 5 days/week, 90 minutes/session
   - Time flexibility: morning (dawn) or afternoon
   - Dietary preferences/allergies (if any)
- `outputs/format`:
   - Weekly workout plans (structured schedules)
   - Exercise instructions with form guidance
   - Integrated weekly meal plans with portion guidance
   - Macronutrient targets aligned with dual goals
   - Progress tracking metrics for both fitness and nutrition
- `constraints`:
   - **Cost**: Free or minimal-cost (freemium model)
   - **Equipment**: Bodyweight and running only
   - **Time**: Programs adaptable to dawn or afternoon slots
   - **Safety**: Safe for general healthy individuals
   - **Scope**: No medical diagnosis or prescription capabilities
   - **Language**: English-only delivery
   - **Dietary Limits**: Should avoid complex medical nutrition therapy
- `key preferences`:
   - Simultaneous weight loss and muscle gain focus
   - Integrated exercise and nutrition approach
   - Strict adherence to 5x90 minute weekly structure
- `additional requirements`:
   - Progression/scaling mechanisms for workouts and nutrition
   - Recovery and rest day guidelines
   - Form correction tips for injury prevention
   - Basic nutritional guidance with calorie/macro calculations
   - Hydration advice and food logging suggestions
   - Dietary flexibility options for preferences/restrictions'''
    )
    response: OutputSchema = workflow_refiner_app.invoke(user, config= config)
    
    print(f'{BLUE}[MAIN] [INFO]{RESET} Response', response) if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')

        print(response['workflow'].model_dump_json(indent= 4))
