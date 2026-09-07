''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
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
from utils.utils import myChatOpenAI, safe_invoke, print_function_name



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
        return f'• {self.name} {subgraph}\n│   ⤷ {self.description}\n│'

class WorkflowEdge(BaseModel):
    source_name: str = Field(description= 'The name of the source node.')
    target_name: str = Field(description= 'The name of the target node.')
    description: str = Field(description= 'The description of the edge, and the why.')

    def __str__(self) -> str:
        # return f'{self.source_name} -> {self.target_name}: {self.description}'
        return f'• {self.source_name} ➜  {self.target_name}\n│   ⤷ {self.description}\n│'

class WorkflowGraph(BaseModel):
    type: Literal['reactive_conversational', 'linear_pipeline', 'planner_executor', 'hybrid'] = Field(
        description= 'The type of the workflow.'
    )
    memory: bool = Field(description= 'Whether the workflow uses memory.')
    name: str = Field(description= 'The name of the workflow.')
    nodes: List[WorkflowNode] = Field(description= 'The nodes of the workflow.')
    edges: List[WorkflowEdge] = Field(description= 'The edges of the workflow.')
    description: str = Field(description= 'The description of the workflow; and the why.')

    def __str__(self) -> str:
        title_bar = f'╭─ {self.name} [{self.type}] ─────' + (' with memory ─────' if self.memory else '')
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
    orchestrator: bool # If it should call the orchestrator to get the inputs.
    clarified_user_input: Optional[str] # The user input refined to be studied and made into a workflow.

''' Output Schema '''
class OutputSchema(BaseModel):
    workflow: WorkflowBundle = Field(description= 'The workflow created from the user input.')

    messages: List[BaseMessage]



''' LLM '''
model = myChatOpenAI(
    temperature= 0.7
).with_structured_output(WorkflowBundle)#, method='function_calling')



''' Nodes '''
def create_workflow(state: InputSchema) -> OutputSchema:
    '''
    This node accepts a the conversation history, and provides a refined version of it.
    '''
    print_function_name() if DEBUG else None
    
    try:
        
        prompt = (
            "You are a Workflow Refiner for a custom AI agent builder.\n"
            "\n"
            "Create a clear and implementable workflow for the custom AI agent described below.\n"
            "\n"
            "Choose the workflow structure that best matches the agent's required behaviour.\n"
            "The workflow should describe how the agent processes an input from start to finish.\n"
            "\n"
            "When creating the workflow:\n"
            "- identify the main processing steps;\n"
            "- keep the workflow as simple as possible while satisfying the request;\n"
            "- use separate nodes for meaningful processing stages;\n"
            "- connect the nodes in the order they should execute;\n"
            "- include branches or loops only when required by the agent's behaviour;\n"
            "- indicate whether the workflow requires persistent memory;\n"
            "- identify whether each step is primarily deterministic code, LLM reasoning, "
            "LLM reasoning with tools, or a subgraph;\n"
            "- use subgraphs only when sufficiently complex;\n"
            "- preserve all workflow requirements contained in the user's specification.\n"
            "\n"
            "Available workflow types are:\n"
            "- reactive_conversational: reacts conversationally to each user request;\n"
            "- linear_pipeline: follows a fixed sequence of processing steps;\n"
            "- planner_executor: plans tasks and then executes them;\n"
            "- hybrid: combines fixed processing with flexible conversational or planning behaviour.\n"
            "\n"
            "Every node description must begin with exactly one of:\n"
            "- \"Execution: CODE.\"\n"
            "- \"Execution: LLM.\"\n"
            "- \"Execution: LLM+TOOLS.\"\n"
            "- \"Execution: SUBGRAPH.\"\n"
            "\n"
            "Always include start and end nodes.\n"
            "Use concise snake_case node names.\n"
            "Edges must reference existing nodes and briefly explain the transition.\n"
            "\n"
            "Do not ask the user questions.\n"
            "Do not add unnecessary functionality.\n"
            "If a minor workflow detail is unspecified, choose the simplest reasonable implementation.\n"
            "\n"
            "Custom agent specification:\n"
            f"{state.get('clarified_user_input')}"
        )
            
        workflow: WorkflowBundle = safe_invoke(model, messages= [SystemMessage(content= prompt)])

        print(f'{GREEN}[NODE] [LLM RESPONSE]{RESET} {workflow}')

        # Add the start and end nodes if ommitted
        graphs: list[WorkflowGraph] = [workflow.root] + (list(workflow.subgraphs.values()) if workflow.subgraphs else [])
        for graph in graphs:
            first_node = graph.nodes[0]
            last_node = graph.nodes[-1]
            if 'start' not in [node.name.lower() for node in graph.nodes]:
                graph.nodes.insert(0, WorkflowNode(name= 'start', description= 'Start node', subgraph_id= None))
                graph.edges.insert(0, WorkflowEdge(source_name= 'start', target_name= first_node.name, description= 'Start node'))
            if 'end' not in [node.name.lower() for node in graph.nodes]:
                graph.nodes.append(WorkflowNode(name= 'end', description= 'End node', subgraph_id= None))
                graph.edges.append(WorkflowEdge(source_name= last_node.name, target_name= 'end', description= 'End node'))

        return OutputSchema(workflow= workflow, messages= [SystemMessage(content= prompt), AIMessage(content= str(workflow))])

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        
        # Return the original
        return state
    



''' Graph '''
workflow_refiner_graph = StateGraph(InputSchema, output_schema= OutputSchema)

workflow_refiner_graph.add_node('create_workflow', create_workflow)

workflow_refiner_graph.add_edge(START, 'create_workflow')
workflow_refiner_graph.add_edge('create_workflow', END)

workflow_refiner_app = workflow_refiner_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(workflow_refiner_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
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
        orchestrator= False,
        clarified_user_input= '''The agent will store and manage user-specific food and drink preferences, including dietary restrictions and allergies, in a structured JSON file. Upon receiving a menu (via photo, link, or text), it will parse the content in a single step and generate a ranked list of recommendations with clear explanations for each suggestion. The agent will support multiple user profiles, adapt over time based on feedback, and operate exclusively within the scope of menu items. It will engage users conversationally with a friendly tone in English, functioning as a WhatsApp chatbot without relying on external tools.

**Agent-Creation Essentials:**
- **Role:** Personalized menu recommendation assistant.
- **Scope/Boundaries:** Focused solely on food/drink menu items; no external tool usage.
- **Inputs/Data Sources:** Menu data (photo/link/text), user preferences (JSON file), user feedback.
- **Outputs/Format:** Ranked list of menu recommendations with explanations; conversational responses.
- **Constraints:**
  - Language: English only.
  - Style: Friendly, interactive tone.
  - Latency: Single-step menu parsing.
  - Safety: No external tools; data stored locally (JSON).
- **Key Preferences:**
  - Multi-user profile support.
  - Adaptive learning from feedback.
  - Deployment as a WhatsApp chatbot.
- **Deadlines:** None specified.'''
    )
    response: OutputSchema = workflow_refiner_app.invoke(user, config= config)
    
    print(f'{BLUE}[MAIN] [INFO]{RESET} Response', response) if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')

        print(response['workflow'].model_dump_json(indent= 4))
