''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.constants import END, START

# Schema imports
from typing import Literal, List, Union, Optional
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, read_state_file
from agents.workflowRefiner.workflow_refiner import WorkflowBundle



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
# General
class Argument(BaseModel):
    name: str = Field(description= 'The name of the argument.')
    type: str = Field(description= 'The type of the argument.')

    def __str__(self):
        if self.name == 'self':
            return f'self'
        return f'{self.name}: {self.type}'
        
class Function(BaseModel):
    function_name: str = Field(description= 'The proposed helper function name.')
    arguments: List[Argument] = Field(description= 'The arguments of the helper function.')
    output: str = Field(description= 'The output of the helper function.')
    docstring: str = Field(description= 'The docstring of the helper function.')
    justification: str = Field(description= 'The justification of the helper function. Why is needed.')

    def __str__(self):
        docstring = self.docstring.replace('\n', '\n\t')
        arguments = ', '.join([str(arg) for arg in self.arguments])
        return f'def {self.function_name}({arguments}) -> {self.output}:\n\t"""\n\t{docstring}\n\t"""\n\t...'
    
    def to_method(self):
        docstring = self.docstring.replace('\n', '\n\t\t')
        # Add the self argument if not present
        args = list(self.arguments)
        if not any(arg.name == 'self' for arg in args):
            args.insert(0, Argument(name='self', type='...'))

        arguments = ', '.join(str(arg) for arg in args)
        return f'\tdef {self.function_name}({arguments}) -> {self.output}:\n\t\t"""\n\t{docstring}\n\t\t"""\n\t\t...'

# Docstring agent
class Docstring(BaseModel):
    function: str = Field(description= 'The function name as given.')
    docstring: str = Field(description= 'The docstring as given.')

    def __str__(self):
        return f'Function: {self.function}\nDocstring: {self.docstring}'

class Docstrings(BaseModel): # Used by the docstring_generator
    thinking_process: str = Field(description= 'The thinking process, TODO list, explanations, and more.')
    docstrings: List[Docstring] = Field(description= 'The docstrings as given from the user.')

    def __str__(self):
        return f'Thinking Process: {self.thinking_process}\n\n' + '\n'.join([f'\n{i}) {docstring}' for i, docstring in enumerate(self.docstrings, start= 1)])

# Schema agent
class SchemaArgument(Argument):
    comment: str = Field(description= 'The comment of the argument.')

    def __str__(self):
        return f'{super().__str__()} # {self.comment}'

class Schema(BaseModel):
    schema_name: str = Field(description= 'The name of the schema in PascalCase.')
    docstring: str = Field(description= 'The docstring of the schema. Be sure to comment on every argument\'s structure.')
    base_class: Literal['BaseModel', 'TypedDict', 'MessagesState'] = Field(description= 'The base class of the schema.')
    arguments: List[SchemaArgument] = Field(description= 'The arguments of the schema.')
    proposed_methods: List[Function] = Field(description= 'The proposed methods of the schema.')

    # If the chema is MessagesState, remove an argument that is named "messages"
    def __post_init__(self):
        if self.base_class == 'MessagesState':
            self.arguments = [arg for arg in self.arguments if arg.name != 'messages']

    def __str__(self):
        arguments = '\n\t'.join([str(arg) for arg in self.arguments])
        docstring = self.docstring.replace('\n', '\n\t')
        methods = '\n\n'.join([method.to_method() for method in self.proposed_methods])
        # If no arguments and no methods, just pass so it follows correct syntax
        if not arguments and not methods:
            return f'class {self.schema_name}({self.base_class}):\n\t"""\n\t{docstring}\n\t"""\n\tpass\n\n'
        
        return f'class {self.schema_name}({self.base_class}):\n\t"""\n\t{docstring}\n\t"""\n\t{arguments}\n\n{methods}\n\n'

class Schemas(BaseModel): # Used by the schema_generator
    thinking_process: str = Field(description= 'The thinking process, TODO list, explanations, and more.')
    schemas: List[Schema] = Field(description= 'The schemas.')

    def __str__(self):
        return f'Thinking Process: {self.thinking_process}\n\n' + '\n\n'.join([f'\n{i}) {schema}' for i, schema in enumerate(self.schemas, start= 1)])
    
# Helpful functions agent
class HelpfulFunctions(BaseModel): # Used by the helpful_function_generator
    thinking_process: str = Field(description= 'The thinking process, TODO list, explanations, and more.')
    helpful_functions: List[Function] = Field(description= 'The helpful functions.')

    def __str__(self):
        return f'Thinking Process: {self.thinking_process}\n\n' + '\n'.join([f'\n{i}) {function.justification}\n{function}' for i, function in enumerate(self.helpful_functions, start= 1)])
    
# Tool functions agent
class ToolFunctions(BaseModel): # Used by the tool_function_generator
    thinking_process: str = Field(description= 'The thinking process, TODO list, explanations, and more.')
    tool_functions: List[Function] = Field(description= 'The tool functions.')

    def __str__(self):
        return f'Thinking Process: {self.thinking_process}\n\n' + '\n'.join([f'\n{i}) {function.justification}\n@tool\n{function}' for i, function in enumerate(self.tool_functions, start= 1)])

# For the LLM Modifier Engineer
class LLMProposalsDict(BaseModel):
    llm_name: str = Field(description= 'The name of the LLM as is in the code.')
    with_structured_output: Optional[str] = Field(description= 'The strucutured output of the LLM. Must be a valid schema.', default= None)
    bind_tools: Optional[List[str]] = Field(description= 'The tools to bind to the LLM.', default= None)
    temp: float = Field(description= 'The temp of the LLM.', le= 1, ge= 0)

    def to_string(self):
        modifier = ''
        tools = output = ''
        if self.bind_tools: tools = ', '.join(self.bind_tools).replace('"', '').replace("'", '')
        if self.with_structured_output: output = self.with_structured_output.replace('"', '').replace("'", '')

        if tools and output:
            modifier = f'.bind_tools([{tools}, {output}])'
        elif tools:
            modifier = f'.bind_tools([{tools}])'
        elif output:
            modifier = f'.with_structured_output({output})'

        return f'{self.llm_name} = myChatOpenAI(\n\ttemperature= {self.temp}\n){modifier}\n'
    
class LLMProposalList(BaseModel): # Used by the tool_or_output_generator
    thinking_process: str = Field(description= 'The thinking process, TODO list, explanations, and more.')
    llm_proposals: List[LLMProposalsDict] = Field(description= 'The LLM proposals as given from the LLM.')

    def get_all_llm_names(self):
        if not self.llm_proposals or self.llm_proposals == []:
            return []
        return [llm_proposal.llm_name for llm_proposal in self.llm_proposals]
    
    def get_all_tool_names(self):
        if not self.llm_proposals or self.llm_proposals == []:
            return []
        tools = [llm_proposal.bind_tools for llm_proposal in self.llm_proposals if llm_proposal.bind_tools]
        return [tool for tool_list in tools for tool in tool_list if tool]
    
    def get_all_schema_names(self):
        if not self.llm_proposals or self.llm_proposals == []:
            return []
        return [llm_proposal.with_structured_output for llm_proposal in self.llm_proposals if llm_proposal.with_structured_output]

    def to_string(self):
        return f'Comments: {self.thinking_process}\n\n' + '\n'.join([f'{llm_proposal.to_string()}' for llm_proposal in self.llm_proposals])
    
    def to_code(self):
        return '\n'.join([f'{llm_proposal.to_string()}' for llm_proposal in self.llm_proposals]) + '\n\n\n\n'



class AblationCodeAnnotations(BaseModel):
    docstrings: List[Docstring] = Field(
        description='Docstrings for the existing workflow node functions.'
    )
    schemas: List[Schema] = Field(
        description='Schemas required by the generated agent, including AgentSchema.'
    )
    helpful_functions: List[Function] = Field(
        description='Necessary deterministic helper function definitions.'
    )
    tool_functions: List[Function] = Field(
        description='Necessary tool function definitions for LLM use.'
    )
    llm_proposals: List[LLMProposalsDict] = Field(
        description='Configuration for every LLM defined in the scaffold.'
    )



''' Input Schema '''
class InputSchema(MessagesState):
    file_path: str # The current file path as given from the user.
    clarified_user_input: str # The clarified user input as given from the clarifier.
    workflow: WorkflowBundle # The proposed workflow as given from the workflow engineer.

    # The step changes as given from the LLM
    step_changes: Union[    
        AblationCodeAnnotations,
        None
    ]



''' LLM '''
model = myChatOpenAI(
    temperature= 0.5
).with_structured_output(AblationCodeAnnotations)#, method='function_calling')



''' Helpful Functions '''
# To get the correct section of the code
def _slice_section(code: str, start_label: str, end_labels: list[str]) -> str:
    '''
    `_slice_section` slices the code between the start label and the end labels.

    `Args`:
        code (str): The code to slice.
        start_label (str): The start label.
        end_labels (list[str]): The end labels.

    `Returns`:
        str: The sliced code.
    '''
    q = r'["\']{3}'
    ws = r'[ \t]*'
    nl = r'(?:\r?\n|$)'

    # Get the start line
    start_re = re.compile(rf'{q}{ws}{start_label}{ws}{q}{ws}{nl}', re.IGNORECASE)
    m = start_re.search(code)
    if not m:
        return ''  # section not found
    
    # Get the end lines
    end_res = [re.compile(rf'{q}{ws}{lbl}{ws}{q}{ws}{nl}', re.IGNORECASE) for lbl in end_labels]
    s = m.end()
    e = len(code)
    # For each end line, get the closest start
    for er in end_res:
        em = er.search(code, s)
        if em:
            e = min(e, em.start())
    
    # Slice
    return code[s:e]



""" Nodes """
''' Docstring Nodes '''
def generate(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None
    
    try:
        # prompt
        prompt = (
            "You are a Code Annotator for a custom AI agent builder.\n"
            "\n"
            "Analyze the user's custom-agent specification, the proposed workflow, and the\n"
            "current Python code scaffold.\n"
            "\n"
            "Your task is to provide the specifications that a later coding agent will use\n"
            "to implement the custom AI agent.\n"
            "\n"
            "Produce all of the following in one response:\n"
            "\n"
            "1. NODE DOCSTRINGS\n"
            "\n"
            "Provide a docstring for every existing workflow node.\n"
            "\n"
            "Each node docstring should explain:\n"
            "- the purpose of the node;\n"
            "- the main processing steps;\n"
            "- the state keys the node reads;\n"
            "- the state keys the node returns or updates;\n"
            "- any helper functions it requires;\n"
            "- any tools an LLM inside the node requires.\n"
            "\n"
            "Do not create new workflow nodes.\n"
            "Do not change existing function names.\n"
            "Keep connected nodes compatible with each other's state inputs and outputs.\n"
            "\n"
            "2. SCHEMAS\n"
            "\n"
            "Define the schemas required by the workflow.\n"
            "\n"
            "Always include AgentSchema. AgentSchema is the main LangGraph state schema and\n"
            "must contain the state required by the complete workflow.\n"
            "\n"
            "Use MessagesState when persistent message history is required.\n"
            "Add other schemas only when useful for structured or complex information.\n"
            "\n"
            "For every schema field, provide an appropriate Python type and a concise\n"
            "description.\n"
            "\n"
            "3. HELPER FUNCTIONS\n"
            "\n"
            "Propose deterministic Python helper functions that are genuinely useful to\n"
            "existing nodes.\n"
            "\n"
            "Each helper function should include:\n"
            "- its name;\n"
            "- arguments and types;\n"
            "- return type;\n"
            "- a useful implementation docstring;\n"
            "- a short justification.\n"
            "\n"
            "The helper function docstring should explain:\n"
            "- Overview\n"
            "- Caller Node\n"
            "- Instructions\n"
            "- Args\n"
            "- Returns\n"
            "\n"
            "Do not create helper functions for:\n"
            "- prompt construction;\n"
            "- parsing normal LLM responses;\n"
            "- formatting normal LLM responses.\n"
            "\n"
            "Do not implement function bodies.\n"
            "\n"
            "4. TOOL FUNCTIONS\n"
            "\n"
            "Propose tools only when an LLM needs to perform an action or obtain information\n"
            "that it cannot produce itself.\n"
            "\n"
            "Examples include:\n"
            "- API requests;\n"
            "- web or database access;\n"
            "- file operations;\n"
            "- persistent storage;\n"
            "- external services;\n"
            "- computation that should be executed by code.\n"
            "\n"
            "Tools must be small and single-purpose.\n"
            "\n"
            "Each tool should include:\n"
            "- its name;\n"
            "- arguments and types;\n"
            "- return type;\n"
            "- a detailed docstring;\n"
            "- a short justification.\n"
            "\n"
            "The tool docstring should explain:\n"
            "- Overview\n"
            "- Caller LLM\n"
            "- Outside-the-Tool Work (Tool Handler Function Responsibilities)\n"
            "- Inside-the-Tool Work (Tool Responsibilities)\n"
            "- Instructions\n"
            "- State Updates (on the caller function)\n"
            "- Args\n"
            "- Returns\n"
            "\n"
            "A tool does not directly receive or modify the graph state.\n"
            "\n"
            "Do not implement tool bodies.\n"
            "\n"
            "5. LLM CONFIGURATION\n"
            "\n"
            "For every LLM definition already present in the code scaffold, provide exactly\n"
            "one configuration proposal.\n"
            "\n"
            "For each LLM:\n"
            "- preserve its existing variable name;\n"
            "- choose a temperature between 0 and 1;\n"
            "- use bind_tools only when that LLM genuinely needs one or more proposed tools;\n"
            "- use with_structured_output only when the LLM must return one of the proposed\n"
            "  schemas;\n"
            "- otherwise use neither.\n"
            "\n"
            "Prefer:\n"
            "- 0.0-0.3 for deterministic tasks;\n"
            "- 0.4-0.6 for balanced reasoning;\n"
            "- 0.7-1.0 for more creative generation.\n"
            "\n"
            "GENERAL RULES\n"
            "\n"
            "The generated agent uses Python, LangChain, and LangGraph.\n"
            "\n"
            "Treat the supplied workflow as the required workflow structure.\n"
            "\n"
            "Do not:\n"
            "- change the workflow structure;\n"
            "- change existing node names;\n"
            "- implement node bodies;\n"
            "- implement helper-function bodies;\n"
            "- implement tool bodies;\n"
            "- add unnecessary functionality;\n"
            "- ask the user questions.\n"
            "\n"
            "Keep schemas, node descriptions, helper functions, tools, and LLM\n"
            "configurations mutually consistent.\n"
            "\n"
            "If a minor implementation detail is unspecified, choose the simplest reasonable\n"
            "design that satisfies the user request and workflow.\n"
            "\n"
            "Custom agent specification:\n"
            "\n"
            "<USER_SPECIFICATION>\n"
            f"{state['clarified_user_input']}\n"
            "</USER_SPECIFICATION>\n"
            "\n"
            "Proposed workflow:\n"
            "\n"
            "<WORKFLOW>\n"
            f"{state['workflow']}\n"
            "</WORKFLOW>\n"
            "\n"
            "Current Python code scaffold:\n"
            "\n"
            "<CODE>\n"
            f"{read_state_file(state)}\n"
            "</CODE>\n"
        )
        # call the LLM
        proposal: AblationCodeAnnotations = safe_invoke(model, messages= [SystemMessage(content= prompt)])
        print(f'{GREEN}[NODE] [PROPOSAL]{RESET} {proposal}') if DEBUG else None

        # Return the new messages and the docstring proposed.
        return {'messages': [], 'step_changes': proposal}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

# Updates the docstrings of the code
def update_docstrings(state: InputSchema) -> InputSchema:
    '''
    Updates the docstrings of the existing node functions using the
    docstring proposals from the single ablation response.
    '''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    docstrings = state['step_changes'].docstrings

    # Read the code
    with open(file_path, 'r', encoding= 'utf-8') as f:
        code = f.read()

    # Get the section containing the node functions
    nodes = _slice_section(
        code,
        'Nodes',
        ['Conditional Functions', 'Graph']
    )

    # Map function names to generated docstrings
    docstring_map = {
        docstring.function: docstring.docstring
        for docstring in docstrings
    }

    # Build the updated node block
    new_nodes = []
    new_node = ''
    function_name = ''

    for line in nodes.split('\n'):

        # Start of a node function
        if line.startswith('def '):
            if new_node:
                new_nodes.append(new_node)

            new_node = line + '\n'
            function_name = line.split(' ')[1].split('(')[0]

        # Existing execution docstring
        elif line.strip().startswith('""" Execution: '):
            function_docstring = docstring_map.get(
                function_name,
                ''
            ).replace('\n', '\n\t')

            # Keep the existing Execution tag and append the generated annotation
            new_node += (
                line[:-3]
                + f'\n\t{function_docstring}'
                + '\n\t"""\n\n'
            )

        else:
            new_node += line + '\n'

    # Append the final node
    if new_node:
        new_nodes.append(new_node)

    new_nodes = '\n'.join(new_nodes)

    # Replace the old node section
    code = code.replace(nodes, new_nodes)

    # Write the updated code
    with open(file_path, 'w', encoding= 'utf-8') as f:
        f.write(code)

    # IMPORTANT:
    # Keep step_changes because later update nodes still need it.
    return {
        'step_changes': state['step_changes']
    }

# Updates the schemas
def update_schemas(state: InputSchema) -> InputSchema:
    '''
    Updates the schema section using the schemas generated in the
    single ablation response.
    '''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    schemas = state['step_changes'].schemas

    # Read the code
    with open(file_path, 'r', encoding= 'utf-8') as f:
        code = f.read()

    # Get the existing schema section
    old_schemas = _slice_section(
        code,
        'Schemas',
        ['Tools']
    )

    # Place AgentSchema last, matching the full Code Annotator
    rest_schemas = [
        str(schema)
        for schema in schemas
        if schema.schema_name != 'AgentSchema'
    ]

    agent_schema = [
        str(schema)
        for schema in schemas
        if schema.schema_name == 'AgentSchema'
    ]

    new_schemas = [''] + rest_schemas + agent_schema + ['']
    new_schemas = '\n'.join(new_schemas)

    # Replace schema section
    code = code.replace(
        old_schemas,
        new_schemas
    )

    # Write the updated code
    with open(file_path, 'w', encoding= 'utf-8') as f:
        f.write(code)

    # Keep the combined proposal for subsequent nodes
    return {
        'step_changes': state['step_changes']
    }

# Updates the helpful functions
def update_helpful_functions(state: InputSchema) -> InputSchema:
    '''
    Adds the helpful-function definitions generated in the
    single ablation response.
    '''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    helpful_functions = state['step_changes'].helpful_functions

    # Read the code
    with open(file_path, 'r', encoding= 'utf-8') as f:
        code = f.read()

    # Generate function definitions
    new_functions = [
        str(function)
        for function in helpful_functions
    ]

    new_functions = '\n\n'.join(new_functions)

    # Only add content when helper functions were proposed
    if new_functions:
        code = code.replace(
            "''' Helpful Functions '''",
            f"''' Helpful Functions '''\n{new_functions}",
            1
        )

    # Write the updated code
    with open(file_path, 'w', encoding= 'utf-8') as f:
        f.write(code)

    # Keep the combined proposal for subsequent nodes
    return {
        'step_changes': state['step_changes']
    }

# Updates the tool functions
def update_tool_functions(state: InputSchema) -> InputSchema:
    '''
    Adds the tool-function definitions generated in the
    single ablation response.
    '''
    print_function_name() if DEBUG else None

    file_path = state['file_path']
    tool_functions = state['step_changes'].tool_functions

    # Read the code
    with open(file_path, 'r', encoding= 'utf-8') as f:
        code = f.read()

    # Generate tool definitions
    new_functions = [
        '@tool\n' + str(function)
        for function in tool_functions
    ]

    new_functions = '\n\n'.join(new_functions)

    # Only modify the section when tools were proposed
    if new_functions:
        code = code.replace(
            "''' Tools '''",
            f"''' Tools '''\n{new_functions}",
            1
        )

    # Write the updated code
    with open(file_path, 'w', encoding= 'utf-8') as f:
        f.write(code)

    # Keep the combined proposal for the LLM modifier node
    return {
        'step_changes': state['step_changes']
    }

# Updates the LLM definitions
def update_llm_modifiers(state: InputSchema) -> InputSchema:
    '''
    Updates all existing LLM definitions using the LLM configuration
    proposals produced by the single ablation response.
    '''
    print_function_name() if DEBUG else None

    file_path = state['file_path']

    # Extract only the LLM proposals from the combined response
    llm_proposals = state['step_changes'].llm_proposals

    # Reuse the existing LLMProposalList rendering logic
    proposal_list = LLMProposalList(
        thinking_process='',
        llm_proposals=llm_proposals
    )

    proposed_llm_definitions = proposal_list.to_code()

    # Read the code
    with open(file_path, 'r', encoding= 'utf-8') as f:
        code = f.read()

    # Replace the LLM section
    old_code = _slice_section(
        code,
        'LLM',
        ['Helpful Functions', 'Nodes']
    )

    code = code.replace(
        old_code,
        proposed_llm_definitions
    )

    # Write the updated code
    with open(file_path, 'w', encoding= 'utf-8') as f:
        f.write(code)

    return {
        'step_changes': state['step_changes']
    }


''' Graph '''
code_annotator_graph = StateGraph(InputSchema)

code_annotator_graph.add_node('generate', generate)
code_annotator_graph.add_node('update_docstrings', update_docstrings)
code_annotator_graph.add_node('update_schemas', update_schemas)
code_annotator_graph.add_node('update_helpful_functions', update_helpful_functions)
code_annotator_graph.add_node('update_tool_functions', update_tool_functions)
code_annotator_graph.add_node('update_llm_modifiers', update_llm_modifiers)

code_annotator_graph.add_edge(START, 'generate')
code_annotator_graph.add_edge('generate', 'update_docstrings')
code_annotator_graph.add_edge('update_docstrings', 'update_schemas')
code_annotator_graph.add_edge('update_schemas', 'update_helpful_functions')
code_annotator_graph.add_edge('update_helpful_functions', 'update_tool_functions')
code_annotator_graph.add_edge('update_tool_functions', 'update_llm_modifiers')
code_annotator_graph.add_edge('update_llm_modifiers', END)

code_annotator_app = code_annotator_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(code_annotator_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/code_annotator_app.png', 'wb') as f:
        f.write(code_annotator_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'codeAnnotator'
    os.environ['LANGSMITH_PROJECT'] = 'codeAnnotator'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'codeAnnotator',
            'run_name': 'codeAnnotator',
            'thread_id': 'codeAnnotator', 
        }
    }

    from test_inputs import file_path, clarified_user_input, workflow
    user = InputSchema(
        file_path= file_path,
        clarified_user_input= clarified_user_input,
        workflow= workflow
    )

    response = code_annotator_app.invoke(user, config= config)
