"""
- `author:` Stefanos Panteli
- `date:` 2025-10-26
- `description:` The code clarifier agent. Sees the the proposed code structure and adds definitions and instructions for nodes, schemas, helpful functions and tool functions.

## How to use
1. Import the app. (`from agents.codeAnnotator.code_annotator import code_annotator_app`)
2. Input a dict with the following keys:
    - `file_path: str`: The path of the file to annotate.
    - `clarified_user_input: str`: The user input refined from the input refiner.
    - `workflow: WorkflowBundle`: The proposed workflow from the workflow refiner.
    - `step_changes: Optional[Union]`: Should not be provided by the user. 
3. Invoke the app.
4. Get the output dict with the following keys:
    - `file_path: str`: The path of the file to annotate.
    - `clarified_user_input: str`: The user input refined from the input refiner.
    - `workflow: WorkflowBundle`: The proposed workflow from the workflow refiner.
    - `step_changes: Optional[Union]`: Should not be provided by the user. 
The output of this agent does not matter, its purpose is to annotate the code.

## Usage
```python
from agents.codeAnnotator.code_annotator import code_annotator_app
graph_input = {
    'file_path': "Clone\creations\fitness_program_generator\fitness_program_generator.py", 
    clarified_user_input: <Output of the input refiner>,
    workflow: <Output of the workflow refiner>
}

response = code_annotator_app.invoke(graph_input)

# response = { # Does not matter
#     'file_path': "Clone\creations\fitness_program_generator\fitness_program_generator.py", 
#     clarified_user_input: <Output of the input refiner>,
#     workflow: <Output of the workflow refiner>
# }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, RemoveMessage

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
import json
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, USER_APPROVALS, read_state_file
from agents.workflowRefiner.workflow_refiner import WorkflowBundle
from agents.codeAnnotator import prompts



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



''' Input Schema '''
class InputSchema(MessagesState):
    file_path: str # The current file path as given from the user.
    clarified_user_input: str # The clarified user input as given from the clarifier.
    workflow: WorkflowBundle # The proposed workflow as given from the workflow engineer.

    # The step changes as given from the LLM
    step_changes: Union[    
        Docstrings,
        Schemas,
        HelpfulFunctions,
        ToolFunctions,
        LLMProposalList,
        None
    ]



''' LLM '''
docstring_generator = myChatOpenAI(
    temperature= 0.5
).with_structured_output(Docstrings)

schema_generator = myChatOpenAI(
    temperature= 0.2
).with_structured_output(Schemas)

helpful_function_generator = myChatOpenAI(
    temperature= 0.2
).with_structured_output(HelpfulFunctions)

tool_function_generator = myChatOpenAI(
    temperature= 0.2
).with_structured_output(ToolFunctions)

tool_or_output_generator = myChatOpenAI(
    temperature= 0.8
).with_structured_output(LLMProposalList)



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
# The node that understands the code and comments the node functions
def generate_docstrings(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and generates docstrings for the node functions in the code.
    The docstrings include: Overview, Instrictions, Input keys, Output keys, Helpful functions and Tool functions.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.ANNOTATE_NODES_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= read_state_file(state),
            history= history
        )

        # call the LLM
        docstring_proposal: Docstrings = safe_invoke(docstring_generator, messages= [SystemMessage(content= prompt)]) # TODO: maybe pass messages to the list of messages

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [DOCSTRING PROPOSAL]{RESET} {docstring_proposal if docstring_proposal.docstrings else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(docstring_proposal)), HumanMessage(content= user_input)]

        # Return the new messages and the docstring proposed.
        return {'messages': new_messages, 'step_changes': docstring_proposal}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
# Updates the docstring of the code
def update_docstrings(state: InputSchema) -> InputSchema:
    '''
    This node retrieves the proposed docstrings from the previous node and updates the docstrings in the code.
    '''
    print_function_name() if DEBUG else None

    # Get the file path and the proposed docstrings
    file_path = state['file_path']
    docstrings = state['step_changes'].model_dump()['docstrings']

    # Read the code
    with open(file_path, 'r') as f:
        code = f.read()

    # Get the section of the code which contains the nodes (between ''' Nodes ''' and ''' Conditional Functions ''' or ''' Graph ''' when conditional functions don't exist)
    nodes = _slice_section(code, 'Nodes', ['Conditional Functions', 'Graph'])

    # Build the new node block, with the docstrings
    new_nodes = []
    new_node = function_name = ''
    # For each line of code
    for line in nodes.split('\n'):
        # Signature line
        if line.startswith('def '):
            # Close previous node
            new_nodes.append(new_node) if new_node else None
            # Start new node
            new_node = line + '\n'
            # Get function name and generated docstring
            function_name = line.split(' ')[1].split('(')[0]
            for docstring in docstrings: 
                if docstring['function'] == function_name:
                    function_docstring = docstring['docstring'].replace('\n', '\n\t')
                    break
            else:
                function_docstring = ''
        # Docstring line
        elif line.strip().startswith('""" Execution: '):
            # Keep the original docstring, remove trailing """, and add the generated docstring
            new_node += line[:-3] + f'\n\t{function_docstring}\n\t"""\n\n'
        # Any other line
        else:
            # Keep the line
            new_node += line + '\n'

    # Append the last node
    new_nodes.append(new_node)
    # Get the whole section as a string
    new_nodes = '\n'.join(new_nodes)
    # Replace the section
    code = code.replace(nodes, new_nodes)

    # Update the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}


''' Schema Nodes '''
# Proposes schemas
def propose_schemas(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and generates any needed schemas. Always includes the AgentSchema.
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.PROPOSE_SCHEMAS_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= read_state_file(state),
            history= history
        )

        # call the LLM
        schemas_proposal: Schemas = safe_invoke(schema_generator, messages= [SystemMessage(content= prompt)]) # TODO: maybe pass messages to the list of messages

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [SCHEMAS PROPOSAL]{RESET} {schemas_proposal if schemas_proposal.schemas else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(schemas_proposal)), HumanMessage(content= user_input)]

        # Return the new messages and the schemas proposed.
        return {'messages': new_messages, 'step_changes': schemas_proposal}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

# Updates the schemas
def update_schemas(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and updates the schemas in the code.
    '''
    print_function_name() if DEBUG else None

    # Get the file path and the proposed schemas
    file_path = state['file_path']
    schemas = state['step_changes'].schemas

    # Read the code
    with open(file_path, 'r') as f:
        code = f.read()

    # Get the section of the code which contains the schemas
    old_schemas = _slice_section(code, 'Schemas', ['Tools'])

    # Read the schemas, placing the AgentSchema last
    rest_schemas = [str(schema) for schema in schemas if schema.schema_name != 'AgentSchema']
    agent_schema = [str(schema) for schema in schemas if schema.schema_name == 'AgentSchema']
    new_schemas = [''] + rest_schemas + agent_schema + ['']
    new_schemas = '\n'.join(new_schemas)
    
    # Replace the section
    code = code.replace(old_schemas, new_schemas)

    # Update the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}


''' Helpful Functions Nodes '''
# The node that reads the annotated code, and adds definitions and instructions for helpful functions
def propose_helpful_functions(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and generates any needed helpful functions. Gets its data mainly from the annotated code.
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.ADD_HELPFUL_FUNCTIONS_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= read_state_file(state),
            history= history
        )

        # call the LLM
        helpful_functions_proposal: HelpfulFunctions = safe_invoke(helpful_function_generator, messages= [SystemMessage(content= prompt)]) # TODO: maybe pass messages to the list of messages

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [HELPFUL FUNCTIONS PROPOSAL]{RESET} {helpful_functions_proposal if helpful_functions_proposal.helpful_functions else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(helpful_functions_proposal)), HumanMessage(content= user_input)]

        # Return the new messages and the helpful functions proposed.
        return {'messages': new_messages, 'step_changes': helpful_functions_proposal}
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

# Updates the helpful functions generated
def update_helpful_functions(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and updates the helpful functions in the code.
    '''
    print_function_name() if DEBUG else None

    # Get the file path and the proposed helpful functions
    file_path = state['file_path']
    helpful_functions = state['step_changes'].helpful_functions

    # Read the code
    with open(file_path, 'r') as f:
        code = f.read()

    # Parse and relace the helpful function section
    new_functions = [str(function) for function in helpful_functions]
    new_functions = '\n\n'.join(new_functions)
    code = code.replace("''' Helpful Functions '''", f"''' Helpful Functions '''\n{new_functions}")

    # Update the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}


''' Tool Functions Nodes '''
# The node that reads the annotated code, and adds definitions and instructions for tool functions
def propose_tool_functions(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and generates any needed tool functions. Gets its data mainly from the annotated code.
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])

        prompt = prompts.ADD_TOOL_FUNCTIONS_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            workflow= state['workflow'],
            code_structure= read_state_file(state),
            history= history
        )

        # call the LLM
        tool_functions_proposal: ToolFunctions = safe_invoke(tool_function_generator, messages= [SystemMessage(content= prompt)]) # TODO: maybe pass messages to the list of messages

        # Ask the user to confirm
        print(f'{GREEN}[NODE] [TOOL FUNCTIONS PROPOSAL]{RESET} {tool_functions_proposal if tool_functions_proposal.tool_functions else "None"}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= str(tool_functions_proposal)), HumanMessage(content= user_input)]

        # Return the new messages and the tool functions proposed.
        return {'messages': new_messages, 'step_changes': tool_functions_proposal}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

# Updates the tool functions
def update_tool_functions(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and updates the tool functions in the code.
    '''
    print_function_name() if DEBUG else None

    # Get the file path and the proposed tool functions
    file_path = state['file_path']
    tool_functions = state['step_changes'].tool_functions

    # Read the code
    with open(file_path, 'r') as f:
        code = f.read()

    # Parse and relace the tool function section
    new_functions = ['@tool\n' + str(function) for function in tool_functions]
    new_functions = '\n\n'.join(new_functions)
    code = code.replace("''' Tools '''", f"''' Tools '''\n{new_functions}")

    # Update the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}


''' bind_tools or with_structured_output Functions '''
# The node that reads the annotated code, and uses the necessary bind_tools or with_structured_output methods
def propose_llm_modifiers(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and proposes any needed bind_tools or with_structured_output methods. Gets its data mainly from the annotated code.
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        history = '\n\n---\n\n'.join([mes.pretty_repr() for mes in state.get('messages', [])])
        code_structure = read_state_file(state)
        # Get the LLM definitions
        llm_section = _slice_section(code_structure, 'LLM',   ['Helpful Functions', 'Nodes'])
        llm_definitions = [line.split(' = ')[0] for line in re.findall(r'(\w+) = myChatOpenAI\(', llm_section)]
        # Get the tool names
        tool_section = _slice_section(code_structure, 'Tools', ['LLM', 'Nodes']).split('\n')
        tool_names = [line.split('def ')[1].split('(')[0] for line in tool_section if line.startswith('def ')]
        # Get the schema names
        schema_section = _slice_section(code_structure, 'Schemas', ['Tools', 'LLM']).split('\n')
        schema_names = [line.split('class ')[1].split('(')[0] for line in schema_section if line.startswith('class ')]
        schema_names = [schema_name for schema_name in schema_names if schema_name != 'AgentSchema']

        prompt = prompts.ADD_LLM_METHODS_PROMPT.format(
            clarified_user_input= state['clarified_user_input'],
            # workflow= state['workflow'],
            code_structure= code_structure,
            history= history,
            llm_definitions= ', '.join(llm_definitions),
            tool_names= ', '.join(tool_names),
            schema_names= ', '.join(schema_names)
        )

        # call the LLM
        llm_method_proposals: LLMProposalList = safe_invoke(tool_or_output_generator, messages= [SystemMessage(content= prompt)]) # TODO: maybe pass messages to the list of messages

        # Check if the proposed LLMs match the LLMs in the code
        proposed_llms = llm_method_proposals.get_all_llm_names()
        llm_not_defined, llm_not_proposed = [llm for llm in proposed_llms if llm not in llm_definitions], [llm for llm in llm_definitions if llm not in proposed_llms]
        if llm_not_defined or llm_not_proposed:
            ndef = nprop = ''
            if llm_not_defined:
                ndef = f' The following defined LLMs were not returned by the agent: {", ".join(llm_not_defined)}.'
            if llm_not_proposed:
                nprop = f' The following LLMs were proposed by the agent, but they are not defined in the code: {", ".join(llm_not_proposed)}.'
            
            print(f'{RED}[NODE] [WARNING]{RESET}{ndef}{nprop}') if DEBUG else None
            human_message = HumanMessage(content= f'You did the following errors:{ndef}{nprop}')
            return {'messages': [AIMessage(content= json.dumps(llm_method_proposals.model_dump())), human_message], 'step_changes': llm_method_proposals}
        
        # Check whether the proposed with_structured_output are in the code
        proposed_schema_bindings = llm_method_proposals.get_all_schema_names()
        wrong_schemas = [schema for schema in proposed_schema_bindings if schema not in schema_names]
        if wrong_schemas:
            w = ', '.join(wrong_schemas)
            print(f'{RED}[NODE] [WARNING]{RESET} The agent added an undefined schema ({w})') if DEBUG else None
            human_message = HumanMessage(content= f'You added an undefined schema ({w})')
            return {'messages': [AIMessage(content= json.dumps(llm_method_proposals.model_dump())), human_message], 'step_changes': llm_method_proposals}
        
        
        # Check whether the proposed bind_tools are in the code
        proposed_tool_bindings = llm_method_proposals.get_all_tool_names()
        wrong_tools = [tool for tool in proposed_tool_bindings if tool not in tool_names and tool not in schema_names]
        if wrong_tools:
            w = ', '.join(wrong_tools)
            print(f'{RED}[NODE] [WARNING]{RESET} The agent added an undefined tool ({w})') if DEBUG else None
            human_message = HumanMessage(content= f'You added an undefined tool ({w})')
            return {'messages': [AIMessage(content= json.dumps(llm_method_proposals.model_dump())), human_message], 'step_changes': llm_method_proposals}

        # All good
        # Ask the user to confirm
        print(f'{GREEN}[NODE] [LLM METHODS PROPOSAL]{RESET} \n{llm_method_proposals.to_string()}')
        
        user_input = input(f'\n{GREEN}[NODE] [CONFIRMATION]{RESET} Is this okay, if not please insert your request (y/request) > ') 
        new_messages = [AIMessage(content= json.dumps(llm_method_proposals.model_dump())), HumanMessage(content= user_input)]

        # Return the new messages and the tool functions proposed.
        return {'messages': new_messages, 'step_changes': llm_method_proposals}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
# Updates the LLM definitions
def update_llm_modifiers(state: InputSchema) -> InputSchema:
    '''
    This node reads the code structure up to now and updates the LLM definitions in the code with the necessary bind_tools or with_structured_output methods.
    '''
    print_function_name() if DEBUG else None

    # Get the file path and the proposed tool functions
    file_path = state['file_path']
    proposed_llm_definitions = state['step_changes'].to_code()

    # Read the code
    with open(file_path, 'r') as f:
        code = f.read()

    # Parse and relace the tool function section
    old_code = _slice_section(code, 'LLM', ['Helpful Functions', 'Nodes'])
    code = code.replace(old_code, proposed_llm_definitions)

    # Update the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Remove all messages and step changes
    remove_messages = [RemoveMessage(id= mes.id) for mes in state.get('messages', [])]
    return {'messages': remove_messages, 'step_changes': None}



''' Conditional Functions '''
# The conditional logic is used to determine what to do after docstring generation
def after_docstrings(state: InputSchema) -> Literal['generate_docstrings', 'update_docstrings']:
    '''
    This conditional function is used to determine what to do after docstring generation.
    If the user approves, then the docstrings are updated. Else the docstrings are generated again.
    '''
    print_function_name() if DEBUG else None

    try:
        # If an error occured and the user did not provide any input, go back
        last_user_message = state['messages'][-1].content
    except IndexError:
        return 'generate_docstrings'
    
    # User approval
    if last_user_message.lower() in USER_APPROVALS:
        return 'update_docstrings'
    
    return 'generate_docstrings'

# The conditional logic is used to determine what to do after schema generation
def after_schemas(state: InputSchema) -> Literal['propose_schemas', 'update_schemas']:
    '''
    This conditional function is used to determine what to do after schema generation.
    If the user approves, then the schemas are updated. Else the schemas are generated again.
    '''
    print_function_name() if DEBUG else None

    try:
        # If an error occured and the user did not provide any input, go back
        last_user_message = state['messages'][-1].content
    except IndexError:
        return 'propose_schemas'
    
    # User approval
    if last_user_message.lower() in USER_APPROVALS:
        return 'update_schemas'
    
    return 'propose_schemas'

# The conditional logic is used to determine what to do after helpful functions
def after_helpful_functions(state: InputSchema) -> Literal['propose_helpful_functions', 'update_helpful_functions']:
    '''
    This conditional function is used to determine what to do after helpful functions.
    If the user approves, then the helpful functions are updated. Else the helpful functions are generated again.
    '''
    print_function_name() if DEBUG else None

    try:
        # If an error occured and the user did not provide any input, go back
        last_user_message = state['messages'][-1].content
    except IndexError:
        return 'propose_helpful_functions'
    
    # User approval
    if last_user_message.lower() in USER_APPROVALS:
        return 'update_helpful_functions'
    
    return 'propose_helpful_functions'

# The conditional logic is used to determine what to do after tool functions
def after_tool_functions(state: InputSchema) -> Literal['propose_tool_functions', 'update_tool_functions']:
    '''
    This conditional function is used to determine what to do after tool functions.
    If the user approves, then the tool functions are updated. Else the tool functions are generated again.
    '''
    print_function_name() if DEBUG else None

    try:
        # If an error occured and the user did not provide any input, go back
        last_user_message = state['messages'][-1].content
    except IndexError:
        return 'propose_tool_functions'
    
    # User approval
    if last_user_message.lower() in USER_APPROVALS:
        return 'update_tool_functions'
    
    return 'propose_tool_functions'

# The conditional logic is used to determine what to do after llm changes
def after_llm_modifiers(state: InputSchema) -> Literal['propose_llm_modifiers', 'update_llm_modifiers']:
    '''
    This conditional function is used to determine what to do after llm changes.
    If the user approves, then the llm changes are updated. Else the llm changes are generated again.
    '''
    print_function_name() if DEBUG else None

    try:
        # If an error occured and the user did not provide any input, go back
        last_user_message = state['messages'][-1].content
    except IndexError:
        return 'propose_llm_modifiers'
    
    # User approval
    if last_user_message.lower() in USER_APPROVALS:
        return 'update_llm_modifiers'
    
    return 'propose_llm_modifiers'



''' Graph '''
code_annotator_graph = StateGraph(InputSchema)

# Logic: Docstrings -> Schemas -> Helpful Functions -> Tool Functions -> LLM Modifiers
code_annotator_graph.add_node('generate_docstrings', generate_docstrings)
code_annotator_graph.add_node('update_docstrings', update_docstrings)
code_annotator_graph.add_node('propose_schemas', propose_schemas)
code_annotator_graph.add_node('update_schemas', update_schemas)
code_annotator_graph.add_node('propose_helpful_functions', propose_helpful_functions)
code_annotator_graph.add_node('update_helpful_functions', update_helpful_functions)
code_annotator_graph.add_node('propose_tool_functions', propose_tool_functions)
code_annotator_graph.add_node('update_tool_functions', update_tool_functions)
code_annotator_graph.add_node('propose_llm_modifiers', propose_llm_modifiers)
code_annotator_graph.add_node('update_llm_modifiers', update_llm_modifiers)

# Start -> Docstrings
code_annotator_graph.add_edge(START, 'generate_docstrings')
code_annotator_graph.add_conditional_edges(
    'generate_docstrings',
    after_docstrings,
    {   # Not needed, for clarity
        'generate_docstrings': 'generate_docstrings',
        'update_docstrings': 'update_docstrings'
    }
)
# Docstrings -> Schemas
code_annotator_graph.add_edge('update_docstrings', 'propose_schemas')
code_annotator_graph.add_conditional_edges(
    'propose_schemas',
    after_schemas,
    {   # Not needed, for clarity
        'propose_schemas': 'propose_schemas',
        'update_schemas': 'update_schemas'
    }
)
# Schemas -> Helpful Functions
code_annotator_graph.add_edge('update_schemas', 'propose_helpful_functions')
code_annotator_graph.add_conditional_edges(
    'propose_helpful_functions',
    after_helpful_functions,
    {   # Not needed, for clarity
        'propose_helpful_functions': 'propose_helpful_functions',
        'update_helpful_functions': 'update_helpful_functions'
    }
)
# Helpful Functions -> Tool Functions
code_annotator_graph.add_edge('update_helpful_functions', 'propose_tool_functions')
code_annotator_graph.add_conditional_edges(
    'propose_tool_functions',
    after_tool_functions,
    {   # Not needed, for clarity
        'propose_tool_functions': 'propose_tool_functions',
        'update_tool_functions': 'update_tool_functions'
    }
)
code_annotator_graph.add_edge('update_tool_functions', 'propose_llm_modifiers')
# Tool Functions -> LLM Modifiers -> END
code_annotator_graph.add_conditional_edges(
    'propose_llm_modifiers',
    after_llm_modifiers,
    {   # Not needed, for clarity
        'propose_llm_modifiers': 'propose_llm_modifiers',
        'update_llm_modifiers': 'update_llm_modifiers'
    }
)
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
