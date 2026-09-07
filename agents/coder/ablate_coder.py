''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

# Schema imports
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Annotated, Tuple
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, read_state_file



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Coder') if DEBUG else None



""" Schemas """
''' General Schemas '''     
# Used at the output_tool. The agent requests from the Software Engineer a function to implement.
class FunctionProposal(BaseModel):
    class Argument(BaseModel):
        name: str = Field(description= 'The name of the argument.')
        type: str = Field(description= 'The type of the argument.')

        def __str__(self):
            return f'{self.name}: {self.type}'
   
    function_type: Literal['helper_function', 'tool'] = Field(description= 'The type of the function you need.')
    function_name: str = Field(description= 'The name of the function.')
    docstring: str = Field(description= 'The docstring of the function.')
    function_arguments: List[Argument] = Field(description= 'The arguments of the function.')
    output: str = Field(description= 'The return type of the function.')
    justification: str = Field(description= 'The justification of the function.')

    def __str__(self):
        if self.function_type == 'tool': 
            tool = '@tool\n'
        else:
            tool = ''
        docstring = self.docstring.replace('\n', '\n\t')
        arguments = ', '.join([str(arg) for arg in self.function_arguments])
        return f'{tool}def {self.function_name}({arguments}) -> {self.output}:\n\t"""\n\t{docstring}\n\t"""\n\t...'

''' Input Schema '''
class InputSchema(MessagesState):
    file_path: str # The path to the file.
    function_name: str # The name of the function.
    software_engineer_instructions: str # The instructions from the software engineer.

    previous_outputs: Annotated[List['OutputSchema'], add] # The previous outputs you provided.
    comments: Annotated[List[str], add] # The comments the software engineer provided.

    previous_implementation: Optional['OutputSchema']
    reviewer_comments: Optional[str]

    run_code: bool

''' Output Schema '''
class OutputSchema(BaseModel):
    code: str = Field(description= 'The implemented code of the function.')
    proposals: Optional[List[FunctionProposal]] = Field(description= 'The requested proposals.', default= None)
    imports: Optional[List[str]] = Field(description= 'The requested imports that dont already exist. Every string should correspond to a line of `from * import *` or `import *`.', default= None)

    # Validate that imports at least contain the `import` keyword
    @field_validator('imports')
    @classmethod
    def validate_mood(cls, value):
        if value:
            for import_line in value:
                if not 'import' in import_line:
                    raise ValueError("imports should contain the `import` keyword")
                
        return value



''' LLM '''
model = myChatOpenAI(
    temperature= 0.5    
).with_structured_output(OutputSchema)#, method='function_calling')



''' Helpful Functions '''
def get_schema_type(state: InputSchema) -> Tuple[str, str]:
    '''
    `get_schema_type` returns the schema type of the state.
    
    `Args:`
        state (InputSchema): The state of the agent. Must have the key 'messages'.

    `Returns:`
        (Tuple[str, str]): The schema type of the state, and how it should be called.
    '''
    code = read_state_file(state)
    # If the AgentSchema is a BaseModel
    if 'AgentSchema(BaseModel):' in code:
        return ('`BaseModel`', '`state.key_name`')
    
    # If the AgentSchema is a MessagesState
    elif 'AgentSchema(MessagesState):' in code:
        return ('`MessagesState`', '`state[\'key_name\']` or `schema.get(\'key_name\', default)`')
    
    # If the AgentSchema is a TypedDict
    return ('`TypedDict`', '`state[\'key_name\']` or `schema.get(\'key_name\', default)`')



''' Nodes '''
def coder_node(state: InputSchema) -> OutputSchema:
    print_function_name() if DEBUG else None

    try:
        code = read_state_file(state)
        schema_type, schema_call = get_schema_type(state)

        history = '\n\n---\n\n'.join([
            f'Output number {i}:\n{previous_output}\nSoftware Engineer comment:\n{comment}'
            for i, (previous_output, comment) in enumerate(zip(state['previous_outputs'], state['comments']), start= 1)
        ])

        previous = ''
        if state['previous_implementation']:
            previous = f"\n# Previous Implementation\n<PREVIOUS_IMPLEMENTATION>\n{state['previous_implementation'].code}\n</PREVIOUS_IMPLEMENTATION>\n\n# Reviewer Comments\n<REVIEWER_COMMENTS>\n{state['reviewer_comments'] or ''}\n</REVIEWER_COMMENTS>\n"

        prompt = (
            "You are a Coder responsible for implementing exactly one Python function in an existing custom AI agent codebase.\n"
            f"You must implement only the function named `{state['function_name']}` and must not implement, rewrite, or modify any other function.\n"
            "Use the complete provided codebase as context and as the primary source of truth for surrounding schemas, functions, tools, LLMs, graph structure, imports, naming, and conventions.\n"
            "Follow the Software Engineer's special instructions closely and give them priority whenever they add implementation requirements that are consistent with the codebase.\n"
            "Use the target function's existing signature and docstring as its main implementation specification.\n"
            "Return the complete implementation of the target function including its existing decorator if one belongs to that function.\n"
            "Do not return the complete Python file and do not return unrelated code.\n"
            "Do not include imports inside the code field.\n"
            "If the implementation requires imports that are not already present in the provided file, return those import statements separately in the imports field.\n"
            "Do not request an import that already exists in the provided code.\n"
            "Every import returned in the imports field must be a complete valid Python import statement containing the `import` keyword.\n"
            "Whenever practical, use type annotations for local variables when they improve readability and make state or data types clearer.\n"
            "Preserve the existing function docstring unless a small adjustment is required to keep it consistent with the implementation.\n"
            "Do not change the target function's signature unless the Software Engineer explicitly requires it or a minimal correction is necessary for consistency with the existing code.\n"
            "Do not invent new state fields, workflow stages, graph nodes, APIs, files, or requirements that are not supported by the codebase or Software Engineer instructions.\n"
            "If a helper function or tool is genuinely required but does not exist, do not implement it inside the target function and instead return a FunctionProposal in the proposals field.\n"
            "Only propose a helper function or tool when it materially improves correctness or is necessary to implement the requested function.\n"
            "A helper-function proposal must describe deterministic reusable Python functionality that should be called directly by ordinary code.\n"
            "A tool proposal must describe functionality that an LLM needs to request for external information, external actions, storage, APIs, file operations, search, or another side effect that should occur outside model reasoning.\n"
            "For each proposal provide a precise function name, function type, arguments, return type, implementation-oriented docstring, and justification.\n"
            "Do not propose prompt-building helpers, normal LLM-response parsers, or trivial wrappers that can be implemented directly inside the target function.\n"
            "The generated code uses Python, LangChain, and LangGraph, so the implementation must follow their runtime conventions.\n"
            f"The AgentSchema in this file is a {schema_type} and state fields should be accessed using the appropriate style such as `{schema_call}`.\n"
            "If AgentSchema is a BaseModel, access state fields as attributes, and if it is TypedDict or MessagesState, access fields using dictionary-style access or `.get()` where appropriate.\n"
            "Read only state keys that exist in AgentSchema and return state updates that match the types and fields defined by AgentSchema.\n"
            "If AgentSchema inherits MessagesState, remember that the `messages` field is already provided by MessagesState.\n"
            "Respect `add_messages` or other reducers defined on state fields and return update values in the form expected by those reducers.\n"
            "Use LangChain message types such as HumanMessage, AIMessage, and ToolMessage where the surrounding code and state schema require them.\n"
            "When the target function calls an LLM, use the LLM instance already defined in the code and respect its `.bind_tools()` or `.with_structured_output()` configuration.\n"
            "If an LLM uses `.with_structured_output(SomeSchema)`, treat the result as the parsed schema object rather than assuming that it is a normal AIMessage with a `.content` field.\n"
            "If an LLM uses `.bind_tools(...)`, the LLM node may request tools but the normal LLM node must not directly execute those bound tools.\n"
            "A normal LLM workflow node may call `safe_invoke` on the model and return the AIMessage or derived state update, but actual bound-tool invocation belongs only in a ToolNode or custom tool-handler function.\n"
            "Do not call a tool decorated with `@tool` as a normal Python callable because LangChain tools should be executed with `.invoke()` when manually executed.\n"
            "You may use `.invoke(args_dict)` on a tool only when the target function is an actual custom tool-handler function responsible for tool execution.\n"
            "If the target function is a custom tool handler, extract tool calls from the latest AIMessage, support tool calls stored in either `.tool_calls` or `.additional_kwargs['tool_calls']`, dispatch each call to the correct tool, parse string arguments when needed, invoke the tool, create ToolMessage objects where required, perform documented non-message state updates, and return the correct state-update dictionary.\n"
            "If the target function is a custom tool handler and several tools may be handled, create or use a local tool-name mapping only when needed and only from tools already defined in the provided code.\n"
            "If the target function is a normal LangGraph node that uses an LLM with tools, do not invoke the tools inside that node and allow the graph's existing ToolNode or custom handler to execute them.\n"
            "If the target function is a routing or conditional function, return only node names or route values already supported by the graph's corresponding conditional-edge mapping.\n"
            "If the target function is a routing function, correctly distinguish tool calls, normal messages, state conditions, terminal routes, and fallbacks according to the existing graph design.\n"
            "Do not introduce an infinite self-loop or a route to a node that does not exist in the graph.\n"
            "If the target function is a tool, implement only the work assigned to the tool itself and do not move caller or tool-handler state responsibilities into the tool.\n"
            "A tool should normally receive only simple JSON-serializable arguments and should not receive the complete LangGraph state unless the existing function signature explicitly requires otherwise.\n"
            "A tool should return a compact result that the calling LLM or handler can use and should handle expected failures in a controlled way when appropriate.\n"
            "If the target function is a helper function, implement deterministic reusable logic and do not introduce LLM or tool behavior unless the existing specification explicitly requires it.\n"
            "If the target function performs file operations, use the file paths and conventions already established by the surrounding code and do not invent unrelated external paths.\n"
            "If the target function uses an external file already referenced elsewhere in the codebase, follow the same path conventions used by those existing functions.\n"
            "If the target function uses `safe_invoke`, do not provide the same conversation context twice by formatting the full message history into the prompt and also passing the same full history separately to `safe_invoke` unless the existing design explicitly requires both.\n"
            "Use existing project utilities such as `safe_invoke`, `print_function_name`, `will_tool_call`, `parse_tool_arguments`, and `USER_APPROVALS` when they are already imported and appropriate for the implementation.\n"
            "Treat functions and classes imported from project modules as implemented and available and do not recreate them inside the target function.\n"
            "Do not add inline imports inside the function body under any circumstances.\n"
            "Do not use unsafe `eval` or `exec` unless the existing function specification explicitly requires such behavior and no safer alternative exists.\n"
            "Avoid command injection, unsafe arbitrary file access, secret exposure, uncontrolled external requests, and other obvious security problems.\n"
            "Handle realistic errors when the target function's responsibilities require error handling, but do not add excessive defensive code for impossible states that are already guaranteed by Pydantic or the existing schema contract.\n"
            "Assume arguments produced through Pydantic-validated structured output or tool schemas have already passed their declared schema validation.\n"
            "The implementation must be syntactically valid, logically correct, LangGraph compatible, consistent with the surrounding code, and free of unresolved names other than any imports or proposals explicitly returned in their corresponding output fields.\n"
            "Do not leave the target function unimplemented with `pass`, `...`, TODO-only logic, or placeholder return values.\n"
            "Do not include markdown fences, commentary, analysis, or explanations in the code field.\n"
            "If no additional helper functions or tools are required, return `proposals=None` or an empty list.\n"
            "If no additional imports are required, return `imports=None` or an empty list.\n"
            "\n"
            "# Complete Codebase\n"
            "<CODE_START>\n"
            f"{code}\n"
            "</CODE_END>\n"
            "\n"
            "# Function to Implement\n"
            f"{state['function_name']}\n"
            "\n"
            "# Software Engineer Instructions\n"
            "<SOFTWARE_ENGINEER_INSTRUCTIONS>\n"
            f"{state['software_engineer_instructions']}\n"
            "</SOFTWARE_ENGINEER_INSTRUCTIONS>\n"
            "\n"
            "# Previous Outputs and Software Engineer Comments\n"
            "<IMPLEMENTATION_HISTORY>\n"
            f"{history}\n"
            "</IMPLEMENTATION_HISTORY>\n"
            f"{previous}"
        )

        response: OutputSchema = safe_invoke(model, messages=[SystemMessage(content=prompt)])

        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        return response

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        if state['previous_implementation']:
            return state['previous_implementation']

        return OutputSchema(code='', proposals=None, imports=None)




''' Graph '''
coder_graph = StateGraph(InputSchema, output_schema= OutputSchema)

coder_graph.add_node('coder_node', coder_node)

coder_graph.add_edge(START, 'coder_node')
coder_graph.add_edge('coder_node', END)

coder_app = coder_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(coder_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/coder_app.png', 'wb') as f:
        f.write(coder_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'coder'
    os.environ['LANGSMITH_PROJECT'] = 'coder'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'coder',
            'run_name': 'coder',
            'thread_id': 'coder', 
        }
    }

    user = InputSchema(
        messages= [],
        file_path= './test.py',
        function_name= 'clean_llm_output',
        software_engineer_instructions= 'Implement the clean_llm_output function. Use the docstring to guide you. ',
        previous_outputs= [],
        comments= [],
        previous_implementation= None,
        reviewer_comments= None,
        run_code= True
    )
    response = coder_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
