''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.constants import END, START

# Schema imports
from typing import Tuple

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, clean_llm_output, read_state_file




''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Software Engineer') if DEBUG else None



""" Schemas """
''' Input Schema '''
# The input schema for the software engineer, only the file path is required
class InputSchema(MessagesState):
    file_path: str # 'The path to the file.
    skip_tool_sections: bool # Whether to skip the tool sections.
    times_reviewed: int # The number of times the code has been reviewed.
    coder_run_code: bool # Wether the coder should run the code to review



''' LLM '''
model = myChatOpenAI(
    temperature= 0.4
)



''' Helpful Functions '''
# Returns the schema type of the state
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
def software_engineer_node(state: InputSchema) -> InputSchema:
    print_function_name() if DEBUG else None

    try:
        code = read_state_file(state)

        # Get the schema type and correct state access style
        schema_type, schema_call = get_schema_type(state)

        # Get the files available beside the generated agent
        files = '\n- '.join([
            file.name
            for file in Path(state['file_path']).parent.iterdir()
            if file.is_file()
        ])

        prompt = (
            "You are a Software Engineer implementing a custom AI agent.\n"
            "\n"
            "You are given an already designed and annotated Python code scaffold.\n"
            "Implement the complete file in ONE pass.\n"
            "\n"
            "The scaffold has already been prepared by previous stages of an AI agent\n"
            "builder. Its schemas, tools, helper functions, LLM definitions, node\n"
            "docstrings, routing functions, and LangGraph structure describe what the\n"
            "agent should do.\n"
            "\n"
            "Your task is to convert this scaffold into complete, runnable Python code.\n"
            "\n"
            "# PRIMARY TASK\n"
            "\n"
            "Implement every unfinished function in the supplied Python file.\n"
            "\n"
            "This may include:\n"
            "- workflow node functions;\n"
            "- helper functions;\n"
            "- tool functions;\n"
            "- custom tool-handler functions;\n"
            "- conditional routing functions;\n"
            "- schema methods;\n"
            "- other functions explicitly defined in the scaffold.\n"
            "\n"
            "Use each function's existing signature and docstring as its implementation\n"
            "specification.\n"
            "\n"
            "Return the COMPLETE implemented Python file.\n"
            "\n"
            "# SOURCE OF TRUTH\n"
            "\n"
            "Treat the provided scaffold as the source of truth.\n"
            "\n"
            "Preserve the existing:\n"
            "- workflow structure;\n"
            "- graph structure;\n"
            "- node names;\n"
            "- function names;\n"
            "- function signatures unless a minor correction is required;\n"
            "- schema names and fields;\n"
            "- tool names and signatures;\n"
            "- LLM variable names and configurations;\n"
            "- section structure;\n"
            "- detailed docstrings.\n"
            "\n"
            "Do not redesign the agent or introduce a different workflow.\n"
            "Do not add functionality that is not required by the scaffold.\n"
            "\n"
            "# IMPORTS\n"
            "\n"
            "Ensure all required imports are present in the Imports section.\n"
            "\n"
            "Rules:\n"
            "- Add imports required by your implementation.\n"
            "- Do not use inline imports inside functions.\n"
            "- Do not duplicate imports.\n"
            "- Keep imports at the top of the file.\n"
            "- Treat functionality imported from project modules as already implemented.\n"
            "- Do not reimplement imported project utilities.\n"
            "\n"
            "# LANGCHAIN AND LANGGRAPH\n"
            "\n"
            "The agent uses Python, LangChain, and LangGraph.\n"
            "\n"
            "A LangGraph invocation starts at START and follows graph edges until END.\n"
            "Every node must complete during the current invocation and return a valid\n"
            "state update.\n"
            "\n"
            "Nodes must not block while waiting for future user input. If the design\n"
            "requires another user message, the current run should produce its response,\n"
            "update the relevant state, and end. A later inbound message starts another\n"
            "graph invocation using persisted state.\n"
            "\n"
            "Messages normally use LangChain message types:\n"
            "- HumanMessage for user messages;\n"
            "- AIMessage for model responses;\n"
            "- ToolMessage for tool results.\n"
            "\n"
            f"The AgentSchema in this file is a {schema_type}.\n"
            f"Access its state fields using the appropriate style: {schema_call}.\n"
            "\n"
            "All state reads and writes must agree with AgentSchema.\n"
            "\n"
            "# NODE FUNCTIONS\n"
            "\n"
            "Implement each node according to its existing docstring.\n"
            "\n"
            "For each node:\n"
            "- read the required state fields;\n"
            "- perform the documented processing steps;\n"
            "- call its assigned LLM when required;\n"
            "- return the documented state updates;\n"
            "- use helper functions where specified;\n"
            "- respect the graph's expected control flow.\n"
            "\n"
            "Use safe_invoke for LLM invocation where appropriate.\n"
            "\n"
            "Do not unnecessarily provide the same context to an LLM twice. For example,\n"
            "if conversation history is already formatted into the system prompt, do not\n"
            "also pass the identical history again as separate messages unless required by\n"
            "the node design.\n"
            "\n"
            "# HELPFUL FUNCTIONS\n"
            "\n"
            "Helpful functions are normal deterministic Python functions.\n"
            "\n"
            "Implement all defined helper functions according to their docstrings.\n"
            "\n"
            "A helper function:\n"
            "- may be called directly from Python code;\n"
            "- should perform reusable deterministic processing;\n"
            "- may receive state values where its signature specifies them;\n"
            "- is not an LLM tool unless decorated with @tool.\n"
            "\n"
            "# TOOLS\n"
            "\n"
            "A tool is a real Python function that an LLM may request to execute.\n"
            "Tools are normally decorated with @tool and bound to an LLM using\n"
            ".bind_tools(...).\n"
            "\n"
            "Implement all tool functions according to their existing signatures and\n"
            "docstrings.\n"
            "\n"
            "Tool implementation rules:\n"
            "- Keep each tool focused on one responsibility.\n"
            "- Validate important arguments.\n"
            "- Handle expected errors gracefully.\n"
            "- Return compact useful results.\n"
            "- Preserve documented side effects.\n"
            "- Do not let a tool directly modify LangGraph state unless the existing\n"
            "  scaffold explicitly requires it.\n"
            "- A tool normally should not receive AgentSchema as an argument.\n"
            "\n"
            "The normal tool lifecycle is:\n"
            "1. An LLM produces a tool call.\n"
            "2. A ToolNode or custom tool-handler executes the tool.\n"
            "3. The tool result is converted into a ToolMessage.\n"
            "4. The graph continues according to its routing logic.\n"
            "\n"
            "# TOOL SECTIONS\n"
            "\n"
            "Carefully inspect each tool's docstring and the graph wiring to determine\n"
            "how the tool should be executed.\n"
            "\n"
            "There are two common tool-handling patterns.\n"
            "\n"
            "## TYPE A: STANDARD TOOLNODE\n"
            "\n"
            "A standard ToolNode is appropriate when:\n"
            "- the tool performs its action independently;\n"
            "- the returned result only needs to become a ToolMessage;\n"
            "- no additional non-message state fields must be updated;\n"
            "- the tool does not itself determine workflow routing;\n"
            "- the tool is not terminal.\n"
            "\n"
            "For Type A tools, use the ToolNode already defined in the scaffold.\n"
            "Do not create or invoke an unnecessary custom tool handler.\n"
            "\n"
            "A common flow is:\n"
            "LLM node -> ToolNode -> LLM node\n"
            "\n"
            "The LLM can then continue reasoning using the returned ToolMessage.\n"
            "\n"
            "## TYPE B: CUSTOM TOOL HANDLER\n"
            "\n"
            "A custom tool handler is appropriate when the tool call requires processing\n"
            "outside the tool itself.\n"
            "\n"
            "This includes cases where:\n"
            "- the tool call represents an intent or routing decision;\n"
            "- the tool is terminal;\n"
            "- the result must update non-message state fields;\n"
            "- the result must be transformed before storing it in state;\n"
            "- the tool docstring specifies work under an Outside-the-Tool Work or\n"
            "  Caller Responsibilities section.\n"
            "\n"
            "Implement existing custom tool handlers according to their docstrings.\n"
            "\n"
            "A custom handler typically needs to:\n"
            "1. Read the most recent AIMessage.\n"
            "2. Extract its tool calls.\n"
            "3. Match each tool call to the correct tool.\n"
            "4. Parse the supplied arguments.\n"
            "5. Invoke the tool.\n"
            "6. Create ToolMessage objects for tool results when required.\n"
            "7. Perform any state updates assigned to the handler rather than the tool.\n"
            "8. Return the correct LangGraph state update.\n"
            "\n"
            "Do not move handler responsibilities into the tool if the existing docstring\n"
            "explicitly separates those responsibilities.\n"
            "\n"
            "# TOOL ROUTING\n"
            "\n"
            "For every LLM using bind_tools, verify that the existing graph supports its\n"
            "tool calls correctly.\n"
            "\n"
            "Routing functions should distinguish between:\n"
            "- a normal LLM response;\n"
            "- a tool call intended for a standard ToolNode;\n"
            "- a tool call intended for a custom tool handler;\n"
            "- a terminal action, where applicable.\n"
            "\n"
            "After a non-terminal Type A tool executes, control normally returns to the\n"
            "LLM so it can process the tool result.\n"
            "\n"
            "After Type B tools, follow the existing intended routing described by the\n"
            "handler docstring and graph.\n"
            "\n"
            "Terminal tools should follow the existing terminal route and must not create\n"
            "an unnecessary continuation loop.\n"
            "\n"
            "Do not invent new graph paths unless a tiny change is absolutely necessary\n"
            "for the existing graph to function correctly.\n"
            "\n"
            "# TOOL CALL EXTRACTION\n"
            "\n"
            "When implementing custom tool handlers or routing functions, account for the\n"
            "tool-call representation used by LangChain messages.\n"
            "\n"
            "Use existing project utilities such as will_tool_call and\n"
            "parse_tool_arguments if they are imported and useful.\n"
            "\n"
            "Do not invoke bound tools directly from normal LLM workflow nodes. Tool\n"
            "execution belongs in ToolNode or custom tool-handler functions.\n"
            "\n"
            "# LLM DEFINITIONS\n"
            "\n"
            "Respect the LLM definitions already present in the scaffold.\n"
            "\n"
            "Understand these configurations correctly:\n"
            "- myChatOpenAI(...) creates the model used by a node;\n"
            "- .bind_tools([...]) allows that model to request those tools;\n"
            "- .with_structured_output(SomeSchema) constrains the model to return that\n"
            "  schema.\n"
            "\n"
            "Do not normally combine .bind_tools(...) and .with_structured_output(...)\n"
            "on the same LLM instance.\n"
            "\n"
            "If the scaffold already uses a schema as a bindable output together with\n"
            "tools, preserve that intended design rather than arbitrarily rewriting it.\n"
            "\n"
            "# STRUCTURED OUTPUT\n"
            "\n"
            "When an LLM uses .with_structured_output(SomeSchema), safe_invoke returns a\n"
            "parsed object matching that schema rather than a normal AIMessage string.\n"
            "\n"
            "Use the returned object accordingly.\n"
            "\n"
            "Do not access `.content` on a structured Pydantic object unless its schema\n"
            "actually defines such a field.\n"
            "\n"
            "# STATE AND MEMORY\n"
            "\n"
            "Treat AgentSchema as the contract for graph state.\n"
            "\n"
            "Follow these rules:\n"
            "- Read only defined state keys.\n"
            "- Return correctly typed values.\n"
            "- Respect Optional values and defaults.\n"
            "- Respect lists, dictionaries, and nested schemas.\n"
            "- If AgentSchema inherits MessagesState, messages already exists.\n"
            "- Respect add_messages annotations where present.\n"
            "- Preserve conversation history where required by the workflow.\n"
            "\n"
            "If the workflow uses fields such as next_action, mode, latest,\n"
            "pending_question, or identifiers, update them exactly as described in the\n"
            "node docstrings.\n"
            "\n"
            "# CONDITIONAL FUNCTIONS\n"
            "\n"
            "Implement every unfinished conditional-routing function.\n"
            "\n"
            "Each routing function must:\n"
            "- return only valid node names already expected by the graph;\n"
            "- match the corresponding add_conditional_edges mapping;\n"
            "- correctly inspect state and/or tool calls;\n"
            "- provide a deterministic fallback route;\n"
            "- avoid accidental infinite self-loops.\n"
            "\n"
            "# GRAPH STRUCTURE\n"
            "\n"
            "The graph has already been designed by previous stages.\n"
            "Do not redesign it.\n"
            "\n"
            "Check that your implementations are compatible with the existing graph:\n"
            "- every node referenced by the graph exists;\n"
            "- ToolNode instances receive existing tool functions;\n"
            "- custom handler nodes return the expected state updates;\n"
            "- conditional functions return keys present in their mapping;\n"
            "- START and END routes remain valid;\n"
            "- node output is compatible with the next node's expected input.\n"
            "\n"
            "# FILE OPERATIONS AND EXTERNAL SERVICES\n"
            "\n"
            "When implementing file or external-service operations:\n"
            "- use explicit encodings when reading and writing text files;\n"
            "- avoid unnecessary destructive file writes;\n"
            "- handle missing files or failed requests when relevant;\n"
            "- do not expose API keys or secrets in logs or responses;\n"
            "- use environment variables or existing configuration where provided.\n"
            "\n"
            "# SAFETY AND CORRECTNESS\n"
            "\n"
            "The final implementation should avoid:\n"
            "- unsafe eval or exec unless explicitly required;\n"
            "- command injection;\n"
            "- uncontrolled arbitrary file access;\n"
            "- accidental secret disclosure;\n"
            "- invalid tool argument handling;\n"
            "- unbounded internal loops;\n"
            "- obvious race or duplicate-side-effect problems where the scaffold accounts\n"
            "  for repeated events.\n"
            "\n"
            "# CODE QUALITY\n"
            "\n"
            "The completed file must be:\n"
            "- syntactically valid Python;\n"
            "- logically consistent;\n"
            "- compatible with LangChain and LangGraph;\n"
            "- importable assuming declared dependencies are installed;\n"
            "- consistent with the existing schemas and docstrings;\n"
            "- free from undefined names caused by your implementation;\n"
            "- free from duplicate imports;\n"
            "- free from inline imports;\n"
            "- free from required functions that remain unimplemented.\n"
            "\n"
            "Do not leave required function bodies containing only:\n"
            "- ...\n"
            "- pass\n"
            "- TODO comments\n"
            "- placeholder return values\n"
            "\n"
            "# EXISTING PROJECT UTILITIES\n"
            "\n"
            "If imported, you may assume the following project utilities are already\n"
            "implemented and safe to use:\n"
            "- myChatOpenAI\n"
            "- safe_invoke\n"
            "- print_function_name\n"
            "- will_tool_call\n"
            "- parse_tool_arguments\n"
            "- clean_llm_output\n"
            "\n"
            "Do not reimplement them.\n"
            "\n"
            "# NEIGHBOURING FILES\n"
            "\n"
            "The following files exist in the same generated-agent directory:\n"
            "<FILES>\n"
            f"- {files}\n"
            "</FILES>\n"
            "\n"
            "Only rely on those files when the current scaffold imports or clearly refers\n"
            "to them. Do not invent APIs based only on their filenames.\n"
            "\n"
            "# OUTPUT\n"
            "\n"
            "Return ONLY the complete implemented Python source file.\n"
            "\n"
            "Do not return:\n"
            "- markdown code fences;\n"
            "- explanations;\n"
            "- analysis;\n"
            "- patches;\n"
            "- diffs;\n"
            "- commentary before or after the code.\n"
            "\n"
            "The first characters of your response should be valid Python source code.\n"
            "\n"
            "# CURRENT ANNOTATED CODE SCAFFOLD\n"
            "\n"
            "<CODE_START>\n"
            f"{code}\n"
            "</CODE_END>\n"
        )

        response = safe_invoke(model, messages= [SystemMessage(content=prompt)]
        )

        implemented_code = clean_llm_output(response.content).strip()

        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {implemented_code}') if DEBUG else None

        if not implemented_code:
            raise ValueError('Software Engineer returned empty code.')

        with open(state['file_path'], 'w', encoding= 'utf-8') as f:
            f.write(implemented_code)

        return {'messages': [response]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state



''' Graph '''
software_engineer_graph = StateGraph(InputSchema)

software_engineer_graph.add_node('software_engineer_node', software_engineer_node)

software_engineer_graph.add_edge(START, 'software_engineer_node')
software_engineer_graph.add_edge('software_engineer_node', END)

software_engineer_app = software_engineer_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(software_engineer_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/software_engineer_app.png', 'wb') as f:
        f.write(software_engineer_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'softwareEngineer'
    os.environ['LANGSMITH_PROJECT'] = 'softwareEngineer'
    client = Client()

    config = {
        'recursion_limit': 150,
        'configurable': {
            'user_id': 'softwareEngineer',
            'run_name': 'softwareEngineer',
            # 'thread_id': 'softwareEngineer', 
        }
    }

    user = InputSchema(file_path= '..\..\creations\menu_recommendation_workflow\menu_recommendation_workflow.py', times_reviewed= 0, skip_tool_sections= False, coder_run_code= True)
    response = software_engineer_app.invoke(user, config= config)

    # print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    # if DEBUG:
    #     for key, value in response.items():
    #         print(f'    {key}: {value}\n')
