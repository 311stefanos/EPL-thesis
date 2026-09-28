''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, BaseMessage

# Langgraph imports
from langgraph.constants import END, START
from langgraph.graph import StateGraph

# Schema imports
from typing import Tuple, TypedDict, Literal, List, Optional, Dict
from pydantic import BaseModel, Field, model_validator

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, USER_APPROVALS, read_state_file


''' Constants '''
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'

print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} {RED}Ablated{RESET} Prompt Engineer Ablation') if DEBUG else None


""" Schemas """
''' General Schemas '''
class Format(BaseModel):
    format_dict: Dict = Field(description='The dictionary used to format the prompt.')
    non_format_messages_list: List[BaseMessage] = Field(description='The list of non-format messages.', default=[])


class Prompt(BaseModel):
    prompt_name: str = Field(description='The name of the prompt.')
    suggested_prompt: str = Field(description='The suggested prompt.', default='')
    necessary_code_changes: List[Tuple[str, str]] = Field(description='The necessary code changes.', default=[])
    format: Format = Field(description='The format of the prompt.', default=Format(format_dict={}))
    user_comments: List[str] = Field(description='The user comments.', default=[])
    latest_response: str = Field(description='The latest response.', default='')
    prompt_reviews: int = Field(description='The number of LLM reviews for the prompt.', default=0)
    response_reviews: int = Field(description='The number of LLM reviews for the response.', default=0)

    def reviewed(self, what: Literal['prompt', 'response']) -> None:
        if what == 'prompt':
            self.prompt_reviews += 1
        elif what == 'response':
            self.response_reviews += 1

    def reset_reviews(self, what: Literal['prompt', 'response']) -> None:
        if what == 'prompt':
            self.prompt_reviews = 0
        elif what == 'response':
            self.response_reviews = 0

    def can_review(self, what: Literal['prompt', 'response']) -> bool:
        if what == 'prompt':
            return self.prompt_reviews < 2
        return self.response_reviews < 2

    def set_format(self, format: Format) -> None:
        self.format = format

    def add_comments(self, comments: str) -> None:
        self.user_comments.append(comments)

    def add_response(self, response: str) -> None:
        self.latest_response = response

    def is_approved(self) -> bool:
        last_comment = self.user_comments[-1] if self.user_comments else ''
        last_comment = last_comment.replace('Review by Expert Reviewer: ', '')
        last_comment = last_comment.replace('Review by User: ', '')
        last_comment = last_comment.replace('# Issues', '')
        last_comment = last_comment.replace('- ', '')
        last_comment = last_comment.replace('`', '')
        return not any(c.strip() not in USER_APPROVALS for c in last_comment.split('\n'))

    def filter_comments(self) -> List[str]:
        comments: List[str] = []
        for comment in self.user_comments:
            clear_comment: str = comment.replace('Review by Expert Reviewer: ', '')
            clear_comment = clear_comment.replace('Review by User: ', '')
            if any(c.strip() not in USER_APPROVALS for c in clear_comment.split('\n')):
                comments.append(comment)
        return comments


class PromptList(BaseModel):
    prompts: List[Prompt] = Field(description='All prompts required by the generated agent.')

    @model_validator(mode='after')
    def unique_prompt_names(self):
        names = [prompt.prompt_name for prompt in self.prompts]
        if len(names) != len(set(names)):
            raise ValueError('Prompt names must be unique.')
        return self


''' Input Schema '''
class InputSchema(TypedDict):
    file_path: str
    prompt_list: Optional[List[Prompt]]
    active_prompt_index: Optional[int]
    error: Optional[bool]
    mode: Literal['llm', 'user', 'both']


''' LLM '''
model = myChatOpenAI(
    temperature= 0.4    
).with_structured_output(PromptList)#, method='function_calling')


''' Helpful Functions '''
def get_prompt_names(file_path: str) -> List[str]:
    '''
    Reads the generated Python file and returns unique prompt names referenced as prompts.PROMPT_NAME.
    '''
    pattern = re.compile(r'^\s*\w+:?\s*\w+?\s*=\s*prompts\.([A-Z][A-Z0-9_]*)\b', re.MULTILINE)
    with open(file_path, 'r', encoding= 'utf-8') as f:
        code = f.read()
    names_in_order = [match.group(1) for match in pattern.finditer(code)]
    seen = set()
    unique_prompt_names: List[str] = []
    for name in names_in_order:
        if name not in seen:
            seen.add(name)
            unique_prompt_names.append(name)
    return unique_prompt_names


''' Nodes '''
def generate_prompts(state: InputSchema) -> InputSchema:
    '''
    Extracts every required prompt name and generates all prompts in one LLM call.
    '''
    print_function_name() if DEBUG else None

    try:
        code = read_state_file(state)
        prompt_names: List[str] = get_prompt_names(state['file_path'])
        prompt_names_text: str = '\n'.join([f'- {name}' for name in prompt_names])

        prompt = (
            "You are the Prompt Engineer for a custom AI agent builder.\n"
            "You are given the complete implemented Python file for a generated custom AI agent and the exact prompt names referenced by that file.\n"
            "Create ALL required prompts in ONE response.\n"
            "Return exactly one Prompt object for every required prompt name and do not create prompts with names that are not listed.\n"
            "Use the complete Python code as the source of truth for each prompt's role, inputs, LLM configuration, tools, schemas, node behaviour, state usage, and expected output.\n"
            "Each generated prompt must be sufficiently detailed for the LLM using it to perform its assigned task correctly without seeing the Python code at runtime.\n"
            "Keep the prompts clear, organized, implementation-specific, and no more complicated than required by the code.\n"
            "For every prompt, identify the node or function where it is formatted and infer all values passed through `.format(...)` as prompt inputs.\n"
            "Every value that must come from `.format(...)` must appear in the generated prompt as a Python format placeholder using single braces such as `{code}`, `{messages}`, or `{user_input}`.\n"
            "Do not invent format placeholders that the current code cannot supply unless you also provide the smallest necessary code change in `necessary_code_changes`.\n"
            "Literal braces that must survive Python `.format(...)` must be escaped as double braces `{{` and `}}` inside the generated prompt string.\n"
            "If the prompt contains JSON, dictionaries, schema examples, or other literal brace syntax, escape those literal braces so Python `.format(...)` will not treat them as placeholders.\n"
            "If the LLM associated with a prompt uses `.with_structured_output(SomeSchema)`, include an `# Output Format` section that clearly describes the schema fields, types, nesting, and optional values required by that structured output.\n"
            "If the LLM associated with a prompt uses `.bind_tools(...)`, include an `# Available Tools` section that lists every bound tool relevant to that LLM and explains each tool's purpose, arguments, return value, and when it should be used.\n"
            "Do not invent tools or schemas that are not defined or imported in the code.\n"
            "When a bound item is a schema used as an output mechanism together with tools, explain that behaviour consistently with the code rather than treating it as an ordinary external-action tool.\n"
            "A tool is a real callable function used for external information, execution, storage, APIs, file operations, or other side effects that the LLM cannot perform by reasoning alone.\n"
            "When describing tools, distinguish the LLM's responsibility from any custom tool-handler responsibility already implemented in the graph.\n"
            "Do not instruct a normal LLM node to directly invoke a LangChain tool with `.invoke()` when the graph uses a ToolNode or custom tool-handler node for execution.\n"
            "If a prompt belongs to an LLM node with tool access, explain when the model should request a tool call and what it should do after tool results become available if that continuation is part of the existing graph.\n"
            "Respect the existing LangGraph architecture and do not use the prompt to redesign nodes, graph edges, tool handlers, or state schemas.\n"
            "If a prompt operates on graph state or conversation history, clearly describe the supplied state-derived inputs without inventing state fields that do not exist in AgentSchema.\n"
            "If a prompt receives message history that is already formatted into the prompt, do not instruct the runtime to provide the same history again separately unless the existing Python code explicitly does so.\n"
            "If the Python code passes conversation messages separately to the LLM instead of formatting them into the prompt, the prompt should not invent a `{messages}` placeholder merely to restate them.\n"
            "Respect whether the associated LLM returns natural language, an AIMessage with tool calls, or a Pydantic structured object and describe the expected response accordingly.\n"
            "Use explicit instructions for constraints that materially affect correctness, especially required output fields, tool-selection behaviour, state compatibility, routing decisions, and content that downstream code depends on.\n"
            "Avoid generic prompt-engineering filler, unnecessary motivational language, redundant rules, and requirements that the surrounding code does not depend on.\n"
            "Do not add examples unless they materially clarify a complex required output or tool argument format.\n"
            "Do not add prompt-injection policies, generic safety sections, or unrelated best practices unless the generated agent's code or task explicitly requires them.\n"
            "Use `necessary_code_changes` only when a prompt genuinely requires a formatting input that the current code does not pass or when a tiny formatting-input preparation change is necessary.\n"
            "Every code change must be represented as an exact `(old_code, new_code)` replacement and must be limited to `.format(...)` arguments or the smallest nearby code needed to prepare those arguments.\n"
            "Do not use `necessary_code_changes` to refactor node logic, graph wiring, schemas, tools, imports, or unrelated code.\n"
            "If no code change is required for a prompt, return an empty `necessary_code_changes` list.\n"
            "Set the remaining Prompt fields to their default values unless their schema requires otherwise because this ablation does not perform prompt review, response testing, user review, or iterative regeneration.\n"
            "Before returning, verify that every required prompt name appears exactly once and that no extra prompt names are present.\n"
            "Before returning, verify that all placeholders used by each prompt either already have matching `.format(...)` arguments in the Python code or are supported by an explicit minimal code change.\n"
            "Before returning, verify that tool-enabled prompts document their actual bound tools and structured-output prompts document their actual output schemas.\n"
            "\n"
            "# Required Prompt Names\n"
            "<PROMPT_NAMES_START>\n"
            f"{prompt_names_text}\n"
            "</PROMPT_NAMES_END>\n"
            "\n"
            "# Complete Python Code\n"
            "<CODE_START>\n"
            f"{code}\n"
            "</CODE_END>\n"
        )

        response: PromptList = safe_invoke(model, messages=[SystemMessage(content=prompt)])
        returned_names: List[str] = [prompt.prompt_name for prompt in response.prompts]

        if set(returned_names) != set(prompt_names) or len(returned_names) != len(prompt_names):
            raise ValueError(f'Generated prompt names do not match required names. Required: {prompt_names}. Returned: {returned_names}.')

        prompts_by_name: Dict[str, Prompt] = {prompt.prompt_name: prompt for prompt in response.prompts}
        ordered_prompts: List[Prompt] = [prompts_by_name[name] for name in prompt_names]

        print(f'{BLUE}[NODE] [INFO] [PROMPTS]{RESET} {[prompt.prompt_name for prompt in ordered_prompts]}') if DEBUG else None

        return {
            'prompt_list': ordered_prompts, 
            'active_prompt_index': len(ordered_prompts) - 1 if ordered_prompts else 0, 
            'error': False
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return {'error': True}


def paste_prompts(state: InputSchema) -> InputSchema:
    '''
    Writes all generated prompts to the prompt file and applies minimal formatting-related code changes.
    '''
    print_function_name() if DEBUG else None

    try:
        prompt_list: List[Prompt] = state.get('prompt_list') or []
        prompts_str: str = '\n\n\n'.join([prompt.prompt_name + ' = """\n' + prompt.suggested_prompt + '\n"""' for prompt in prompt_list])
        prompt_file_path: str = state['file_path'].replace('.py', '_prompts.py')

        with open(prompt_file_path, 'w', encoding= 'utf-8') as f:
            f.write(prompts_str)

        code: str = read_state_file(state)

        for generated_prompt in prompt_list:
            for old_code, new_code in generated_prompt.necessary_code_changes:
                if old_code not in code:
                    raise ValueError(f'Code change for {generated_prompt.prompt_name} could not be applied because the old code was not found: {old_code}')
                code = code.replace(old_code, new_code, 1)

        with open(state['file_path'], 'w', encoding= 'utf-8') as f:
            f.write(code)

        return {'error': False}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return {'error': True}


''' Graph '''
prompt_engineer_graph = StateGraph(InputSchema)

prompt_engineer_graph.add_node('generate_prompts', generate_prompts)
prompt_engineer_graph.add_node('paste_prompts', paste_prompts)

prompt_engineer_graph.add_edge(START, 'generate_prompts')
prompt_engineer_graph.add_edge('generate_prompts', 'paste_prompts')
prompt_engineer_graph.add_edge('paste_prompts', END)

prompt_engineer_app = prompt_engineer_graph.compile()


''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage
    GraphImage(prompt_engineer_app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/prompt_engineer_ablation_app.png', 'wb') as f:
        f.write(prompt_engineer_app.get_graph().draw_mermaid_png())

    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'promptEngineerAblation'
    os.environ['LANGSMITH_PROJECT'] = 'promptEngineerAblation'
    client = Client()

    config = {'recursion_limit': 100, 'configurable': {'user_id': 'promptEngineerAblation', 'run_name': 'promptEngineerAblation'}}

    user = {'file_path': '../../creations/menu_recommendation_workflow/menu_recommendation_workflow.py', 'mode': 'both'}
    response = prompt_engineer_app.invoke(user, config=config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response: {response}') if DEBUG else None
