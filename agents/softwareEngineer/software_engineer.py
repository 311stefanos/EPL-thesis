"""
- `author:` Stefanos Panteli
- `date:` 2025-11-03
- `description:` The software engineer is an agent that is used to implement a python file. It is given an already created 
                 code structure with a basic outline, and is asked to implement it with multiple tools. Its main function
                 is to orchestrate coder subagents, and to build the code step by step.

## How to use
1. Import the app. (`from agents.softwareEngineer.softwareEngineer import software_engineer_app`)
2. Input a dict with the following keys:
    - `file_path: str`: The path of the file to implement. This file should be the outlined code structure.
3. Invoke the app.
4. Get the output dict with the following keys:
    - `file_path: str`: The path of the file that was implemented.
The output of this agent does not matter, its purpose is to implement the code.

## Usage
```python
from agents.softwareEngineer.software_engineer import software_engineer_app
graph_input = { 'file_path': path/to/file.py" }

response = software_engineer_app.invoke(graph_input)

# response = { 'file_path': path/to/file.py" }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode

# Schema imports
from typing import Literal, List, Optional, Dict
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import json
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments
from agents.softwareEngineer import prompts
from agents.coder.coder import (
    InputSchema as CoderInputSchema,
    FunctionProposal,
    OutputSchema as CoderOutputSchema,
    coder_app
)



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
''' General Schemas '''
# A schema used to store the coders' outputs. Also keeps metadata such as whether the coder has approved or disapproved the code.
class CoderSchema(CoderOutputSchema):
    approved: bool = Field(description= 'Whether the coder has approved the code.', default= False)
    disapproved: bool = Field(description= 'Whether the coder has disapproved the code.', default= False)

    def approve(self) -> None:
        self.approved = True
        self.disapproved = False

    def disapprove(self) -> None:
        self.disapproved = True
        self.approved = False

# A schema used to store the comments from the software engineer for the disapproved coder's code..
class CoderComment(BaseModel):
    comment: Optional[str] = Field(description= 'The comments from the software engineer.', default= None)

# A schema used by the QA to notify the software engineer for code issues, after the code submition.
class CodeIssues(BaseModel):
    class Issue(BaseModel):
        issue: str = Field(description= 'The code issue.')
        comment: Optional[str] = Field(description= 'The comment for the software engineer to read. Can be ommited', default= None)

    general_comments: Optional[str] = Field(description= 'General comments for the whole code base.', default= None)
    issues: List[Issue] = Field(description= 'A list of the code issues.', default= [])

''' Input Schema '''
# The input schema for the software engineer, only the file path is required
class InputSchema(MessagesState):
    file_path: str = Field(description= 'The path to the file.')



''' Global Variables '''
# A dict used to store the coders' outputs in the format {function_name: CoderSchema}
coders: Dict[str, CoderSchema] = {}
# A dict used to store the comments from the software engineer for the disapproved coder's code in the format {function_name: CoderComment}
comments: Dict[str, CoderComment] = {}
# A pydantic schema used to store the identified problems in the code
code_issues: CodeIssues = CodeIssues()



''' Tools '''
# A tool used to write code to a file as a whole.
@tool
def write_code_to_file(file_path: str, code: str) -> str:
    '''
    `write_code_to_file` writes the contents of `code` to the file `file_path`
    This tool should be called a coder returned a code you approved.
    
    `Args:`
        file_path (str): The path to the file to overwrite.
        code (str): The code to write to the file.

    `Returns:`
        (str) If the file overwrite was successful
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    try:
        # Remove possible tags from the LLM's output
        while code[0] == '<':
            index = code.find('>\n')
            code = code[index + 1:].strip()

        while code.strip()[-1] == '>':
            index = code.rfind('<')
            code = code[:index].strip()

        # Write the code to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        print(f'{BLUE}[TOOL] [INFO] [OVERWRITE]{RESET} The file {file_path} was overwritten successfully.') if DEBUG else None
        return f'[GOOD] The file {file_path} was overwritten successfully.'

    except Exception as e:
        print(f'{RED}[TOOL] [ERROR] [OVERWRITE]{RESET} The file {file_path} could not be overwritten due to the error: {e}') if DEBUG else None
        return f'[ERROR] The file {file_path} could not be overwritten due to the error: {e}'

# A tool used to call a coder agent.
@tool
def call_coder(function_name: str, special_instructions: str, file_path: str) -> Dict[str, CoderSchema]:
    '''
    `call_coder` calls a coder to implement the given function.

    `Args:`
        function_name (str): The name of the function to implement.
        special_instructions (str): The special instructions from the software engineer to the coder.
        file_path (str): The path to the file to implement the function in.

    `Returns:`
        (Dict[str, CoderSchema]): {function_name: CoderSchema}
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Check if the function definition is in the file, otherwise the coder cannot complete the task -> return
    if f'def {function_name}(' not in code:
        print(f'{RED}[TOOL] [ERROR] [APPROVE]{RESET} The definition of the function could not be found in the file hence a coder cannot complete the task. {function_name}') if DEBUG else None
        return {
            'code': 'The definition of the function could not be found in the file hence a coder cannot complete the task. You should define the function first.', 
            'proposals': None
        }

    # If the coder does not exist in the coders dict, add it
    if function_name not in coders:
        coders[function_name] = CoderSchema(code= '', proposals= None, approved= False, disapproved= False)
        comments[function_name] = CoderComment()

    # Call the coder
    args: CoderInputSchema = {
        'messages': [],
        'file_path': file_path,
        'function_name': function_name,
        'software_engineer_instructions': special_instructions,
        'previous_outputs': [coders[function_name].code] if coders[function_name].code else [],
        'comments': [comments[function_name].comment] if comments[function_name].comment else [],
    }
    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'softwareEngineer',
            'run_name': 'softwareEngineer',
            'thread_id': function_name, 
        }
    }

    try:
        response: CoderOutputSchema = coder_app.invoke(args, config)
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
    
    print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None
    # Save the coder's output
    coders[function_name] = CoderSchema(code= response['code'], proposals= response['proposals'], approved= False, disapproved= False)

    # Return the coder's output for the ToolMessage
    return {function_name: coders[function_name]}

# A tool used to comment on the coder's output code and disapproving it.
@tool
def disapprove_and_comment_on_coder_code(function_name: str, comment: str) -> str:
    '''
    `disapprove_and_comment_on_coder_code` use it to comment on the coder's output code. Only use it when you think the coder's output is incorrect and you did not approve it.
    Should be used after `call_coder` if the coder's output is incorrect and you did not approve it.

    `Args:`
        function_name (str): The name of the function to comment on.
        comment (str): The comment to add to the coder's output.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    # Check if the coder exists
    all_keys = set(list(coders.keys()) + list(comments.keys()))
    if function_name not in all_keys:
        print(f'{RED}[TOOL] [ERROR] [COMMENT]{RESET} The coder for function {function_name} does not exist.') if DEBUG else None
        return f'[ERROR] The coder for function {function_name} does not exist.'

    # Disapprove the coder, and add a comment
    coders[function_name].disapprove()
    comments[function_name].comment = comment

    print(f'{BLUE}[TOOL] [INFO] [COMMENT]{RESET} Commented on the coder\'s output for function {function_name}: {comment}') if DEBUG else None
    return f'[SUCCESS] Commented on the coder\'s output for function {function_name}: {comment}\n\nNow ready to be called again and understand the incorrect code.'

# A tool used to approve the coder's output code
@tool
def approve_function_code(file_path: str, function_name: str) -> str:
    '''
    `approve_coder_output` approves the coder's output code. Only use it when you think the coder's output is correct.
    Should be used after `call_coder`, if the coder's output is correct and approved.

    `Args:`
        file_path (str): The name of the file to overwrite.
        function_name (str): The name of the function to approve.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    # Check if the coder exists
    all_keys = set(list(coders.keys()) + list(comments.keys()))
    if function_name not in all_keys:
        print(f'{RED}[TOOL] [ERROR] [APPROVE]{RESET} The coder for function {function_name} does not exist.') if DEBUG else None
        return f'[ERROR] The coder for function {function_name} does not exist.'
    
    # Check if the coder has an implementation
    previous_code: str = coders[function_name].code
    if not previous_code:
        print(f'{RED}[TOOL] [ERROR] [APPROVE]{RESET} The coder for function {function_name} does not have a coder implementation.') if DEBUG else None
        return f'[ERROR] The coder for function {function_name} does not have a previous implementation.'
    
    # Replace the code in the file
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Get the code sections from the function onwards
    is_tool = ''
    if '@tool' in previous_code:
        is_tool = '@tool\n'
    code_section: str = f'def {function_name}(' + f'def {function_name}('.join(code.split(f'{is_tool}def {function_name}(')[1:])
    code_section = code_section.split("''' Conditional Functions '''")[0].split("''' Graph '''")[0].strip()
    code_section = code_section.split('""" Conditional Functions """')[0].split('""" Graph """')[0].strip()
    for line in code_section.split('\n')[1:]:
        if (
            (line.startswith('def ') and line.endswith(':')) or
            '= myChatOpenAI(' in line or
            line in [
                '# TODO: Add Helpful Functions (if needed)',
                '# TODO: Add Tools (if needed)',
                "''' Nodes '''",
                '@tool'
            ]
        ):
            code_section = code_section.split(line)[0].strip()
            break

    # Check if the function has a code section
    if not code_section:
        print(f'{RED}[TOOL] [ERROR] [APPROVE]{RESET} The coder for function {function_name} does not have a code section in the file.') if DEBUG else None
        return f'[ERROR] The coder for function {function_name} does not have a code section in the file.'
    
    # Replace the code
    code = code.replace(code_section, previous_code)
    if '@tool\n@tool' in code:
        code = code.replace('@tool\n@tool', '@tool')

    print(f'{BLUE}[TOOL] [OLD SECTION]{RESET} {code_section}') if DEBUG else None
    print(f'{BLUE}[TOOL] [NEW SECTION]{RESET} {previous_code}') if DEBUG else None

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    if coders[function_name].imports:
        add_imports.invoke(coders[function_name].imports, file_path)

    # Approve the coder's implementation
    coders[function_name].approve()

    print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Approved the coder\'s output for function {function_name}') if DEBUG else None
    return f'[SUCCESS] Approved the coder\'s output for function {function_name}.'

# A tool used to approve the coder's function proposals
@tool
def approve_function_proposals(approved_function_proposals: List[FunctionProposal], file_path: str) -> str:
    # TODO: check proposal_coder_function_name: str, approved_function_proposal_names: List[str]
    '''
    `approve_function_proposals` approves a subset of a coder's function proposals. Only use it when you think the coder's function proposals are correct.
    Should be used after `call_coder`, if the coder's function proposals are correct and approved.
    - Note: You may approve multiple function proposals at once.
    - You may approve a function proposal even if the coder's code output is not approved.

    `Args:`
        function_proposals (List[FunctionProposal]): The function proposals to approve. As given from the coder.
        file_path (str): The path to the file to implement the function in.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    # Split the proposals into tools and helper functions
    proposed_tools = [str(function_proposal) for function_proposal in approved_function_proposals if function_proposal.function_type == 'tool']
    proposed_functions = [str(function_proposal) for function_proposal in approved_function_proposals if function_proposal.function_type == 'helper_function']

    # Add the tools and helper functions to the file - as definitions
    code = read_state_file({'file_path': file_path})
    code = code.replace('# TODO: Add Tools (if needed)', '\n\n'.join(proposed_tools) + '\n\n# TODO: Add Tools (if needed)')
    code = code.replace('# TODO: Add Helpful Functions (if needed)', '\n\n'.join(proposed_functions) + '\n\n# TODO: Add Helpful Functions (if needed)')
    write_code_to_file.invoke({"file_path": file_path, "code": code}) 

    print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Approved the coder\'s function proposals: {[afp.function_name for afp in approved_function_proposals]}') if DEBUG else None
    return f'[SUCCESS] Approved the coder\'s function proposals: {[afp.function_name for afp in approved_function_proposals]}.\n\nThe file contents have been updated.'

# A tool to add imports fast
@tool
def add_imports(imports: List[str], file_path: str) -> str:
    '''
    `add_imports` adds imports to the file. you should use this tool as little as possible.

    `Args:`
        imports (List[str]): The imports to add to the file. Should input the full import line.
        file_path (str): The path to the file to implement the function in.
    
    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None
    
    try:
        code = read_state_file({'file_path': file_path})
        
        old_code_section = code.split("''' Imports '''")[-1].split("''' Constants '''")[0].strip()
        new_code_section = old_code_section + '\n\n' + '\n'.join(imports)

        code = code.replace(old_code_section, new_code_section)

        print(f'{BLUE}[TOOL] [OLD SECTION]{RESET} {old_code_section}') if DEBUG else None
        print(f'{BLUE}[TOOL] [NEW SECTION]{RESET} {new_code_section}') if DEBUG else None

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Added the imports to the file.') if DEBUG else None
        return f'[SUCCESS] Added the imports to the file.'

    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
        return f'[ERROR] {e}'

# Creates an external file
@tool
def create_file(file_path: str, contents: str) -> str:
    '''
    `create_file` creates a file with the given contents.

    `Args:`
        file_path (str): The relative path to the file to create.
        contents (str): The contents of the file to create.

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None
    try:
        # Should be under the Path(__file__).resolve().parent.parent directory
        base_dir = Path(__file__).resolve().parent.parent
        target = Path(file_path).resolve()

        # Ensure target is a child of base_dir
        try:
            is_child = target.is_relative_to(base_dir)
        except AttributeError:
            # Python < 3.9 fallback
            is_child = os.path.commonpath([str(target), str(base_dir)]) == str(base_dir)

        if not is_child:
            print(f'{RED}[TOOL] [ERR]{RESET} The file {file_path} must be inside {base_dir}.') if DEBUG else None
            return f'[ERROR] The file {file_path} must be a child of {base_dir}.'

        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            print(f'{RED}[TOOL] [ERR]{RESET} The file {target} already exists.') if DEBUG else None
            return f'[ERROR] The file {target} already exists.'

        with open(target, 'w', encoding='utf-8') as f:
            f.write(contents)

        print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Created the file {target} successfully.') if DEBUG else None
        return f'[SUCCESS] Created the file {target} successfully.'

    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()
        return f'[ERROR] {e}'

# A tool used to submit the final code
@tool
def submit_final_code(file_path: str) -> None:
    '''
    `submit_final_code` submits the final implementation of the file. You must implement all the functions before calling this.

    `Args:`
        file_path (str): The path to the file to implement the function in.
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None
    # Just returns a success message
    print(f'{BLUE}[NODE] [INFO] [SUBMIT]{RESET} {file_path} implemented successfully.') if DEBUG else None

# A tool used to resolve code issues
@tool
def code_issue_resolved(resolved_issues: List[str]) -> str:
    '''
    `code_issue_resolved` resolves a code issue that was proposed by the Quality Assurance team.

    `Args:`
        resolved_issues (List[str]): The issues that have been resolved. Must must exact wording as given from the Quality Assurance team (can be found in the prompt).

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    # Split the 'resolved' code issues into resolved and not resolved
    not_resolved_issues = []
    resolved_issues = []

    for resolved_issue in resolved_issues:
        # Check if the code issue exists
        if resolved_issue not in code_issues:
            print(f'{RED}[TOOL] [ERROR] [RESOLVE]{RESET} The code issue {resolved_issue} does not exist.') if DEBUG else None
            not_resolved_issues.append(resolved_issue)
        
        elif resolved_issue in code_issues:
            code_issues.issues.remove(resolved_issue)
            resolved_issues.append(resolved_issue)

    print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Resolved the coder\'s code issues: {resolved_issues}') if DEBUG else None
    print(f'{RED}[TOOL] [ERROR] [NOT RESOLVED]{RESET} The following code issues were not resolved: {not_resolved_issues}') if DEBUG else None
    return f'[SUCCESS] Resolved the coder\'s code issues: {resolved_issues}\n[ERROR] The following code issues were not resolved (due to not exact wording): {not_resolved_issues}'

tools = [
    write_code_to_file, 
    call_coder, 
    disapprove_and_comment_on_coder_code, 
    approve_function_code, 
    approve_function_proposals, 
    add_imports, 
    create_file, 
    submit_final_code, 
    code_issue_resolved
]

# Dictionary of tools: tool name -> tool
tools_by_name = {tool.name: tool for tool in tools}

''' LLM '''
# The agent that adds the tool sections
tool_adder = myChatOpenAI(
    temperature= 0.1,
    model= 'mistralai/devstral-2512:free'
)

# The Software Engineer that orchestrates the tools
software_engineer = myChatOpenAI(
    temperature= 0.4,
    model= 'mistralai/devstral-2512:free'
).bind_tools(tools, parallel_tool_calls= False)

# The Quality Assurance team that validates the code and proposes code issues
code_validator = myChatOpenAI(
    temperature= 0.6,
    model= 'mistralai/devstral-2512:free'
).with_structured_output(CodeIssues)



''' Helpful Functions '''
# Reads the contents of state['file_path']
def read_state_file(state: InputSchema) -> str:
    '''
    `read_state_file` reads the contents of state['file_path']
    
    `Args:`
        state (InputSchema): The state of the agent. Must have the key 'file_path'.

    `Returns:`
        code: str
    '''
    with open(state['file_path'], 'r', encoding='utf-8') as f:
        code = f.read()
    return code



''' Nodes '''
def add_tool_sections(state: InputSchema) -> InputSchema:
    '''
    Identifies which LLMs have access to tools and modifies the graph accordingly.
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        code = read_state_file(state)

        prompt = prompts.TOOL_SECTION_ADDER_PROMPT.format(code= code)

        response = safe_invoke(tool_adder, prompt).content

        with open(state['file_path'], 'w', encoding='utf-8') as f:
            f.write(response)

        return state
    except Exception as e:
        traceback.print_exc()
        return state

# The Software Engineer node where the Software Engineer is prompted to call the tools
def software_engineer_node(state: InputSchema) -> InputSchema:
    '''
    This node calls the Software Engineer to implement the code with the help of various tools.
    Available tools:
    - write_code_to_file
    - call_coder
    - disapprove_and_comment_on_coder_code
    - approve_function_code
    - approve_function_proposals
    - add_imports
    - create_file
    - submit_final_code
    - code_issue_resolved
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        # A simple line to guide the Software Engineer through the process
        last_message = state['messages'][-1] if state['messages'] else None
        last_prompt = ''
        if hasattr(last_message, 'name'):
            if last_message.name == 'call_coder':
                last_prompt = '\n# Next Step:\nApprove requests, Approve code, Disapprove code: using the respective tools.\n\n'

        # Get the code issues from the Quality Assurance
        issues = ''
        for issue_schema in code_issues.issues:
            issue = issue_schema.issue
            comment = issue_schema.comment
            if comment:
                issues += f'{issue}\n    Comment: {comment} (Do not include comment in the `code_issue_resolved` tool call)\n'
            else:
                issues += f'{issue}\n'

        code_issues_prompt = prompts.CODE_ISSUES_SECTION.format(
            code_issues= issues
        ) if issues else ''

        # Files under file_path/..
        files = '\n- '.join([Path(state['file_path']).parent])
        
        # Get the implemented functions from the Coders that have not been approved or disapproved yet
        functions = '\n\n---\n\n'.join([
            coder_schema.code + (f'\n\nProposals:\n{coder_schema.proposals}' if coder_schema.proposals else '')
            for _, coder_schema in coders.items()
            if not coder_schema.approved and not coder_schema.disapproved
        ])

        # Get the implemented functions from the Coders that have been disapproved
        disapproved_functions = '\n\n---\n\n'.join([
            coder_schema.code
            for _, coder_schema in coders.items()
            if coder_schema.disapproved
        ])
        
        prompt = prompts.SOFTWARE_ENGINEER_PROMPT.format(
            file_path= state['file_path'],
            code= read_state_file(state),
            tool_messages= '\n\n'.join([message.pretty_repr() for message in state['messages'][-3:]]),
            files= files,
            functions= functions,
            disapproved_functions= disapproved_functions,
            code_issues= code_issues_prompt,
        ) + last_prompt

        # call the LLM
        response = safe_invoke(software_engineer, [SystemMessage(content= prompt)])
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {response}') if DEBUG else None

        return {'messages': [response]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state

# The node where tools are executed. Gets called specially when the Software Engineer calls the approve_function_code, approve_function_proposals, add_imports tools
def tool_node(state: InputSchema) -> InputSchema:
    '''
    This node executes the tools, which are used to gather information from the web, or structure the thought process or information.
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
        observations = []
        for tool_call in tool_calls:
            # Get the tool and arguments
            if from_kwargs:
                tool_call = tool_call['function']
            tool = tools_by_name[tool_call['name']]

            args = tool_call.get('args', {}) or tool_call.get('arguments', {})
            # Parse the tool arguments if needed.
            if isinstance(args, str):
                args = parse_tool_arguments(args)

            print(f'\n{BLUE}[NODE] [INFO] [TOOL CALL]{RESET} {tool_call["name"]} with {args}') if DEBUG else None

            try:
                observation = None
                # Execute the tool
                observation = tool.invoke(args)
                # Add the observation to the list
                if observation:
                    observations.append(observation)
            except Exception as e:
                print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
                traceback.print_exc() if DEBUG else None
                # If the tool fails, skip it
                continue

            print(f'{BLUE}[NODE] [INFO] [LAST OBSERVATION]{RESET} {observation}') if DEBUG else None

        # Add them to the state
        return {'messages': [
            ToolMessage(
                content= observation,
                name= tool_call['name'],
                tool_call_id= tool_call['id']
            ) for observation in observations
        ]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

# The Quality Assurance node where the Quality Assurance agent is prompted to validate the code
def last_check(state: InputSchema) -> InputSchema:
    '''
    This node calls the Quality Assurance agent to validate the code and identify code issues.
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        prompt = prompts.LAST_CHECK_PROMPT.format(
            code= read_state_file(state).split('if __name__ ==')[0]
        )

        # call the LLM and update the code issues
        global code_issues
        code_issues = safe_invoke(code_validator, [SystemMessage(content= prompt)])
        print(f'{BLUE}[NODE] [INFO] [RESPONSE]{RESET} {code_issues}') if DEBUG else None

        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state



''' Conditional Functions '''
# This conditional logic is used to determine what to do after the Software Engineer: Call a tool, go to the last check, or go back to the Software Engineer
def after_software_engineer(state: InputSchema) -> Literal['last_check', 'software_engineer_node', 'approve_tool', 'tools']:
    '''
    This function is used to determine what to do after the Software Engineer: Call a tool, go to the last check, or go back to the Software Engineer
    Returns:
        Literal['last_check', 'software_engineer_node', 'tools']
        - Exact node names
    '''
    print_function_name() if DEBUG else None

    # Plain language response
    if not will_tool_call(state['messages']):
        return 'software_engineer_node'
    
    # Get the last message and extract the tool calls
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', [])

    # If the last message is not a tool call, go back to the coder node
    if tool_calls is []:
        return 'software_engineer_node'
    
    # Get the names of all called tools
    called_tools = []
    for tool_call in tool_calls:
        if 'function' in tool_call:
            called_tools.append(tool_call['function']['name'])

        else:
            called_tools.append(tool_call['name'])

    print(f'{BLUE}[NODE] [INFO] [CALLED TOOLS]{RESET} {called_tools}') if DEBUG else None

    # If the tool is the output tool (submit_final_code), go to the output node
    if 'submit_final_code' in called_tools:
        return 'last_check'
    
    # If the last tool call has the approve_function_code, approve_function_proposals, add_imports tools
    if any(called_tool in ['approve_function_code', 'approve_function_proposals', 'add_imports'] for called_tool in called_tools):
        return 'approve_tool'

    # Else, go to the tool node
    return 'tools'

# This conditional logic is used to determine what to do after the Quality Assurance: Pass the last check, or go back to the Software Engineer to fix the issues
def passed_last_check(state: InputSchema) -> Literal['software_engineer_node', '__end__']:
    '''
    This function is used to determine what to do after the Quality Assurance: Pass the last check, or go back to the Software Engineer to fix the issues
    Returns:
        Literal['software_engineer_node', '__end__']
        - Exact node names
    '''
    print_function_name() if DEBUG else None

    # If the code issues are empty, end
    if not code_issues.issues:
        return '__end__'
    
    return 'software_engineer_node' 



''' Graph '''
software_engineer_graph = StateGraph(InputSchema)

software_engineer_graph.add_node('add_tool_sections', add_tool_sections)
software_engineer_graph.add_node('software_engineer_node', software_engineer_node)
software_engineer_graph.add_node('approve_tool', tool_node)
software_engineer_graph.add_node('tools', ToolNode(tools))
software_engineer_graph.add_node('last_check', last_check)

software_engineer_graph.add_edge(START, 'add_tool_sections')
software_engineer_graph.add_edge('add_tool_sections', 'software_engineer_node')
software_engineer_graph.add_conditional_edges(
    'software_engineer_node', 
    after_software_engineer,
    {   # Not needed, for clarity
        'last_check': 'last_check',
        'software_engineer_node': 'software_engineer_node',
        'approve_tool': 'approve_tool',
        'tools': 'tools'
    }
)
software_engineer_graph.add_edge('approve_tool', 'software_engineer_node')
software_engineer_graph.add_edge('tools', 'software_engineer_node')
software_engineer_graph.add_conditional_edges(
    'last_check', 
    passed_last_check,
    {   # Not needed, for clarity
        'software_engineer_node': 'software_engineer_node',
        '__end__': END
    }
)

software_engineer_app = software_engineer_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(software_engineer_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
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
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'softwareEngineer',
            'run_name': 'softwareEngineer',
            # 'thread_id': 'softwareEngineer', 
        }
    }

    user = InputSchema(file_path= '..\..\creations\whatsapp_menu_suggestion_workflow\whatsapp_menu_suggestion_workflow.py')
    response = software_engineer_app.invoke(user, config= config)

    # print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    # if DEBUG:
    #     for key, value in response.items():
    #         print(f'    {key}: {value}\n')
