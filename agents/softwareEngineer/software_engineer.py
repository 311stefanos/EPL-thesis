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
from typing import Literal, List, Optional, Dict, Set, Tuple
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import json
import os

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
# A set of import strings to be added to the code
imports: Set[str] = set()



''' Tools '''
# A tool used to write code to a file as a whole.
@tool
def replace_code(file_path: str, old_code: str, new_code: str) -> str:
    '''
    `replace_code` replaces the old code with the new code in the file.
    New code can have the same code as old code with some modifications or additions.
    Basically it just calls `code.replace(old_code, new_code)`.

    `Args:`
        file_path (str): The path to the file.
        old_code (str): The old code.
        new_code (str): The new code.

    `Returns:`
        (str) TA confirmation message.
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    try:
        old_code = clean_llm_output(old_code)
        new_code = clean_llm_output(new_code)

        # Read the code from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Replace the old code with the new code
        code = code.replace(old_code, new_code)

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
            'proposals': None,
            'imports': None
        }

    # If the coder does not exist in the coders dict, add it
    if function_name not in coders:
        coders[function_name] = CoderSchema(code= '', proposals= None, imports= None, approved= False, disapproved= False)
        comments[function_name] = CoderComment()

    # Call the coder
    args: CoderInputSchema = {
        'messages': [],
        'file_path': file_path,
        'function_name': function_name,
        'software_engineer_instructions': special_instructions,
        'previous_outputs': [coders[function_name].code] if coders[function_name].code else [],
        'comments': [comments[function_name].comment] if comments[function_name].comment else [],
        'previous_implementation': None,
        'reviewer_comments': None
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
    coders[function_name] = CoderSchema(code= response['code'], proposals= response['proposals'], imports= response['imports'], approved= False, disapproved= False)

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
    global coders, comments, imports

    # Check if the coder exists
    all_keys = set(list(coders.keys()) + list(comments.keys()))
    if function_name not in all_keys:
        print(f'{RED}[TOOL] [ERROR] [APPROVE]{RESET} The coder for function {function_name} does not exist.') if DEBUG else None
        return f'[ERROR] The coder for function {function_name} does not exist.'
    
    # Check if the coder has an implementation
    previous_code: str = coders[function_name].code.strip()
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
    # Get only the function that was implemented, in order to update it
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

    # print(f'{BLUE}[TOOL] [OLD SECTION]{RESET} {code_section}') if DEBUG else None
    # print(f'{BLUE}[TOOL] [NEW SECTION]{RESET} {previous_code}') if DEBUG else None

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    # Add the imports to the set
    if coders[function_name].imports:
        imports.update(coders[function_name].imports)
        
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
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    code = code.replace('# TODO: Add Tools (if needed)', '\n\n'.join(proposed_tools) + '\n\n# TODO: Add Tools (if needed)')
    code = code.replace('# TODO: Add Helpful Functions (if needed)', '\n\n'.join(proposed_functions) + '\n\n# TODO: Add Helpful Functions (if needed)')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(code)

    print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Approved the coder\'s function proposals: {[afp.function_name for afp in approved_function_proposals]}') if DEBUG else None
    return f'[SUCCESS] Approved the coder\'s function proposals: {[afp.function_name for afp in approved_function_proposals]}.\n\nThe file contents have been updated.'

# A tool to add imports fast
@tool
def add_imports(new_imports: List[str], file_path: str) -> str:
    '''
    `add_imports` adds imports to the file. you should use this tool as little as possible.

    `Args:`
        new_imports (List[str]): The new imports to add to the file. Should input the full import line.
        file_path (str): The path to the file to implement the function in.
    
    `Returns:`
        (str) Either a success message or an error message
    '''
    def _is_simple_from_import_line(s: str) -> bool:
        s = s.strip()
        if not (s.startswith('from ') and ' import ' in s):
            return False
        # avoid editing multiline/parenthesized styles
        if '(' in s or ')' in s or '\\' in s:
            return False
        return True

    def _parse_from_import_line(s: str) -> Tuple[str, List[str]]:
        # assumes `from X import a, b`
        s = s.strip()
        before, after = s.split(' import ', 1)
        module = before.replace('from', '', 1).strip()
        names = [x.strip() for x in after.split(',') if x.strip()]
        return module, names

    def _is_simple_import_line(s: str) -> bool:
        s = s.strip()
        if not s.startswith('import '):
            return False
        if '(' in s or ')' in s or '\\' in s:
            return False
        return True

    def _parse_import_line(s: str) -> List[str]:
        # `import os, sys` or `import os as myos`
        s = s.strip()[len('import ') :]
        mods = [x.strip() for x in s.split(',') if x.strip()]
        return mods
    
    print_function_name(colour= MAGENTA) if DEBUG else None

    global imports
    
    try:
        code = read_state_file({'file_path': file_path})

        start_marker = "''' Imports '''"
        end_marker = "''' Constants '''"

        start_idx = code.find(start_marker)
        end_idx = code.find(end_marker)

        if start_idx == -1:
            return f'[ERROR] Cannot import.'
        if end_idx == -1 or end_idx <= start_idx:
            return f'[ERROR] Cannot import.'

        import_block = code[start_idx:end_idx]
        lines = import_block.splitlines()

        from_line_idx: Dict[str, int] = {}
        from_names: Dict[str, List[str]] = {}
        from_name_set: Dict[str, Set[str]] = {}

        import_line_idx: Dict[str, int] = {}
        import_mods: Dict[str, List[str]] = {}
        import_mod_set: Dict[str, Set[str]] = {}

        for i, line in enumerate(lines):
            stripped = line.strip()

            if _is_simple_from_import_line(stripped):
                module, names = _parse_from_import_line(stripped)
                # only track the first simple line per module to avoid messing with formatting choices
                if module not in from_line_idx:
                    from_line_idx[module] = i
                    from_names[module] = names[:]  # keep order
                    from_name_set[module] = set(names)

            if _is_simple_import_line(stripped):
                mods = _parse_import_line(stripped)
                # track each module token in this import line
                for m in mods:
                    if m not in import_line_idx:
                        import_line_idx[m] = i

                # also track the line’s full list for extension (keyed by line index)
                # We’ll store per-line lists using a synthetic key.
                key = f'__line__{i}'
                if key not in import_mods:
                    import_mods[key] = mods[:]
                    import_mod_set[key] = set(mods)

        # map line index -> synthetic key for import lines we can edit
        linekey_for_import_line: Dict[int, str] = {
            int(k.replace('__line__', '')): k for k in import_mods.keys()
        }

        # ---- apply new imports ----
        appended_lines: List[str] = []
        removed_from_global: List[object] = []

        for raw in new_imports:
            if not raw or not raw.strip():
                continue
            new_line = raw.strip()

            # FROM-import case
            if new_line.startswith('from ') and ' import ' in new_line:
                # if new line is complex, just append as-is
                if not _is_simple_from_import_line(new_line):
                    appended_lines.append(new_line)
                    removed_from_global.append(raw)
                    continue

                module, new_names_list = _parse_from_import_line(new_line)

                if module in from_line_idx:
                    # extend existing line with missing names only, keeping order
                    existing_list = from_names[module]
                    existing_set = from_name_set[module]

                    added_any = False
                    for nm in new_names_list:
                        if nm not in existing_set:
                            existing_list.append(nm)
                            existing_set.add(nm)
                            added_any = True

                    if added_any:
                        idx = from_line_idx[module]
                        lines[idx] = f"from {module} import {', '.join(existing_list)}"

                    removed_from_global.append(raw)
                else:
                    appended_lines.append(new_line)
                    removed_from_global.append(raw)

                continue

            # IMPORT case
            if new_line.startswith('import '):
                if not _is_simple_import_line(new_line):
                    appended_lines.append(new_line)
                    removed_from_global.append(raw)
                    continue

                new_mods = _parse_import_line(new_line)

                # figure out which ones are already present anywhere in simple import lines
                # (either in import-line sets, or as imported modules via from-lines doesn't count)
                already_imported: Set[str] = set()
                for key, sset in import_mod_set.items():
                    already_imported |= sset

                missing = [m for m in new_mods if m not in already_imported]

                if not missing:
                    removed_from_global.append(raw)
                    continue

                # if we have at least one editable import line, extend the first one
                editable_import_line_indices = sorted(linekey_for_import_line.keys())
                if editable_import_line_indices:
                    target_i = editable_import_line_indices[0]
                    key = linekey_for_import_line[target_i]
                    existing_list = import_mods[key]
                    existing_set = import_mod_set[key]

                    for m in missing:
                        if m not in existing_set:
                            existing_list.append(m)
                            existing_set.add(m)

                    lines[target_i] = f"import {', '.join(existing_list)}"
                else:
                    # no editable import line exists, append a minimal import line
                    appended_lines.append(f"import {', '.join(missing)}")

                removed_from_global.append(raw)
                continue

            # Unknown format: keep as-is
            appended_lines.append(new_line)
            removed_from_global.append(raw)

        # Append new lines near the end of the import block, preserving one blank line
        # Find last non-empty line in import block
        last_nonempty = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                last_nonempty = i
                break

        if appended_lines:
            insert_at = last_nonempty + 1 if last_nonempty != -1 else len(lines)
            # ensure there is exactly one blank line before appends if needed
            if insert_at > 0 and lines[insert_at - 1].strip() != '':
                lines.insert(insert_at, '')
                insert_at += 1
            for l in appended_lines:
                lines.insert(insert_at, l)
                insert_at += 1
            # ensure trailing blank line (optional, but keeps sections readable)
            if lines and lines[-1].strip() != '':
                lines.append('')

        new_import_block = '\n'.join(lines) + '\n'

        new_code = code[:start_idx] + new_import_block + code[end_idx:]

        print(f'{BLUE}[TOOL] [OLD SECTION]{RESET}\n{import_block}') if DEBUG else None
        print(f'{BLUE}[TOOL] [NEW SECTION]{RESET}\n{new_import_block}') if DEBUG else None

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_code)

        # Best-effort removal from global `imports` (whatever its internal representation is)
        try:
            for item in removed_from_global:
                if item in imports:
                    imports.remove(item)
        except Exception:
            # don't fail the tool because of bookkeeping
            pass

        return '[SUCCESS] Added the new imports to the file.'

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
def code_issue_resolved(resolved_issues: List[str]) -> str: # TODO: add index
    '''
    `code_issue_resolved` resolves a code issue that was proposed by the Quality Assurance team.

    `Args:`
        resolved_issues (List[str]): The issues that have been resolved. Must must exact wording as given from the Quality Assurance team (can be found in the prompt).

    `Returns:`
        (str) Either a success message or an error message
    '''
    print_function_name(colour= MAGENTA) if DEBUG else None

    global code_issues

    if len(resolved_issues) == 0:
        print(f'{BLUE}[TOOL] [INFO] [NO ISSUES]{RESET} No code to were resolved.') if DEBUG else None
        return '[NO ISSUES] No code issues to resolved.'

    # Split the 'resolved' code issues into resolved and not resolved
    not_resolved_issues = []
    actually_resolved_issues = []

    for resolved_issue in resolved_issues:
        # Check if the code issue exists
        if resolved_issue not in code_issues:
            print(f'{RED}[TOOL] [ERROR] [RESOLVE]{RESET} The code issue {resolved_issue} does not exist.') if DEBUG else None
            not_resolved_issues.append(resolved_issue)
        
        elif resolved_issue in code_issues:
            code_issues.issues.remove(resolved_issue)
            actually_resolved_issues.append(resolved_issue)

    print(f'{BLUE}[TOOL] [INFO] [SUCCESS]{RESET} Resolved the coder\'s code issues: {actually_resolved_issues}') if DEBUG else None
    print(f'{RED}[TOOL] [FAIL] [NOT RESOLVED]{RESET} The following code issues were not resolved: {not_resolved_issues}') if DEBUG else None
    return f'[SUCCESS] Resolved the coder\'s code issues: {actually_resolved_issues}\n[FAIL] The following code issues were not resolved (due to not exact wording): {not_resolved_issues}'

tools = [
    replace_code, 
    call_coder, 
    disapprove_and_comment_on_coder_code, 
    approve_function_code, 
    approve_function_proposals, 
    add_imports, 
    submit_final_code, 
    code_issue_resolved
]

# Dictionary of tools: tool name -> tool
tools_by_name = {tool.name: tool for tool in tools}

''' LLM '''
# The agent that adds the tool sections
tool_adder = myChatOpenAI(
    temperature= 0.4,
    model= 'mistralai/devstral-2512:free'
)

# The Software Engineer that orchestrates the tools
software_engineer = myChatOpenAI(
    temperature= 0.4,
    model= 'mistralai/devstral-2512:free'
).bind_tools(tools)

# The Quality Assurance team that validates the code and proposes code issues
code_validator = myChatOpenAI(
    temperature= 0.6,
    model= 'mistralai/devstral-2512:free'
).with_structured_output(CodeIssues)

# TODO: add model to create necessary files. for files that require keys, insert [dummy data]



''' Helpful Functions '''
# Remove heading and trailing tags or markdown special characters
def clean_llm_output(code: str) -> str:
    '''
    `clean_llm_output` removes heading and trailing tags or markdown special characters from the LLM's output

    `Args:`
        code (str): The LLM's output

    `Returns:`
        code: str
    '''
    if not code:
        return code

    # Remove possible tags or markdown special characters from the LLM's output
    while code[0] in ['<', '`']:
        # Removing tags
        while code.strip().startswith('<'):
            # Remove the line
            index = code.find('\n')
            code = code[index + 1:].strip()
            
        while code.strip().endswith('>'):
            for i, char in enumerate(reversed(code)):
                if char == '<':
                    index = len(code) - i
                    code = code[:index].strip()
                    break

        # Removing markdown ```
        while code.strip().startswith('`'):
            # Remove the line
            index = code.find('\n')
            code = code[index + 1:].strip()

        while code.strip().endswith('`'):
            for i, char in enumerate(reversed(code)):
                if char == '`':
                    index = len(code) - i - 1
                    code = code[:index].strip()
                    break

    return code.strip()

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
# This nodes is called first, and the agent tries to make the file tool compatible.
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

        cleaned_response = clean_llm_output(response)

        with open(state['file_path'], 'w', encoding='utf-8') as f:
            f.write(cleaned_response)

        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc()

        return state

# The Software Engineer node where the Software Engineer is prompted to call the tools
def software_engineer_node(state: InputSchema) -> InputSchema:
    '''
    This node calls the Software Engineer to implement the code with the help of various tools.
    Available tools:
    - replace_code
    - call_coder
    - disapprove_and_comment_on_coder_code
    - approve_function_code
    - approve_function_proposals
    - add_imports
    - submit_final_code
    - code_issue_resolved
    '''
    print_function_name() if DEBUG else None

    global coders, code_issues, imports

    try:
        # prompt
        # A simple line to guide the Software Engineer through the process
        last_message = state['messages'][-1] if state['messages'] else None
        last_prompt = ''
        if hasattr(last_message, 'name'):
            if last_message.name == 'call_coder':
                last_prompt = '\n# Next Step:\nApprove requests, Approve code, Disapprove code: using the respective tools.\n\n'

        # Get the code issues from the Quality Assurance
        issues = '' # TODO: add indexing
        for issue_schema in code_issues.issues:
            issue = issue_schema.issue
            comment = issue_schema.comment
            if comment:
                issues += f'- Issue: {issue}\n    Comment: {comment} (Do not include comment in the `code_issue_resolved` tool call)\n\n'
            else:
                issues += f'- Issue: {issue}\n\n'

        code_issues_prompt = prompts.CODE_ISSUES_SECTION.format(
            code_issues= issues
        ) if issues else ''

        # Files under file_path/..
        files = '\n- '.join([f.name for f in Path(state['file_path']).parent.iterdir() if f.is_file()])
        
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

        # Parse the imports
        parsed_imports = '\n'.join([f'- {f}' for f in imports])
        
        prompt = prompts.SOFTWARE_ENGINEER_PROMPT.format(
            file_path= state['file_path'],
            code= read_state_file(state),
            tool_messages= '\n\n'.join([message.pretty_repr() for message in state['messages'][-3:]]),
            files= files,
            functions= functions,
            disapproved_functions= disapproved_functions,
            imports= parsed_imports,
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
    This node executes the tools, which change the contents of the file
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

        # Check whether add_imports was called during the last 3 messages
        add_imports_called_before = False
        last_5_messages = state['messages'][-5:]
        for message in last_5_messages:
            if (
                (hasattr(message, 'tool_calls') and message.tool_calls) or
                (hasattr(message, 'additional_kwargs') and message.additional_kwargs.get('tool_calls', False))
            ):
                for tool_call in message.tool_calls:
                    if tool_call['name'] == 'add_imports':
                        add_imports_called_before = True

        print(json.dumps(tool_calls, indent= 4)) if DEBUG else None

        # Execute all tool calls
        observations = []
        for tool_call in tool_calls:
            # Get the tool and arguments
            if from_kwargs:
                tool_call = tool_call['function']

            # Skip add_imports if it was called before
            if tool_call['name'] == 'add_imports' and add_imports_called_before:
                err_msg = '[ERR] You cannot add imports because add_imports was called shorty before'
                observations.append((err_msg, tool_call['name'], tool_call['id']))

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
                    observations.append((observation, tool_call['name'], tool_call['id']))
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
                name= name,
                tool_call_id= id_
            ) for (observation, name, id_) in observations
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
        print(f'{BLUE}[NODE] [INFO] [NO TOOL CALL]{RESET} Back to Software Engineer') if DEBUG else None
        return 'software_engineer_node'
    
    # Get the last message and extract the tool calls
    last_message = state['messages'][-1]
    tool_calls = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', [])

    # If the last message is not a tool call, go back to the coder node
    if tool_calls is []:
        print(f'{BLUE}[NODE] [INFO] [NO TOOL CALL]{RESET} Back to Software Engineer') if DEBUG else None
        return 'software_engineer_node'
    
    # Get the names of all called tools
    called_tools = []
    for tool_call in tool_calls:
        if 'function' in tool_call:
            called_tools.append((tool_call['function']['name'], tool_call['function']['args']))

        else:
            called_tools.append((tool_call['name'], tool_call['args']))

    formatted_called_tools = '\n- '.join([f'{tool_name} {args}' for (tool_name, args) in called_tools])
    print(f'{BLUE}[NODE] [INFO] [CALLED TOOLS]{RESET} {formatted_called_tools}') if DEBUG else None

    # If the tool is the output tool (submit_final_code), go to the output node
    if 'submit_final_code' in [called_tool[0] for called_tool in called_tools]:
        print(f'{BLUE}[NODE] [INFO] [TOOL CALL]{RESET} submit_final_code') if DEBUG else None
        return 'last_check'
    
    # If the last tool call has the approve_function_code, approve_function_proposals, add_imports tools
    if any(called_tool[0] in ['approve_function_code', 'approve_function_proposals', 'add_imports'] for called_tool in called_tools):
        print(f'{BLUE}[NODE] [INFO] [APPROVAL TOOL CALL]{RESET} approve_tool') if DEBUG else None
        return 'approve_tool'

    # Else, go to the tool node
    print(f'{BLUE}[NODE] [INFO] [TOOL CALL]{RESET} tools') if DEBUG else None
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

# software_engineer_graph.add_edge(START, 'add_tool_sections')
# software_engineer_graph.add_edge('add_tool_sections', 'software_engineer_node')
software_engineer_graph.add_edge(START, 'software_engineer_node')
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
        'recursion_limit': 100, # TODO: change
        'configurable': {
            'user_id': 'softwareEngineer',
            'run_name': 'softwareEngineer',
            # 'thread_id': 'softwareEngineer', 
        }
    }

    user = InputSchema(file_path= '..\..\creations\menu_recommendation_workflow\menu_recommendation_workflow.py')
    response = software_engineer_app.invoke(user, config= config)

    # print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    # if DEBUG:
    #     for key, value in response.items():
    #         print(f'    {key}: {value}\n')
