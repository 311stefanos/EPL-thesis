# Langchain imports
from langchain_core.messages import BaseMessage

# General imports
from typing import List, Literal, Dict, Callable
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import uuid
import os

# My imports (ordered by call order)
from agents.inputRefiner.input_refiner import input_refiner_app
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
from utils.build_code import create_file
from agents.codeAnnotator.code_annotator import code_annotator_app
from agents.softwareEngineer.software_engineer import software_engineer_app
from agents.promptEngineer.prompt_engineer import prompt_engineer_app
from agents.fileHandler.file_handler import file_handler_app



load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent / '.env')

''' Constants '''
DEBUG = os.getenv('DEBUG')
BRIGHT_WHITE = '\n\033[1m\033[4m\033[97m'
RESET = '\033[0m'

def print_agent(agent_name: str) -> None:
    '''
    `print_agent` prints the name of the next agent to be called.
    
    `Args:`
        `agent_name` (str): The name of the next agent to be called.
    '''
    print(f'{BRIGHT_WHITE}[NEXT AGENT] {agent_name}{RESET}')

def print_to_file(agent_name: str, result: dict, date: str) -> None:
    '''
    `print_to_file` prints the result of the agent to a file.
    
    `Args:`
        `agent_name` (str): The name of the agent.
        `result` (dict): The result of the agent.
        `date` (str): The date of the run.
    '''
    if not DEBUG:
        return
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    
    if not os.path.exists(f'./logs/logs_{date}'):
        os.makedirs(f'./logs/logs_{date}')

    with open(f'./logs/logs_{date}/{agent_name}.txt', 'w', encoding= 'utf-8') as f:
        for key, value in result.items():
            if key == 'messages':
                f.write('Messages:\n')
                for message in value:
                    message: BaseMessage
                    f.write(f'{message.pretty_repr()}\n')
                continue
            try:
                f.write(f'{key}: {str(value)}\n\n')
            except Exception as e:
                f.write(e)

def copy_file(after_agent_name: str, file_path: str, date: str) -> None:
    '''
    `copy_file` copies the file to a new file.
    
    `Args:`
        `after_agent_name` (str): The name of the agent before this function gets called.
        `file_path` (str): The path to the file to copy.
        `date` (str): The date of the run.
    '''
    if not DEBUG:
        return
    
    with open(file_path, 'r') as f:
        contents = f.read()

    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    if not os.path.exists(f'./logs/logs_{date}'):
        os.makedirs(f'./logs/logs_{date}')

    with open(f'./logs/logs_{date}/after_{after_agent_name}_{Path(file_path).name}', 'w', encoding= 'utf-8') as f:
        f.write(contents)

def main(user_request: str, orchestrator: bool= True, prompt_review_mode: Literal['llm', 'user', 'both'] = 'both') -> None:
    '''
    `main` is the main function of the program.
    It invokes the input refiner, workflow refiner, code annotator, software engineer, prompt engineer and file handler agents.
    
    `Args:`
        `user_request` (str): The user request.
        `orchestrator` (bool): Whether to use the orchestrator.
        `prompt_review_mode` (Literal['llm', 'user', 'both']): The prompt review mode.
    '''
    # Config
    uuid_: str = str(uuid.uuid4())
    date: str = datetime.now().strftime('%d-%m-%y')
    config: Callable[[str], Dict] = lambda agent_name: {
        'recursion_limit': 150,
        'configurable': {
            'user_id': 'main',
            'run_name': 'main',
            'thread_id': f'main:{agent_name}:{date}:{uuid_}',
        }
    }

    # Input Refiner
    print_agent('Input Refiner (internal: Clarification  Orchestrator)')
    input_refiner_response = input_refiner_app.invoke({
        'orchestrator': orchestrator,
        'user_input': user_request
    }, config= config('input_refiner')) # corrected_original, refined_text
    print_to_file('input_refiner', input_refiner_response, date)
    clarified_user_input = input_refiner_response['refined_text']

    # Workflow Refiner
    print_agent('Workflow Refiner (internal: Clarification  Orchestrator)')
    workflow_refiner_response = workflow_refiner_app.invoke({
        'messages': [],
        'orchestrator': orchestrator,
        'clarified_user_input': clarified_user_input
    }, config= config('workflow_refiner')) # workflow
    print_to_file('workflow_refiner', workflow_refiner_response, date)
    workflow_bundle = workflow_refiner_response['workflow']

    # Create code structures
    files: List[str] = create_file(workflow_bundle)
    for file in files:
        copy_file('code_structure', file, date)

        # Code Annotator
        print_agent(f'Code Annotator (file: {file})')
        code_annotator_response = code_annotator_app.invoke({
            'messages': [],
            'file_path': file,
            'clarified_user_input': clarified_user_input,
            'workflow': workflow_bundle,
        }, config= config(f'code_annotator:{file}'))
        print_to_file('code_annotator', code_annotator_response, date)
        copy_file('code_annotator', file, date)

        # Software Engineer
        print_agent(f'Software Engineer (file: {file}) (internal: Coder)')
        software_engineer_response = software_engineer_app.invoke({
            'messages': [],
            'file_path': file,
            'times_reviewed': 0,
            'skip_tool_sections': False
        }, config= config(f'software_engineer:{file}'))
        print_to_file('software_engineer', software_engineer_response, date)
        copy_file('software_engineer', file, date)

        # Prompt Engineer
        print_agent(f'Prompt Engineer (file: {file})')
        prompt_engineer_response = prompt_engineer_app.invoke({
            'file_path': file,
            'mode': prompt_review_mode
        }, config= config(f'prompt_engineer:{file}'))
        print_to_file('prompt_engineer', prompt_engineer_response, date)
        copy_file('prompt_engineer', file, date)

        # File Handler
        print_agent(f'File Handler (file: {file})')
        file_handler_response = file_handler_app.invoke({
            'messages': [],
            'file_path': file
        }, config= config(f'file_handler:{file}'))
        print_to_file('file_handler', file_handler_response, date)
        copy_file('file_handler', file, date)



if __name__ == '__main__':
    user_request: str = (
        'I want an agent that solves Python programming benchmark tasks such as HumanEval and MBPP. '
        'The agent should receive inputs based on the two mentioned benchmarks. '
        'It should understand the required function, generate correct Python code, '
        'optionally check the code for syntax or test failures, repair mistakes if needed, '
        'and return only the final Python solution in the format expected by the benchmark evaluator.'
    )
    main(
        user_request,
        orchestrator= True,
        prompt_review_mode= 'llm'
    )