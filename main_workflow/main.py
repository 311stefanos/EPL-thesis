# Langchain imports
from langchain_core.messages import BaseMessage
from langsmith import Client

# General imports
from typing import List, Literal, Dict, Callable, Any
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import argparse
import uuid
import json
import os

# My imports (ordered by call order)
from agents.inputRefiner.input_refiner import input_refiner_app
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
from utils.build_code import create_file
from agents.codeAnnotator.code_annotator import code_annotator_app
from agents.softwareEngineer.software_engineer import software_engineer_app
from agents.promptEngineer.prompt_engineer import prompt_engineer_app
from agents.fileHandler.file_handler import file_handler_app



load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')


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

def print_to_file(agent_name: str, result: dict, run_name: str) -> None:
    '''
    `print_to_file` prints the result of the agent to a file.
    
    `Args:`
        `agent_name` (str): The unique name of the agent/log entry.
        `result` (dict): The result of the agent.
        `run_name` (str): The run directory under `./logs/`.
    '''
    if not DEBUG:
        return

    def to_serializable(value: Any) -> Any:
        # Pydantic v2
        if hasattr(value, 'model_dump') and callable(value.model_dump):
            return value.model_dump()

        # Pydantic v1
        if hasattr(value, 'dict') and callable(value.dict):
            return value.dict()

        if isinstance(value, dict):
            return {
                key: to_serializable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [to_serializable(item) for item in value]

        return value

    log_dir: Path = Path('./logs') / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / f'{agent_name}.txt', 'w', encoding= 'utf-8') as f:
        for key, value in result.items():
            if key == 'messages':
                f.write('Messages:\n')
                for message in value:
                    message: BaseMessage
                    f.write(f'{message.pretty_repr()}\n')

                f.write('\n')
                continue

            try:
                serializable_value = to_serializable(value)

                if isinstance(serializable_value, (dict, list)):
                    f.write(f'{key}:\n{json.dumps(serializable_value, indent=4, default=str)}\n\n')
                else:
                    f.write(f'{key}: {serializable_value}\n\n')

            except Exception as e:
                f.write(f'{key}: {str(value)}\n')
                f.write(f'[Serialization error: {e}]\n\n')

def copy_file(after_agent_name: str, file_path: str, run_name: str) -> None:
    '''
    `copy_file` copies the file to the run's log directory.
    
    `Args:`
        `after_agent_name` (str): The unique stage name before this function gets called.
        `file_path` (str): The path of the file to copy.
        `run_name` (str): The run directory under `./logs/`.
    '''
    if not DEBUG:
        return
    
    with open(file_path, 'r', encoding= 'utf-8') as f:
        contents = f.read()

    log_dir: Path = Path('./logs') / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(log_dir / f'after_{after_agent_name}.py', 'w', encoding= 'utf-8') as f:
        f.write(contents)



def main(user_request: str, orchestrator: bool=True, prompt_review_mode: Literal['llm', 'user', 'both']='both', coder_run_code: bool=False, run_name: str='run') -> None:
    '''
    `main` is the main function of the program.
    It invokes the input refiner, workflow refiner, code annotator, software engineer, prompt engineer and file handler agents.
    
    `Args:`
        `user_request` (str): The user request.
        `orchestrator` (bool): Whether to use the orchestrator.
        `prompt_review_mode` (Literal['llm', 'user', 'both']): The prompt review mode.
        `coder_run_code` (bool): Whether the Coder should run generated code during review.
        `run_name` (str): The directory name under `./logs/` used for this run.
    '''
    # Config
    uuid_: str = str(uuid.uuid4())
    date: str = datetime.now().strftime('%d-%m-%y')

    config: Callable[[str], Dict] = lambda agent_name: {
        'recursion_limit': 150,
        'configurable': {
            'user_id': 'main',
            'run_name': f'main:{uuid_}',
            'thread_id': f'main:{agent_name}:{date}:{uuid_}',
        }
    }

    # Connect to langsmith
    os.environ['LANGCHAIN_PROJECT'] = 'main_workflow'
    os.environ['LANGSMITH_PROJECT'] = 'main_workflow'
    client = Client()

    # Initial Request
    print_to_file('initial_request', {'initial_request': user_request}, run_name)

    # Input Refiner
    print_agent('Input Refiner (internal: Clarification Orchestrator)')
    input_refiner_response = input_refiner_app.invoke({
        'orchestrator': orchestrator, 
        'user_input': user_request
    }, config= config('input_refiner'))
    print_to_file('input_refiner', input_refiner_response, run_name)
    clarified_user_input = input_refiner_response['refined_text']

    # Workflow Refiner
    print_agent('Workflow Refiner (internal: Clarification Orchestrator)')
    workflow_refiner_response = workflow_refiner_app.invoke({
        'messages': [], 
        'orchestrator': orchestrator, 
        'clarified_user_input': clarified_user_input
    }, config= config('workflow_refiner'))
    print_to_file('workflow_refiner', workflow_refiner_response, run_name)
    workflow_bundle = workflow_refiner_response['workflow']

    # Create code structures
    files: List[str] = create_file(workflow_bundle)
    for file_index, file in enumerate(files, start=1):
        agent_name: str = Path(file).stem
        file_id: str = f'{file_index}_{agent_name}'

        # Save initial code structure
        copy_file(f'{file_id}_code_structure', file, run_name)

        # Code Annotator
        print_agent(f'Code Annotator (file: {file})')
        code_annotator_response = code_annotator_app.invoke({
            'messages': [], 
            'file_path': file, 
            'clarified_user_input': clarified_user_input, 
            'workflow': workflow_bundle
        }, config= config(f'code_annotator:{agent_name}'))
        print_to_file(f'{file_id}_code_annotator', code_annotator_response, run_name)
        copy_file(f'{file_id}_code_annotator', file, run_name)

        # Software Engineer
        internal = 'Coder' + ('& CodeTester' if coder_run_code else '')
        print_agent(f'Software Engineer (file: {file}) (internal: {internal})')
        software_engineer_response = software_engineer_app.invoke({
            'messages': [], 
            'user_request': clarified_user_input,
            'file_path': file, 
            'times_reviewed': 0, 
            'skip_tool_sections': True, 
            'coder_run_code': coder_run_code
        }, config= config(f'software_engineer:{agent_name}'))
        print_to_file(f'{file_id}_software_engineer', software_engineer_response, run_name)
        copy_file(f'{file_id}_software_engineer', file, run_name)

        # Prompt Engineer
        print_agent(f'Prompt Engineer (file: {file})')
        prompt_engineer_response = prompt_engineer_app.invoke({
            'file_path': file, 
            'mode': prompt_review_mode
        }, config= config(f'prompt_engineer:{agent_name}'))
        print_to_file(f'{file_id}_prompt_engineer', prompt_engineer_response, run_name)
        copy_file(f'{file_id}_prompt_engineer', file, run_name)

        # File Handler
        print_agent(f'File Handler (file: {file})')
        file_handler_response = file_handler_app.invoke({
            'messages': [], 
            'file_path': file
        }, config= config(f'file_handler:{agent_name}'))
        print_to_file(f'{file_id}_file_handler', file_handler_response, run_name)
        copy_file(f'{file_id}_file_handler', file, run_name)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'run_name', 
        type= str, 
        help= 'Directory name under ./logs/ used to store the logs for this run.'
    )
    args = parser.parse_args()

    user_request: str = (
        'I want a personal agent that I can send my receipts (in a photo) through a WhatsApp chat,'
        ' and it should read the photo, understand the receipt (cost, date, items, location, etc.), and insert the'
        ' data into an Excel file. The Excel file should have columns for cost, date, items (format: item1 (quantity'
        ' x price currency), ...), location, category. The category should be automatically determined from the'
        ' agent based on the items. It should also answer questions about spending when prompted by the user'
        ' (e.g. ‘how much did I spend on groceries this month?’, ‘what are the top 3 items I spent the most on?’,'
        ' etc.). For OCR processing, you may call a vision model through OpenRouter.'
        ' For the WhatsApp API, consider it out-of-scope.'
    )

    # python main.py <run_name>

    main(
        user_request, 
        orchestrator= True, 
        prompt_review_mode= 'both', 
        coder_run_code= True, 

        run_name= args.run_name
    )