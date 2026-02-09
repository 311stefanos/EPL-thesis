# General imports
from typing import List, Literal, Dict, Callable
from datetime import datetime
import uuid

# My imports (ordered by call order)
from agents.inputRefiner.input_refiner import input_refiner_app
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
from utils.build_code import create_file
from agents.codeAnnotator.code_annotator import code_annotator_app
from agents.softwareEngineer.software_engineer import software_engineer_app
from agents.promptEngineer.prompt_engineer import prompt_engineer_app
from agents.fileHandler.file_handler import file_handler_app

BRIGHT_WHITE = '\n\033[1m\033[4m\033[97m'
RESET = '\033[0m'

def print_agent(agent_name: str) -> None:
    print(f'{BRIGHT_WHITE}[NEXT AGENT] {agent_name}{RESET}')

def main(user_request: str, orchestrator: bool= True, prompt_review_mode: Literal['llm', 'user', 'both'] = 'both') -> None:
    uuid_: str = str(uuid.uuid4())
    date: str = datetime.now().strftime('%d/%m/%y')
    config: Callable[[str], Dict] = lambda agent_name: {
        'recursion_limit': 150,
        'configurable': {
            'user_id': 'main',
            'run_name': 'main',
            'thread_id': f'main:{agent_name}:{date}:{uuid_}',
        }
    }

    print_agent('Input Refiner (internal: Classification Orchestrator)')
    input_refiner_response = input_refiner_app.invoke({
        'orchestrator': orchestrator,
        'user_input': user_request
    }, config= config('input_refiner')) # corrected_original, refined_text
    clarified_user_input = input_refiner_response['refined_text']

    print_agent('Workflow Refiner (internal: Classification Orchestrator)')
    workflow_refiner_response = workflow_refiner_app.invoke({
        'messages': [],
        'orchestrator': orchestrator,
        'clarified_user_input': clarified_user_input
    }, config= config('workflow_refiner')) # workflow
    workflow_bundle = workflow_refiner_response['workflow']

    files: List[str] = create_file(workflow_bundle)

    for file in files:
        print_agent(f'Code Annotator (file: {file})')
        code_annotator_app.invoke({
            'messages': [],
            'file_path': file,
            'clarified_user_input': clarified_user_input,
            'workflow': workflow_bundle,
        }, config= config(f'code_annotator:{file}'))

        print_agent(f'Software Engineer (file: {file}) (internal: Coder)')
        software_engineer_app.invoke({
            'messages': [],
            'file_path': file,
            'times_reviewed': 0
        }, config= config(f'software_engineer:{file}'))

        print_agent(f'Prompt Engineer (file: {file})')
        prompt_engineer_app.invoke({
            'file_path': file,
            'mode': prompt_review_mode
        }, config= config(f'prompt_engineer:{file}'))

        print_agent(f'File Handler (file: {file})')
        file_handler_app.invoke({
            'messages': [],
            'file_path': file
        }, config= config(f'file_handler:{file}'))



if __name__ == '__main__':
    user_request: str = 'I want an agent to help me find relevant academic information/books/articles on a given topic. it should find the most relevant information about a topic discussed in a conversation, summarise it by article/book/... provide links and references. it should be an academic assistant.'
    main(
        user_request,
        orchestrator= True,
        prompt_review_mode= 'both'
    )