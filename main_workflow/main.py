from typing import List, Literal

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
    config = {
        'recursion_limit': 150,
        'configurable': {
            'user_id': 'main',
            'run_name': 'main',
            'thread_id': 'main', 
        }
    }

    print_agent('Input Refiner (internal: Classification Orchestrator)')
    input_refiner_response = input_refiner_app.invoke({
        'orchestrator': orchestrator,
        'user_input': user_request
    }, config= config) # corrected_original, refined_text
    clarified_user_input = input_refiner_response['refined_text']

    print_agent('Workflow Refiner (internal: Classification Orchestrator)')
    workflow_refiner_response = workflow_refiner_app.invoke({
        'orchestrator': orchestrator,
        'clarified_user_input': clarified_user_input
    }, config= config) # workflow
    workflow_bundle = workflow_refiner_response['workflow']

    files: List[str] = create_file(workflow_bundle)

    for file in files:
        print_agent(f'Code Annotator (file: {file})')
        code_annotator_app.invoke({
            'file_path': file,
            'clarified_user_input': clarified_user_input,
            'workflow': workflow_bundle,
        }, config= config)

        print_agent(f'Software Engineer (file: {file}) (internal: Coder)')
        software_engineer_app.invoke({
            'file_path': file,
            'times_reviewed': 0
        }, config= config)

        print_agent(f'Prompt Engineer (file: {file})')
        prompt_engineer_app.invoke({
            'file_path': file,
            'mode': prompt_review_mode
        }, config= config)

        print_agent(f'File Handler (file: {file})')
        file_handler_app.invoke({
            'file_path': file
        }, config= config)



if __name__ == '__main__':
    user_request: str = 'I want an agent that helps me with math questions. I can provide my lecture notes in documents for the agent to read or RAG.'
    main(
        user_request,
        orchestrator= True,
        prompt_review_mode= 'both'
    )