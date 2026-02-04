from agents.inputRefiner.input_refiner import input_refiner_app
from agents.workflowRefiner.workflow_refiner import workflow_refiner_app
from utils.build_code import create_file
from agents.codeAnnotator.code_annotator import code_annotator_app
from agents.softwareEngineer.software_engineer import software_engineer_app
from agents.promptEngineer.prompt_engineer import prompt_engineer_app
from agents.fileHandler.file_handler import file_handler_app

def main(user_request: str) -> None:
    input_refiner_response = input_refiner_app.invoke({
        ...
    })

    workflow_refiner_response = workflow_refiner_app.invoke({
        ...
    })

    create_file(workflow_refiner_response[...])

    code_annotator_response = code_annotator_app.invoke({
        ...
    })

    software_engineer_response = software_engineer_app.invoke({
        ...
    })

    prompt_engineer_response = prompt_engineer_app.invoke({
        ...
    })
    
    file_handler_response = file_handler_app.invoke({
        ...
    })
