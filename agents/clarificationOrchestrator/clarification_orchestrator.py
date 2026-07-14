"""
- `author:` Stefanos Panteli
- `date:` 2025-10-25
- `description:` This agent is used by the clarifying agents as a middle-man between the user and the clarifying agents.
                 Its role is to answer the questions raised by the clarifying agents and answer them if the user already answered something similar.

## How to use
1. Import the app. (`from agents.clarificationOrchestrator.clarificationOrchestrator import clarification_orchestrator_app`)
2. Input a dict with the following keys:
    - `question: str` # the question raised by the clarifying agent`
3. Invoke the app.
4. Get the output dict with the following keys:
    - qna: QnA # the question and answer
        - `question: str` # the question raised by the clarifying agent
        - `answer: str` # the answer provided by the user
        - `justification: str` # the reasoning behind the answer

## Usage
```python
from agents.clarificationOrchestrator.clarification_orchestrator import clarification_orchestrator_app
graph_input = {'question': 'Clarification: Do you mean [X]?'}

response = clarification_orchestrator_app.invoke(graph_input)

# response = {
#   qna: {
#       'question': 'Clarification: Do you mean [X]?', 
#       'answer': 'Yes', 
#       'justification': 'The answer is Yes, because the user answered [V] to [Y]'
#   }
# }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# Langgraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

# Schema imports
from pydantic_core._pydantic_core import ValidationError as PydanticValidationError
from typing import TypedDict, List, Optional, Annotated
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name
from agents.clarificationOrchestrator import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Clarification Orchestrator') if DEBUG else None



""" Schemas """
''' General Schemas '''
# A Schema to store the questions and answers up to now.
# Also used to structure the output of the LLM
class QnA(BaseModel):
    # The question and answer
    question: str = Field(description= 'The question you answered.')
    answer: str = Field(description= 'The answer you provided.')
    # The reasoning behind the answer so that the LLM does not hallucinate so much
    justification: str = Field(description= 'The reasoning behind the answer you provided.')

    def __str__(self):
        return f'- Question:\n{self.question}\n- Answer:\n{self.answer}'

class CoordinatorSchema(BaseModel):
    # The confidence score
    score: float = Field(description='The confidence score of the answer', ge=0, le=1)
    # The questions and answers
    qna: Optional[List[QnA]]
    # Possible unanswered questions
    unanswered_questions: Optional[List[str]]

    def __str__(self):
        answered_questions = (f'\nAnswered questions:\n' + '\n\n'.join([str(qna) + '\nJustification:\n' + qna.justification + '\n' for qna in self.qna])) if self.qna else ''
        unanswered_questions = (f'\nUnanswered questions:\n' + '\n'.join([str(question) for question in self.unanswered_questions])) if self.unanswered_questions else ''
        return f'Score: {self.score}\n{answered_questions}\n{unanswered_questions}'

''' Input Schema '''
class InputSchema(BaseModel):
    # The question
    question: str
    # The questions and answers up to now (memory)
    questions_answers: Annotated[List[QnA], add] = Field(default_factory= list)

''' Output Schema '''
class OutputSchema(TypedDict):
    # The user input
    qna: QnA # The question and answer.



''' LLM '''
coordinator = myChatOpenAI(
    temperature= 0
).with_structured_output(CoordinatorSchema)



''' Nodes '''
# The LLM is called to answer the question, if not enough information is provided or the score is low, the user is asked to provide answers
def answer_question(state: InputSchema) -> InputSchema:
    '''
    This node answers the question provided by the schema `InputSchema`
    '''
    print_function_name() if DEBUG else None

    try:
        # prompt
        prompt = prompts.ANSWER_QUESTION_PROMPT.format(question= state.question.split('# RESOLVED')[0])

        # memory as messages
        memory: List[AIMessage, HumanMessage] = []
        for qna in state.questions_answers:
            memory.append(AIMessage(content= qna.question.split('# RESOLVED')[0]))
            memory.append(HumanMessage(content= {qna.answer}))

        # call the LLM
        response: CoordinatorSchema = safe_invoke(
            coordinator, 
            messages= [SystemMessage(content= prompt), *memory, AIMessage(content= state.question.split('# RESOLVED')[0])], 
            raise_pydantic= True
        )
        print(f'{BLUE}[NODE] [LLM RESPONSE]{RESET} {response}') if DEBUG else None

        # If the score is low, ask the user for input
        if response.score <= 0.8:
            print(f'{RED}[NODE] [INFO]{RESET} Low Score') if DEBUG else None
            print(f'{GREEN}[NODE] [QUESTION]{RESET} Please answer the following question:')
            user_answer = input(state.question + '\n\n > ')
            question = state.question
            answer = user_answer
            justifications = 'User provided answer'

        # If the score is high, use the answer
        else:
            question = '\n'.join([qna.question for qna in response.qna])
            answer = '\n'.join([qna.answer for qna in response.qna])
            justifications = '\n'.join([qna.justification for qna in response.qna])

            # If there are unanswered questions, ask the user
            if response.unanswered_questions:
                print(f'{RED}[NODE] [INFO]{RESET} Unanswered Questions') if DEBUG else None
                q = '\n'.join(response.unanswered_questions)
                print(f'{GREEN}[NODE] [QUESTION]{RESET} Please answer the following question:')
                # Ask the user
                user_answer = input(q + '\n\n > ')
                question += '\n' + q
                answer += '\n' + user_answer
                justifications += f' and User provided answer for the latest questions.'
        
        # Append the question and answer to state
        return {'questions_answers': [QnA(question= question, answer= answer, justification= justifications)]}
    
    # If error, ask the user
    except Exception as e:
        print(f"{RED}[NODE] [ERR]{RESET}", e) if DEBUG else None
        # Ask the user
        user_answer = input(state.question + '\n\n > ')
        # Append the question and answer to state
        return {'questions_answers': [QnA(question= state.question, answer= user_answer, justification= 'User provided the answer.')]}

# Return the last question and answer
def return_answer(state: InputSchema) -> OutputSchema:
    '''
    This node returns the last question and answer.
    '''
    return OutputSchema(qna= state.questions_answers[-1])


''' Graph '''
clarification_orchestrator_graph = StateGraph(InputSchema, output_schema= OutputSchema)

clarification_orchestrator_graph.add_node('answer_question', answer_question)
clarification_orchestrator_graph.add_node('return_answer', return_answer)

clarification_orchestrator_graph.add_edge(START, 'answer_question')
clarification_orchestrator_graph.add_edge('answer_question', 'return_answer')
clarification_orchestrator_graph.add_edge('return_answer', END)

clarification_orchestrator_app = clarification_orchestrator_graph.compile(checkpointer= MemorySaver())