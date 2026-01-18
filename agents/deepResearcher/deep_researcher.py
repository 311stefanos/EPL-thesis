"""
- `author:` Stefanos Panteli
- `date:` 2025-09-10
- `description:` This is the deep researcher agent. It gets a simple input and provides relevant 
information. It uses the researcher agent to get relevant information from the internet.

## How to use
1. Import the app. (`from agents.deepResearcher.deepResearcher import deep_researcher_app`)
2. Input a dict with the following keys:
    - `research_topic: str`: The topic to research.
3. Invoke the app.
4. Get the output dict with the following keys:
    - `answer: str`: The answer after deep research.

## Usage
```python
from agents.deepResearcher.deepResearcher import deep_researcher_app
graph_input = InputSchema(research_topic= 'what is the best place to live in Cyprus?')

response = deep_researcher_app.invoke(graph_input)

# response = {
#     answer: 'Paphos is the clear front-runner for anyone looking for 
#              the single best place to live in Cyprus: it marries the 
#              island's lowest crime rate and cheapest cost of living 
#              (about 35 % below Limassol) with the largest, longest-settled 
#              expat community, so English is spoken everywhere and clubs, 
#              healthcare and bureaucracy are newcomer-friendly. A family can 
#              live well on roughly €2 600 a month, renting a modern one- or 
#              three-bed flat for €900-€1 350, while still enjoying a picturesque 
#              harbour, Blue-Flag beaches, mountain trails and Unesco-listed ruins 
#              all within a 15-minute drive. Limassol has more corporate jobs and 
#              nightlife, Larnaca gives the best food-per-euro ratio, and Nicosia 
#              supplies capital-city culture (but no sea); younger party-seekers 
#              gravitate to Ayia Napa, while families after quieter sand head to 
#              Protaras/Fig Tree Bay. Inland, stone villages such as Lefkara and 
#              Omodos in the Troodos wine region cut monthly costs to €300-€600 
#              rent and €1 000-€1 800 total living expenses, yet retain tight-knit 
#              communities, near-zero crime and fast road links to the coast. 
#              Overall, Paphos offers the widest, easiest blend of affordability, 
#              safety, scenery and ready-made expat life, making it the top all-round 
#              choice for most newcomers.'
# }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage

# Langgraph imports
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

# Schema imports
from typing import TypedDict, Literal, List, Optional
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import os
    # Thread imports
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# My imports
from agents.researcher.researcher import researcher_app
from utils.utils import myChatOpenAI, safe_invoke, print_function_name
from agents.deepResearcher import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Deep Researcher') if DEBUG else None



""" Schemas """
''' General Schemas '''
# A Schema for the results of a research query
class QnA(BaseModel):
    question: str = Field(description= 'The question.')
    answer: str = Field(description= 'The answer.')

    def __str__(self):
        return f'Question: {self.question}\nAnswer: {self.answer}'
    
# Schema for the LLM to structure its output
class StructuredOutput(BaseModel):
    questions_to_research: List[str] = Field(description= 'The questions to be researched.')

''' Input Schema '''
# The input schema
class InputSchema(TypedDict):
    research_topic: str = Field(description= 'The topic to be researched.')
    not_yet_researched_questions: Optional[List[str]] = Field(description= 'The question to be researched.', default_factory= List)
    researched_questions_answers: Optional[List[QnA]] = Field(
        description= 'A list of questions and answers.',
        default_factory= List
    )
    # If there was an error during the breakdown logic
    error_occurred: bool = Field(description= 'If there was an error during the breakdown logic.')

''' Output Schema '''
# The output schema, just the final summary
class OutputSchema(BaseModel):
    answer: str = Field(description= 'The answer after researching.')



''' LLM '''
# The LLM that breaks down the topic
deep_researcher = myChatOpenAI(
    temperature= 0.6
).with_structured_output(StructuredOutput)

# The LLM that summarises the results
summariser = myChatOpenAI(
    temperature= 0.3
)



''' Nodes '''
# The node that breaks down the topic
def breakdown_research_topic(state: InputSchema) -> InputSchema:
    '''
    This node breaks down the topic into questions for the researcher team to answer.
    '''
    print_function_name() if DEBUG else None
    state['error_occurred'] = False

    try:
        # prompt
        qna = '\n---\n\n'.join([str(qna) for qna in state['researched_questions_answers']]) if state.get('researched_questions_answers') else ''

        prompt = prompts.BREAKDOWN_PROMPT.format(
            topic= state['research_topic'], 
            qna= qna
        )
        
        # call the LLM
        results: StructuredOutput = safe_invoke(deep_researcher, [SystemMessage(content= prompt)])

        print(f'{BLUE}[NODE] [INFO] [RESULTS]{RESET} {results}') if DEBUG else None

        # Get the questions from the LLM response
        state['not_yet_researched_questions'] = [question.strip() for question in results.questions_to_research]
        
        return state
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        state['error_occurred'] = True
        return state
    
# The node that researches all the questions
def research_questions(state: InputSchema) -> InputSchema:
    '''
    This node researches all the questions in parallel, using the researcher team.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # Start a multi threaded call to each researcher. Let the max workers be 5, even tho the LLM is instructed to output <= 3 sub topics.
        with ThreadPoolExecutor(max_workers= 5) as executor:
            # Just a different thread_id for each agent
            config = lambda i: {'recursion_limit': 100, 'configurable': {'thread_id': f'researcher_{i}'}}
            futures = {
                executor.submit(
                    # Call the researcher with the topic as input, and the configuration
                    researcher_app.invoke, {'research_topic': research_topic}, config(i)
                ): research_topic # With the value being the topic
                for i, research_topic in enumerate(state['not_yet_researched_questions'])
            }

            results: list[QnA] = []
            # For each of the researcher's outputs, as they complete
            for future in concurrent.futures.as_completed(futures):
                # Get the result, parse it and add it to the results
                response = future.result()
                result_qna = QnA(question= response['research_topic'], answer= response['summary'])
                results.append(result_qna)

                print(f'{BLUE}[NODE] [INFO] [QNA]{RESET} {result_qna}') if DEBUG else None

        print(f'{BLUE}[NODE] [INFO] [FINISHED]{RESET}') if DEBUG else None

        # Clear the not yet researched questions
        state['not_yet_researched_questions'] = []
        # Add the results
        state['researched_questions_answers'] = state.get('researched_questions_answers', []) + results
        return state
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
# Serial
# The node that researches all the questions
def research_questions_serial(state: InputSchema) -> InputSchema:
    '''
    This node researches all the questions in parallel, using the researcher team.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # Just a different thread_id for each agent
        researched_questions_answers = []
        config = lambda i: {'recursion_limit': 100, 'configurable': {'thread_id': f'researcher_{i}'}}
        for i, research_topic in enumerate(state['not_yet_researched_questions']):
            # Call the researcher with the topic as input, and the configuration
            response = researcher_app.invoke({'research_topic': research_topic}, config(i))

            # Get the result, parse it and add it to the results
            result_qna = QnA(question= response['research_topic'], answer= response['summary'])
            researched_questions_answers.append(result_qna)

            print(f'{BLUE}[NODE] [INFO] [QNA]{RESET} {result_qna}') if DEBUG else None

        print(f'{BLUE}[NODE] [INFO] [FINISHED]{RESET}') if DEBUG else None

        # Clear the not yet researched questions
        state['not_yet_researched_questions'] = []
        state['researched_questions_answers'] = state.get('researched_questions_answers', []) + researched_questions_answers
        
        return state
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state

# The node that summarises the results
def summarise(state: InputSchema) -> OutputSchema:
    '''
    This node summarises the results of the research, in order to return a strutured paragraph containing all the relevant information.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # prompt
        deep_research_findings = '\n---\n\n'.join([str(r) for r in state['researched_questions_answers']])

        prompt = prompts.SUMMARY_PROMPT.format(
            topic= state['research_topic'],
            deep_research_findings= deep_research_findings
        )

        # call the LLM
        summary = safe_invoke(summariser, [SystemMessage(content= prompt)]).content

        print(f'{BLUE}[NODE] [INFO] [SUMMARY]{RESET} {summary}') if DEBUG else None

        return OutputSchema(answer= summary)

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return OutputSchema(answer= '\n\n'.join([qna.answer for qna in state['researched_questions_answers']]))



''' Conditional Functions '''
def should_summarise(state: InputSchema) -> Literal['breakdown_research_topic', 'summarise', 'research_questions']:
    '''
    The conditional logic that determines what to do after breaking down the topic: research the topics or summarise them.
    '''
    print_function_name() if DEBUG else None

    # Error occured, go back
    if state.get('error_occurred', False):
        return 'breakdown_research_topic'

    # No more questions to research, summarise
    if not state['not_yet_researched_questions']:
        return 'summarise'
    
    # Limit imposed
    if len(state.get('researched_questions_answers', [])) >= 5:
        return 'summarise'
    
    return 'research_questions'



''' Graph '''
deep_researcher_graph = StateGraph(InputSchema, output_schema= OutputSchema)

deep_researcher_graph.add_node('breakdown_research_topic', breakdown_research_topic)
deep_researcher_graph.add_node('research_questions', research_questions)
# deep_researcher_graph.add_node('research_questions', research_questions_serial)
deep_researcher_graph.add_node('summarise', summarise)

deep_researcher_graph.add_edge(START, 'breakdown_research_topic')
deep_researcher_graph.add_edge('research_questions', 'breakdown_research_topic')
deep_researcher_graph.add_conditional_edges(
    'breakdown_research_topic', 
    should_summarise, 
    {   # Not needed, just for clarity
        'breakdown_research_topic': 'breakdown_research_topic',
        'summarise': 'summarise',
        'research_questions': 'research_questions'
    }
)
deep_researcher_graph.add_edge('summarise', END)

deep_researcher_app = deep_researcher_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(deep_researcher_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/deep_researcher_app.png', 'wb') as f:
        f.write(deep_researcher_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'deepResearcher'
    os.environ['LANGSMITH_PROJECT'] = 'deepResearcher'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'deepResearcher',
            'run_name': 'deepResearcher',
            'thread_id': 'deepResearcher'
        }
    }

    user = InputSchema(research_topic= 'Give me details about all events and parties in Cyprus for the netx 3 months.')
    response = deep_researcher_app.invoke(user, config= config)
    
    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
