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
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from dotenv import load_dotenv
from pathlib import Path
import traceback
import os

from typing import TypedDict, Literal, List, Optional, Annotated
from pydantic import BaseModel, Field
from operator import add

from agents.researcher.researcher import researcher_app
from agents.deepResearcher import prompts


''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'


print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Deep Researcher') if DEBUG else None



""" Schemas """
''' General Schemas '''
class QnA(BaseModel):
    question: str = Field(description= 'The question.')
    answer: str = Field(description= 'The answer.')

    def __str__(self):
        return f'Question: {self.question}\nAnswer: {self.answer}'

''' Input Schema '''
class InputSchema(TypedDict):
    research_topic: str = Field(description= 'The topic to be researched.')
    not_yet_researched_questions: Optional[List[str]] = Field(description= 'The question to be researched.')
    researched_questions_answers: Optional[Annotated[List[QnA], add]] = Field(
        description= 'A list of questions and answers.'
    )

''' Output Schema '''
class OutputSchema(BaseModel):
    answer: str = Field(description= 'The answer after researching.')



''' LLM '''
deep_researcher = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'moonshotai/kimi-k2:free', 
    temperature= 0.3
)

summariser = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'moonshotai/kimi-k2:free', 
    temperature= 0.5
)



''' Nodes'''
def breakdown_research_topic(state: InputSchema) -> InputSchema:
    print(f'\n{BLUE}[NODE]{RESET} deep_researcher/breakdown_research_topic') if DEBUG else None

    # Preprocessing
    if state.get('not_yet_researched_questions') == None: 
        state['not_yet_researched_questions'] = []
    if state.get('researched_questions_answers') == None: 
        state['researched_questions_answers'] = []

    try:
        # prompt
        qna = '\n---\n\n'.join([str(qna) for qna in state['researched_questions_answers']])

        prompt = prompts.BREAKDOWN_PROMPT.format(
            topic= state['research_topic'], 
            qna= qna
        )
        
        # call the LLM
        results = deep_researcher.invoke([SystemMessage(content= prompt)])
        # Fallback if the LLM does a mistake
        if results.content in ['~~~', 'empty', '<empty>']:
            results.content = ''

        print(f'{BLUE}[NODE] [INFO] [RESULTS]{RESET} {results}') if DEBUG else None

        questions = results.content.split('~~~') if results.content != '' else []
        state['not_yet_researched_questions'] = [question.strip() for question in questions]

        return state
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
def research_questions(state: InputSchema) -> InputSchema:
    print(f'\n{BLUE}[NODE]{RESET} deep_researcher/research_questions') if DEBUG else None

    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            config = lambda i: {'recursion_limit': 100, 'configurable': {'thread_id': f'researcher_{i}'}}
            futures = {
                executor.submit(
                    researcher_app.invoke, {'research_topic': research_topic}, config(i)
                ): research_topic 
                for i, research_topic in enumerate(state['not_yet_researched_questions'])
            }

            results: list[QnA] = []
            for future in concurrent.futures.as_completed(futures):
                response = future.result()
                result = QnA(question= response['research_topic'], answer= response['summary'])
                results.append(result)

                print(f'{BLUE}[NODE] [INFO] [QNA]{RESET} {result}') if DEBUG else None

        print(f'{BLUE}[NODE] [INFO] [FINISHED]{RESET}') if DEBUG else None

        state['not_yet_researched_questions'] = None
        state['researched_questions_answers'] = (state['researched_questions_answers'] or []) + results
        return state
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
def summarise(state: InputSchema) -> OutputSchema:
    '''
    This node summarises the results of the research, in order to return a strutured paragraph containing all the relevant information.
    '''
    print(f'\n{BLUE}[NODE]{RESET} deep_researcher/summarise') if DEBUG else None

    try:
        # prompt
        deep_research_findings = '\n---\n\n'.join([str(r) for r in state['researched_questions_answers']])

        prompt = prompts.SUMMARY_PROMPT.format(
            topic= state['research_topic'],
            deep_research_findings= deep_research_findings
        )

        # call the LLM
        summary = summariser.invoke(prompt).content

        print(f'{BLUE}[NODE] [INFO] [SUMMARY]{RESET} {summary}') if DEBUG else None

        return OutputSchema(answer= summary)

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return OutputSchema(answer= '\n\n'.join([qna.answer for qna in state['researched_questions_answers']]))



''' Conditional Functions '''
def should_summarise(state: InputSchema) -> Literal['summarise', 'research_questions']:
    print(f'\n{BLUE}[NODE]{RESET} deep_researcher/should_summarise') if DEBUG else None

    if not state['not_yet_researched_questions']:
        return 'summarise'
    
    # Limit imposed
    if len(state['researched_questions_answers']) >= 5:
        return 'summarise'
    
    return 'research_questions'


''' Graph '''
deep_researcher_graph = StateGraph(InputSchema, output_schema= OutputSchema)

deep_researcher_graph.add_node('breakdown_research_topic', breakdown_research_topic)
deep_researcher_graph.add_node('research_questions', research_questions)
deep_researcher_graph.add_node('summarise', summarise)

deep_researcher_graph.add_edge(START, 'breakdown_research_topic')
deep_researcher_graph.add_edge('research_questions', 'breakdown_research_topic')
deep_researcher_graph.add_conditional_edges(
    'breakdown_research_topic', 
    should_summarise, 
    {
        'summarise': 'summarise',
        'research_questions': 'research_questions'
    }
)
deep_researcher_graph.add_edge('summarise', END)

deep_researcher_app = deep_researcher_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(deep_researcher_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
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

    user = InputSchema(research_topic= 'what is the best place to live in Cyprus?')
    response = deep_researcher_app.invoke(user, config= config)

    import json
    print(f'{BLUE}[MAIN] [INFO]{RESET}', json.dumps(response, indent= 4)) if DEBUG else None
