"""
- `author:` Stefanos Panteli
- `date:` 2025-08-05
- `description:` Gets a user input and decomposes it into tasks, which can then be executed by other agents.

## How to use
1. Import the app. (`from agents.taskDecomposer.task_decomposer import task_decomposer_app`)
2. Input a dict with the following keys:
    - `user_input: str`: The user input to be decomposed.
3. Invoke the app.
4. Get the output dict with the following keys:
    - `user_input: str`: The user input, may include web findings.
    - `refined_input: str`: The refined user input.
    - `output: Optional[TaskPlan]`: The decomposed task plan.
        - `type: Literal['single-agent', 'multi-agent']`: Whether the task requires a single agent or multiple agents.
        - `justification: str`: Justification for the agent's type assignment.
        - `agents: List[SingleAgentPlan]`: List of agents involved in the task.
            - `role: str`: Role of the agent.
            - `scope: str`: Scope of the agent's responsibility.
    - `error: Optional[str]`: The error, if any.
    - `should_stop: bool`: The flag to stop the task decomposition.
    - `number_of_iterations: int`: The number of iterations.

## Usage
```python
from agents.taskDecomposer.task_decomposer import task_decomposer_app
graph_input = {'user_input': 'I want a personall fitness coach.'}

response = task_decomposer_app.invoke(graph_input)

# response = {
#     'user_input': 'User input: I want a personal fitness coach.\n\n \
#                    Refined input: I am looking to hire a personal fitness trainer for customized workout guidance. \
#                    \n\n---\n\n \
#                    #### Web Findings (query: "I am looking to hire a personal fitness trainer for customized workout guidance."):\n \
#                    Score is 0-1, with 1 most relevant.\n \
#                    0. Score: 0.8095324 - Virtual real person trainer for custom workout plan?\n \
#                       You can hire a coach through...',
#     'refined_input': 'I am looking to hire a personal fitness trainer for customized workout guidance.',
#     'output': {
#         'type': 'multi-agent',
#         'justification': 'While the core need is fitness training, optimal results require complementary nutrition guidance and progress tracking capabilities.',
#         'agents': [
#             {
#                 'role': 'Fitness Coach',
#                 'scope': 'Provide customized workout plans and guidance'
#             },
#             {
#                 'role': 'Nutrition Advisor',
#                 'scope': 'Offer dietary recommendations to complement fitness goals'
#             },
#             {
#                 'role': 'Progress Tracker',
#                 'scope': 'Monitor and analyze fitness metrics and improvements'
#             }
#         ],
#     },
#     'should_stop': True,
#     'number_of_iterations': 2
# }
```
"""



''' Imports '''
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph
from langgraph.constants import END, START

import os
import json
import traceback
from pathlib import Path
from dotenv import load_dotenv

from typing import Literal, List, Optional
from pydantic import BaseModel, Field

from agents.taskDecomposer import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DEBUG = os.getenv('DEBUG')

print('\n[AGENT] [INFO] [STARTUP] Task Decomposer') if DEBUG else None



""" Schemas """
''' General Schemas '''
class SingleAgentPlan(BaseModel):
    role: str = Field(description= 'Role of the agent.')
    scope: str = Field(description= 'Scope of the agent\'s responsibility.')

    def __eq__(self, other: 'SingleAgentPlan') -> bool:
        return self.role == other.role and self.scope == other.scope

class TaskPlan(BaseModel):
    type: Literal['single-agent', 'multi-agent'] = Field(description= 'Whether the task requires a single agent or multiple agents.')
    justification: str = Field(description= 'Justification for the agent\'s type assignment.')
    agents: List[SingleAgentPlan] = Field(description= 'List of agents involved in the task.')

    def __eq__(self, other: 'TaskPlan') -> bool:
        return (
            self.type == other.type and 
            len(self.agents) == len(other.agents) and 
            all(agent == other_agent for agent, other_agent in zip(self.agents, other.agents))
        )

''' Input/Output Schema '''
class TaskDecomposerState(BaseModel):
    user_input: str
    refined_input: str = ''
    output: Optional[TaskPlan] = None
    error: Optional[str] = None
    should_stop: bool = False
    number_of_iterations: int = 0
    
    def increase(self) -> None:
        self.number_of_iterations += 1

    def stop(self) -> bool:
        print('[TASK_DECOMPOSER_STATE] stop') if DEBUG else None
        self.should_stop = True



''' Tools '''
tavily_search_tool = TavilySearch(
    tavily_api_key= TAVILY_API_KEY
)



''' LLM '''
decomposer = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'deepseek/deepseek-chat-v3-0324:free', 
    temperature=0
).with_structured_output(TaskPlan)



''' Nodes'''
def tavily_search(state: TaskDecomposerState) -> TaskDecomposerState:
    '''
    Use Tavily to search the web for relevant information about the topic.
    '''
    print('\n[NODE] tavily_search') if DEBUG else None
    try:
        # Run the search using the refined input or the original user input
        search_query = state.refined_input or state.user_input
        result = tavily_search_tool.invoke(search_query)

        # Parse the output to get a summary of the web findings
        summary_lines = []
        for i, entry in enumerate(result.get('results', []), 1):
            title = entry.get('title', 'Untitled')
            content = entry.get('content', '')[:500].strip()
            score = entry.get('score', 'No score found')
            if score < 0.35: # Skip irrelevant results
                continue

            summary_lines.append(f'\n{i}. Score: {score} - {title}\n  {content}\n')

        web_summary = '\n---\n'.join(summary_lines) if summary_lines else 'No relevant web results found.'

        # Format the new input with web findings appended
        enriched_input = (
            f'User input: {state.user_input.strip()}\n\n'
            f'Refined input: {state.refined_input.strip()}\n\n'
            f'---\n\n'
            f'#### Web Findings (query: "{search_query}"):\n'
            f'Score is 0-1, with 1 most relevant.\n{web_summary}\n'
        )

        print(f'[NODE] [INFO] Enriched input: {enriched_input}') if DEBUG else None

        return state.model_copy(update={'user_input': enriched_input})
    
    except Exception as e:
        print('[NODE] [ERR]', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state

def decompose_prompt(state: TaskDecomposerState) -> TaskDecomposerState:
    '''
    This node decomposes the task into subtasks, assigned to agents.
    '''
    print('\n[NODE] decompose_prompt') if DEBUG else None

    try:
        state.increase()

        # format the prompt
        agent_ideas = (
            json.dumps(state.output.model_dump(), indent=4)
            if state.output else '{}'
        )

        prompt = prompts.TASK_DECOMPOSER_PROMPT.format(
            agent_ideas= agent_ideas,
            user_input= state.user_input
        )

        # call the LLM, and ensure it returns a TaskPlan
        response = decomposer.invoke(
            [SystemMessage(content= prompt)]
        )

        # update the state
        if state.output and state.output == response:
            state.stop()

        print(f'[NODE] [INFO] Previous output: {state.output.model_dump_json()}') if DEBUG and state.output else None
        print(f'[NODE] [INFO] New output: {response.model_dump_json()}') if DEBUG else None

        return state.model_copy(update={'output': response})
    except Exception as e:
        # update the state, if an error occurs
        print('[NODE] [ERR]', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state.model_copy(update={'error': str(e)})



''' Conditional Functions '''
def should_continue(state: TaskDecomposerState) -> Literal['decompose_prompt', 'end']:
    '''
    This node checks if the task decomposition should continue.
    '''
    print('\n[CONDITION] should_continue') if DEBUG else None
    return 'end' if state.should_stop or state.number_of_iterations > 3 else 'retry'



''' Graph '''
task_decomposer_graph = StateGraph(TaskDecomposerState)

task_decomposer_graph.add_node('tavily_search', tavily_search)
task_decomposer_graph.add_node('decompose_prompt', decompose_prompt)

task_decomposer_graph.add_edge(START, 'tavily_search')
task_decomposer_graph.add_edge('tavily_search', 'decompose_prompt')
task_decomposer_graph.add_conditional_edges(
    'decompose_prompt', 
    should_continue,
    {
        'retry': 'decompose_prompt',
        'end': END
    }
)

task_decomposer_app = task_decomposer_graph.compile()



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(task_decomposer_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/task_decomposer_app.png', 'wb') as f:
        f.write(task_decomposer_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'TaskDecomposer'
    os.environ['LANGSMITH_PROJECT'] = 'TaskDecomposer'
    client = Client()

    config = {
        'configurable': {
            'user_id': 'Test-TaskDecomposer',
            'run_name': 'Test-TaskDecomposer',
            'tags': ['TaskDecomposer', 'deepseek'],
        }
    }

    # Run the graph
    # 'I want a personall fitness coach.'
    # 'I want an agent that knows about dolphins.'
    # 'I want an agent to help me, guide me and test me on my math course.'
    # 'I want an agent that thinks and creates tools for LLM agents to use, for specific topics, which will be provided by the user.'
    # 'I want an agent that helps me about the most difficult topic today (1).'
    user = TaskDecomposerState(user_input= 'I want an agent that accepts a user input, then first corrects it, and after understanding it thoroughly (can ask questions), refines it into a well structured prompt for llms to use.')
    result = task_decomposer_app.invoke(user, config)

    print('\n[MAIN]', result)

    print('\n[MAIN]', result['number_of_iterations'])

    print('\n[MAIN]', result['output'].model_dump_json(indent=2))

