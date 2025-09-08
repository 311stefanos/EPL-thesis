"""
- `author:` Stefanos Panteli
- `date:` 2025-09-04
- `description:` A research agent that gets a simple input and provides relevant information from the internet or knowledge base.

## How to use
1. Import the app. (`from agents.userInputRefiner.researcher import researcher_app`)
2. Input a dict with the following keys:
    - `research_topic: str`: The topic to research.
    - `research_queries: Optional[List[str]]`: The agent's queries made to find the appropriate results, so far. (Leave Empty [] or None)
    - `results: Optional[List[ResearchResult]]`: The agent's results of the research. (Leave Empty [] or None)
3. Invoke the app.
4. Get the output dict with the following keys:
    - `research_topic: str`: The topic to research. (Same as input)
    - `summary: str`: The agent's summary of the results.

## Usage
```python
from agents.userInputRefiner.researcher import researcher_app
graph_input = InputSchema(research_topic= 'Tell me the admission deadlines for cs in keio to start my masters in fall 2026')

response = researcher_app.invoke(graph_input)

# response = {
#     research_topic: 'Tell me the admission deadlines for cs in keio to start my masters in fall 2026',
#     summary: '' TODO: Add summary
# }
```
"""



''' Imports '''
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
import traceback
import os

from typing import Literal, Optional, List
from pydantic import BaseModel, Field

from agents.researcher import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Researcher') if DEBUG else None



""" Schemas """
''' General Schemas '''
# A Schema for the results of a research query
class ResearchResult(BaseModel):
    ''' You should use this schema to represent the results of a research query. Should be used after each search. '''
    research_query: str = Field(description= 'The query made to find the appropriate results.')
    url: List[Optional[str]] = Field(description= 'The url of the document(s) used in the information extraction.')
    title: List[Optional[str]] = Field(description= 'The title of the document(s) used in the information extraction.')
    date_created: List[Optional[str]] = Field(description= 'The date of the document.')
    author: List[Optional[str]] = Field(description= 'The author of the document.')
    relevant_information: str = Field(description= 'The relevant information extracted from the document(s).')

''' Input Schema '''
# The input schema
class InputSchema(MessagesState):
    research_topic: str = Field(
        description='The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).'
    )
    research_queries: Optional[List[str]] = Field(description= 'The queries made to find the appropriate results, so far.')
    results: Optional[List[ResearchResult]] = Field(description= 'The results of the research.')


''' Output Schema '''
# The output schema
class OutputSchema(BaseModel):
    research_topic: str = Field(
        description='The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).'
    )
    summary: str = Field(description= 'The results of the research in a beautifully written summary.')


''' Tools '''
# The think tool, is for strategic reflection pf the agent
@tool(description= 'Strategic reflection tool for research planning')
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f'Reflection recorded: {reflection}'

# Tavily, to search and gather information from the web
tavily_search = TavilySearch(
    tavily_api_key= TAVILY_API_KEY,
    search_depth= "advanced",
    max_results= 5,
    include_answer= True
).as_tool()

# Wikipedia, to search and gather information from the web
wikipedia_search = WikipediaQueryRun(
    api_wrapper= WikipediaAPIWrapper(top_k_results= 3, doc_content_chars_max= 500)
).as_tool()

# DDG search as well
@tool(description= 'Search the web with duckduckgo for information on a topic.')
def duckduckgo_search(query: str) -> str:
    '''
    Search the web with duckduckgo for information on a topic.
    '''
    search = DuckDuckGoSearchResults()
    return search.run(query)

# Research Result as a tool, in order to be filled from the agent
reseach_result = tool(ResearchResult, description= 'Represent the results of a research query. Always use this after using web search tools.')

# List of tools
tools = [think_tool, tavily_search, wikipedia_search, duckduckgo_search, reseach_result]
# Dictionary of tools: tool name -> tool
tools_by_name = {tool.name: tool for tool in tools}


''' LLM '''
researcher = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'moonshotai/kimi-k2:free', 
    temperature= 0.7
).bind_tools(tools)

summariser = ChatOpenAI(
    base_url= 'https://openrouter.ai/api/v1', 
    api_key= OPENROUTER_API_KEY,
    model= 'moonshotai/kimi-k2:free', 
    temperature= 0.5
)



''' Helpful Functions '''
# Get today's date in a human-readable format
def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"



''' Nodes'''
# The do_research node, where the agent does the actual research by calling the tools
def do_research(state: InputSchema) -> InputSchema:
    print(f'\n{BLUE}[NODE]{RESET} do_research') if DEBUG else None
    # Initialize state
    if state['research_queries'] == None: 
        state['research_queries'] = []
    if state['results'] == None: 
        state['results'] = []

    try:
        # prompt
        prompt = prompts.RESEARCH_PROMPT.format(
            date= get_today_str(), 
            topic= state['research_topic']
        )
        # call the LLM
        results = researcher.invoke([SystemMessage(content= prompt)] + state['messages'])

        print(f'{BLUE}[NODE] [INFO] [RESULTS]{RESET} {results}') if DEBUG else None

        return {'messages': [results]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
# The tool_node node, where the agent uses the tools
def tool_node(state: InputSchema) -> InputSchema:
    print(f'\n{BLUE}[NODE]{RESET} tool_node') if DEBUG else None

    try:
        # Get the last message, and extract the tool calls
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', [])

        # Execute all tool calls
        observations = []
        for tool_call in tool_calls:
            # Get the tool and arguments
            tool = tools_by_name[tool_call["name"]]
            args = tool_call["args"]

            print(f'{BLUE}[NODE] [INFO] [TOOL CALL]{RESET} {tool_call["name"]} with {args}') if DEBUG else None

            try:
                observation = None
                # Execute the tool
                observation = tool.invoke(args)
                # Add the observation to the list
                if observation:
                    observations.append(observation)
            except Exception as e:
                print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
                traceback.print_exc() if DEBUG else None
                # If the tool fails, skip it
                continue

            print(f'{BLUE}[NODE] [INFO] [LAST OBSERVATION]{RESET} {observation}') if DEBUG else None

            # If the tool is a web search, add the query to the list
            if tool in [tavily_search, wikipedia_search, duckduckgo_search]:
                state['research_queries'].append(args['query'])

            # If the tool is a research result, add the result to the list
            elif tool in [ResearchResult]:
                state['results'].append(observation)

        # Create a list of tool outputs, as ToolMessage
        tool_outputs = [
            ToolMessage(
                content= observation,
                name= tool_call['name'],
                tool_call_id= tool_call['id']
            ) for observation, tool_call in zip(observations, tool_calls)
        ]
        
        # Add them to the state
        return {'messages': tool_outputs}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
# The summarise node, where the LLM summarises the results of the research
def summarise(state: InputSchema) -> OutputSchema:
    print(f'\n{BLUE}[NODE]{RESET} summarise') if DEBUG else None

    try:
        # prompt
        results = []
        # format the results
        for result in state['results']:
            formated_result = (
                f'Query: {result.research_query}\n'
                f'URLs: {result.url}\n'
                f'Titles: {result.title}\n'
                f'Dates Created: {result.date_created}\n'
                f'Authors: {result.author}\n\n'
                f'Extracted Relevant Information: {result.relevant_information}\n'
            )
            results.append(formated_result)

        prompt = prompts.SUMMARY_PROMPT.format(
            date= get_today_str(),
            topic= state['research_topic'], 
            web_findings= '\n---\n'.join(results),
            history= ''.join([message.pretty_repr() for message in state['messages']])
        )

        # call the LLM
        summary = summariser.invoke(prompt)

        print(f'{BLUE}[NODE] [INFO] [SUMMARY]{RESET} {summary}') if DEBUG else None

        return OutputSchema(research_topic= state['research_topic'], summary= summary.content)

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return OutputSchema(research_topic= state['research_topic'], summary= '')



''' Conditional Functions '''
# The should_continue function that checks if the agent should continue
def should_continue(state: InputSchema) -> Literal['tool_node', 'summarise']:
    print(f'\n{BLUE}[NODE]{RESET} should_continue') if DEBUG else None

    try:
        # Get the last message and extract the tool calls
        last_message = state['messages'][-1]
        tool_calls = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', [])

        # If there are tool calls, go to the tool node
        if tool_calls:
            return 'tool_node'
        # Else, go to the summarise node
        else:
            print(f'{BLUE}[NODE] [INFO] [NO TOOL CALLS]{RESET} Go to summarise') if DEBUG else None
            return 'summarise'
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return False



''' Graph '''
researcher_graph = StateGraph(InputSchema, output_schema= OutputSchema)

researcher_graph.add_node('do_research', do_research)
researcher_graph.add_node('tool_node', tool_node)
researcher_graph.add_node('summarise', summarise)

researcher_graph.add_edge(START, 'do_research')
researcher_graph.add_conditional_edges(
    'do_research',
    should_continue,
    {   # Not needed, just for clarity
        'tool_node': 'tool_node',
        'summarise': 'summarise'
    }
)
researcher_graph.add_edge('tool_node', 'do_research')
researcher_graph.add_edge('summarise', END)

researcher_app = researcher_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image

    # Visualize the graph
    Image(researcher_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/researcher_app.png', 'wb') as f:
        f.write(researcher_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'researcher'
    os.environ['LANGSMITH_PROJECT'] = 'researcher'
    client = Client()

    config = {
        'configurable': {
            'user_id': 'researcher',
            'run_name': 'researcher',
            'thread_id': 'researcher', 
            'recursion_limit': 50
        }
    }

    user = InputSchema(
        research_topic= 'Tell me the admission deadlines for cs in keio to start my masters in fall 2026'
    )
    response = researcher_app.invoke(user, config= config)

    import json
    print(f'{BLUE}[MAIN] [INFO]{RESET}', json.dumps(response, indent= 4)) if DEBUG else None
