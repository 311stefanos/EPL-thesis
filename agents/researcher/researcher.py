"""
- `author:` Stefanos Panteli
- `date:` 2025-09-04
- `description:` A research agent that gets a simple input and provides relevant information from the internet or knowledge base.

## How to use
1. Import the app. (`from agents.researcher.researcher import researcher_app`)
2. Input a dict with the following keys:
    - `research_topic: str`: The topic to research.
    - `results: Optional[List[ResearchResult]]`: The agent's results of the research. (Leave Empty [] or None)
3. Invoke the app.
4. Get the output dict with the following keys:
    - `research_topic: str`: The topic to research. (Same as input)
    - `summary: str`: The agent's summary of the results.

## Usage
```python
from agents.researcher.researcher import researcher_app
graph_input = InputSchema(research_topic= 'Visa and residency requirements for international students pursuing master's CS in Tokyo 
#                                          (student visa, work permits, post-study work visa, application process, duration, restrictions)')

response = researcher_app.invoke(graph_input)

# response = {
#     research_topic: 'Visa and residency requirements for international students pursuing master's CS in Tokyo 
#                      (student visa, work permits, post-study work visa, application process, duration, restrictions)',
#     summary: 'International students admitted to a two-year master's in Computer Science in Tokyo first secure a “Student” 
#               visa from their home country, then apply for a “Designated Activities” visa through a Certificate of Eligibility 
#               that their university—not the student—files with Japan's Immigration Services Agency; once the COE is issued they 
#               complete the visa at a Japanese embassy abroad, normally receiving permission to stay for up to 4 years 3 months.
#               After arrival they must register their address within 14 days, carry the resulting Residence Card at all times, 
#               and, if they wish to work part-time, obtain an extra “permission to engage in other activities” stamp that caps 
#               employment at 28 h per week during term and 40 h per week in long university breaks, with adult-entertainment jobs 
#               strictly forbidden; breaching these limits risks deportation and a five-year entry ban.  Near graduation they have 
#               three main onward routes: a Designated Activities visa (6 months, one renewal possible with a university recommendation) 
#               to continue job-hunting, an immediate work visa (1, 3 or 5 years, employer-sponsored) if they secure a relevant position, 
#               or—if they meet business-setup criteria—up to two years under Designated Activities for entrepreneurship. Visa extensions 
#               must be requested three months before expiry, and the student status (and its work permission) terminates the moment 
#               the studies end or the holder leaves the country.'
# }
```
"""



''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, ToolMessage, RemoveMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START

# Schema imports
from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field

# General imports
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from time import sleep
import traceback
import json
import os
import re

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, parse_tool_arguments
from agents.researcher import prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DEBUG = os.getenv('DEBUG')
MODEL_NAME = os.getenv('MODEL_NAME')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
MAGENTA = '\033[95m' # TOOLS
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} Researcher') if DEBUG else None



""" Schemas """
''' General Schemas '''
# A Schema for the results of a research query
class ResearchResult(BaseModel):
    ''' You should use this schema to represent the results of a research query. Should be used after each search. '''
    research_query: Union[str, List[str]] = Field(description= 'The query(s) made to find the appropriate results.')
    url: List[Optional[str]] = Field(description= 'The url of the document(s) used in the information extraction.')
    title: List[Optional[str]] = Field(description= 'The title of the document(s) used in the information extraction.')
    date_created: List[Optional[str]] = Field(description= 'The date of the document.')
    author: List[Optional[str]] = Field(description= 'The author of the document.')
    relevant_information: str = Field(description= 'The relevant information extracted from the document(s).')

    def __str__(self):
        return (
            f'Query: {self.research_query}\n'
            f'URLs: {self.url}\n'
            f'Titles: {self.title}\n'
            f'Dates Created: {self.date_created}\n'
            f'Authors: {self.author}\n\n'
            f'Extracted Relevant Information: {self.relevant_information}\n'
        )

''' Input Schema '''
# The input schema
class InputSchema(MessagesState):
    research_topic: str = Field(
        description='The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).'
    )
    results: Optional[List[ResearchResult]] = Field(description= 'The results of the research.', default_factory= List)
    # If there was an error during the research
    error_occurred: bool = Field(description= 'If there was an error during the research.', default= False)


''' Output Schema '''
# The output schema
class OutputSchema(BaseModel):
    research_topic: str = Field(
        description='The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).'
    )
    summary: str = Field(description= 'The results of the research in a beautifully written summary.')



''' Tools '''
# The think tool, is for strategic reflection of the agent
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
    print_function_name(colour= MAGENTA) if DEBUG else None
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
    print_function_name(colour= MAGENTA) if DEBUG else None
    search = DuckDuckGoSearchResults()
    return search.run(query)

# Research Result as a tool, in order to be filled from the agent
research_result = tool(ResearchResult, description= 'Represent the results of a research query. Always use this after using web search tools.')

# List of tools
tools = [think_tool, tavily_search, wikipedia_search, duckduckgo_search, research_result]
# Dictionary of tools: tool name -> tool
tools_by_name = {tool.name: tool for tool in tools}



''' LLM '''
researcher = myChatOpenAI(
    temperature= 0.7
).bind_tools(tools)

summariser = myChatOpenAI(
    temperature= 0.3
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



''' Nodes '''
# The do_research node, where the agent does the actual research by calling the tools
def do_research(state: InputSchema) -> InputSchema:
    '''
    In this node, the agent does the actual research by calling the tools.
    '''
    print_function_name() if DEBUG else None
    state['error_occurred'] = False

    try:
        # prompt
        prompt = prompts.RESEARCH_PROMPT.format(
            date= get_today_str(), 
            topic= state['research_topic']
        )
        # call the LLM
        results = safe_invoke(researcher, [SystemMessage(content= prompt)] + state['messages'])

        print(f'{BLUE}[NODE] [INFO] [RESULTS]{RESET} {results}') if DEBUG else None

        return {'messages': [results], 'error_occurred': False}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        state['error_occurred'] = True
        return state
    
# The tool_node node, where the agent uses the tools
def tool_node(state: InputSchema) -> InputSchema:
    '''
    This node executes the tools, which are used to gather information from the web, or structure the thought process or information.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # Get the last message, and extract the tool calls
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            from_kwargs = False
            tool_calls = last_message.tool_calls
        else:
            from_kwargs = True
            tool_calls = last_message.additional_kwargs.get('tool_calls', [])

        print(json.dumps(tool_calls, indent= 4)) if DEBUG else None

        # Execute all tool calls
        observations = []
        update: dict[str, list] = {'messages': [], 'results': state.get('results', [])}
        for tool_call in tool_calls:
            # Get the tool and arguments
            if from_kwargs:
                tool_call = tool_call['function']
            tool = tools_by_name[tool_call['name']]

            args = tool_call.get('args', {}) or tool_call.get('arguments', {})
            # Parse the tool arguments if needed.
            if isinstance(args, str):
                args = parse_tool_arguments(args)                

            print(f'{BLUE}[NODE] [INFO] [TOOL CALL]{RESET} {tool_call["name"]} with {args}') if DEBUG else None

            try:
                observation = None

                # Sometimes the query is a list of queries, because of wrong tool arguments by the LLM
                if hasattr(args, 'query') and isinstance(args, list):
                    # Get a copy of the arguments
                    arg = args.copy()
                    for query in args['query']:
                        # Run it with the query
                        arg['query'] = query
                        observation = tool.invoke(arg)
                        if observation:
                            observations.append(observation)
                            
                        print(f'{BLUE}[NODE] [INFO] [LAST OBSERVATION]{RESET} {observation}') if DEBUG else None
                        # Create a list of tool outputs, as ToolMessage
                        update['messages'].append(
                            ToolMessage(
                                content= observation,
                                name= tool_call['name'],
                                tool_call_id= tool_call['id']
                            )
                        )
                    # This tool is finished
                    continue

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

            # If the tool is a research result, add the result to the list
            if tool in [ResearchResult]:
                update['results'].append(observation)

            # Create a list of tool outputs, as ToolMessage
            update['messages'].append(
                ToolMessage(
                    content= observation,
                    name= tool_call['name'],
                    tool_call_id= tool_call['id']
                )
            )
        
        # Add them to the state
        return update

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return state
    
# The summarise node, where the LLM summarises the results of the research
def summarise(state: InputSchema) -> OutputSchema:
    '''
    This node summarises the results of the research, in order to return a strutured paragraph containing all the relevant information.
    '''
    print_function_name() if DEBUG else None
    
    try:
        # prompt
        results = '\n---\n\n'.join([str(r) for r in state.get('results', [])])

        print(f'{BLUE}[NODE] [INFO] [ALL RESULTS]{RESET} {results}') if DEBUG else None

        prompt = prompts.SUMMARY_PROMPT.format(
            date= get_today_str(),
            topic= state['research_topic'], 
            web_findings= '\n---\n'.join(results),
            history= ''.join([message.pretty_repr() for message in state['messages']])
        )

        # call the LLM
        summary = safe_invoke(summariser, SystemMessage(content= prompt))

        print(f'{BLUE}[NODE] [INFO] [SUMMARY]{RESET} {summary}') if DEBUG else None

        return OutputSchema(research_topic= state['research_topic'], summary= summary.content)

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None

        return OutputSchema(research_topic= state['research_topic'], summary= '')



''' Conditional Functions '''
# The should_continue function that checks if the agent should continue
def should_continue(state: InputSchema) -> Literal['tool_node', 'summarise', 'do_research']:
    '''
    Decides if the agent should continue or not (if there are tool calls)
    '''
    print_function_name() if DEBUG else None
    
    try:
        # If an error occured go back
        if state.get('error_occurred', False):
            return 'do_research'

        # Get the last message and extract the tool calls
        last_message = state['messages'][-1]

        # If last message has error code 400 because the context is too long, remove the first message
        if hasattr(last_message, 'error') and str(last_message.error.code) == '400': 
            if 'maximum context length is' in last_message.error['message']:
                print(f'{RED}[NODE] [INFO] [CONTEXT TOO LONG]{RESET} Removed first message') if DEBUG else None
                add_messages(state, RemoveMessage(state['messages'][0].id))

        tool_calls = last_message.tool_calls or last_message.additional_kwargs.get('tool_calls', None)

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

        return 'do_research'



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
        'summarise': 'summarise',
        'do_research': 'do_research'
    }
)
researcher_graph.add_edge('tool_node', 'do_research')
researcher_graph.add_edge('summarise', END)

researcher_app = researcher_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(researcher_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
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
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'researcher',
            'run_name': 'researcher',
            'thread_id': 'researcher'
        }
    }

    user = InputSchema(
        research_topic= 'Tell me the admission deadlines for cs in keio to start my masters in fall 2026'
    )
    response = researcher_app.invoke(user, config= config)
    
    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
