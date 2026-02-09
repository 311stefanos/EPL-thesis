''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool

# Langgraph imports
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.prebuilt import ToolNode

# Schema imports
from typing import TypedDict, Literal, List, Optional, Annotated, Union, Dict, Any
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
import traceback
import json
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS, read_state_file, clean_llm_output
from creations.academic_query_responder import academic_query_responder_prompts as prompts

from typing import List, Dict, Union



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} academic_query_responder') if DEBUG else None



""" Schemas """

class AgentSchema(MessagesState):
	"""
	Main state schema inheriting from MessagesState. Tracks conversation history through built-in 'messages' list. Used as the state type throughout the academic research agent workflow.
	"""
	pass




''' Tools '''
@tool
def query_researcher_publications(researcher_name: str, topic: str) -> List[Dict[str, Union[str, int, float]]]:
    """
    Overview: Web search for publications by specified researcher on given topic
    Returns mock academic publication data for testing purposes.
    """
    # Input validation
    if not isinstance(researcher_name, str) or not isinstance(topic, str):
        raise TypeError("Both researcher_name and topic must be strings")
    
    if not researcher_name.strip() or not topic.strip():
        return []

    # Normalize inputs for matching
    researcher_lower = researcher_name.lower().strip()
    topic_lower = topic.lower().strip()

    # Mock datasets
    john_doe_ai = [
        {
            'title': 'Advanced Neural Networks in Modern AI Systems',
            'authors': 'John Doe, Sarah Smith',
            'year': 2023,
            'url': 'https://arxiv.org/abs/2310.12345',
            'relevance_score': 0.95
        },
        {
            'title': 'Ethical Considerations in Machine Learning',
            'authors': 'John Doe et al.',
            'year': 2022,
            'url': 'https://doi.org/10.1145/1122445.1122456',
            'relevance_score': 0.88
        },
        {
            'title': 'Automated Reasoning Systems Survey',
            'authors': 'Doe, J.; Johnson, R.',
            'year': 2021,
            'url': 'https://dl.acm.org/doi/10.5555/12345678',
            'relevance_score': 0.82
        }
    ]

    jane_smith_bio = [
        {
            'title': 'CRISPR Gene Editing Advancements',
            'authors': 'Jane Smith, Michael Brown',
            'year': 2023,
            'url': 'https://www.nature.com/articles/s41586-023-06756-4',
            'relevance_score': 0.97
        },
        {
            'title': 'Synthetic Biology in Pharmaceutical Development',
            'authors': 'Smith, J.; Wilson, T.',
            'year': 2022,
            'url': 'https://doi.org/10.1016/j.cell.2022.12.045',
            'relevance_score': 0.91
        }
    ]

    # Pattern matching
    if researcher_lower == 'john doe' and 'ai' in topic_lower:
        return john_doe_ai
    elif researcher_lower == 'jane smith' and 'biology' in topic_lower:
        return jane_smith_bio
    
    # Empty result case
    return []

@tool
def google_scholar_api(query: str, max_results: int) -> List[Dict[str, Union[str, int]]]:
    """
    Overview: Executes academic search via Google Scholar API
    Returns mock academic publication data matching Google Scholar structure.
    """
    # Input validation
    if not isinstance(query, str) or not query.strip():
        return []
    
    if not 1 <= max_results <= 5:
        raise ValueError("max_results must be between 1 and 5")

    # Normalize query for matching
    query_lower = query.lower().strip()

    # Curated mock datasets with 3 papers each
    machine_learning_papers = [
        {
            'title': 'Attention Is All You Need',
            'authors': 'Vaswani, A.; Shazeer, N.; Parmar, N.',
            'year': 2017,
            'url': 'https://arxiv.org/abs/1706.03762',
        },
        {
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
            'authors': 'Devlin, J.; Chang, M.; Lee, K.',
            'year': 2018,
            'url': 'https://arxiv.org/abs/1810.04805',
        },
        {
            'title': 'Deep Residual Learning for Image Recognition',
            'authors': 'He, K.; Zhang, X.; Ren, S.',
            'year': 2015,
            'url': 'https://arxiv.org/abs/1512.03385',
        }
    ]

    biology_papers = [
        {
            'title': 'CRISPR-Cas9 Structures and Mechanisms',
            'authors': 'Jinek, M.; Chylinski, K.; Fonfara, I.',
            'year': 2014,
            'url': 'https://doi.org/10.1146/annurev-biophys-051013-022950',
        },
        {
            'title': 'Single-cell transcriptomics reveals bimodality',
            'authors': 'Patel, A.; Tirosh, I.; Trombetta, J.',
            'year': 2014,
            'url': 'https://doi.org/10.1126/science.1254259',
        },
        {
            'title': 'The Human Cell Atlas',
            'authors': 'Regev, A.; Teichmann, S.; Lander, E.',
            'year': 2017,
            'url': 'https://doi.org/10.7554/eLife.27041',
        }
    ]

    quantum_papers = [
        {
            'title': 'Quantum supremacy using a programmable superconducting processor',
            'authors': 'Arute, F.; Arya, K.; Babbush, R.',
            'year': 2019,
            'url': 'https://doi.org/10.1038/s41586-019-1666-5',
        },
        {
            'title': 'A blueprint for demonstrating quantum supremacy with superconducting qubits',
            'authors': 'Boixo, S.; Isakov, S.; Smelyanskiy, V.',
            'year': 2018,
            'url': 'https://arxiv.org/abs/1803.04402',
        },
        {
            'title': 'Quantum computational advantage using photons',
            'authors': 'Zhong, H.-S.; Wang, H.; Deng, Y.-H.',
            'year': 2020,
            'url': 'https://doi.org/10.1126/science.abe8770',
        }
    ]

    # Query-based routing with expanded keywords
    if 'machine learning' in query_lower or 'ai' in query_lower or 'artificial intelligence' in query_lower:
        return machine_learning_papers[:max_results]
    elif 'biology' in query_lower or 'genetics' in query_lower or 'genome' in query_lower:
        return biology_papers[:max_results]
    elif 'quantum' in query_lower or 'physics' in query_lower:
        return quantum_papers[:max_results]

    # Default return empty
    return []

@tool
def tavily_search(query: str, max_results: int) -> List[Dict[str, Union[str, int]]]:
    """
    Overview: Fallback academic search via Tavily API
    Returns mock academic results with validation and discipline-specific examples.

    Args:
    - query: str - Search terms
    - max_results: int - Max papers to return (1-5)

    Returns:
    - List of dicts with keys: 'title'(str), 'source'(str), 'year'(int), 'url'(str)
    """
    # Input validation
    if not isinstance(query, str) or not query.strip():
        return []

    # Clamp max_results to 1-5 range
    max_results = max(1, min(5, int(max_results)))

    # Enhanced mock datasets with academic sources
    cs_publications = [
        {
            'title': 'Transformer Architectures for Efficient NLP',
            'source': 'NeurIPS Proceedings',
            'year': 2023,
            'url': 'https://proceedings.neurips.cc/paper/2023/hash/abc123'
        },
        {
            'title': 'Ethical Implications of Generative AI',
            'source': 'ACM Digital Library',
            'year': 2022,
            'url': 'https://dl.acm.org/doi/10.1145/123456'
        },
        {
            'title': 'Advances in Federated Learning Systems',
            'source': 'IEEE Transactions',
            'year': 2024,
            'url': 'https://ieeexplore.ieee.org/document/987654'
        },
        {
            'title': 'Quantum Machine Learning Algorithms',
            'source': 'Nature Machine Intelligence',
            'year': 2023,
            'url': 'https://www.nature.com/articles/s42256-023-00735-2'
        }
    ]

    bio_publications = [
        {
            'title': 'CRISPR-Cas9 Genome Editing Advancements',
            'source': 'Nature Biotechnology',
            'year': 2023,
            'url': 'https://www.nature.com/articles/s41587-023-01834-4'
        },
        {
            'title': 'Synthetic Biology in Vaccine Development',
            'source': 'Cell Systems',
            'year': 2022,
            'url': 'https://doi.org/10.1016/j.cels.2022.09.002'
        },
        {
            'title': 'Microbiome Analysis Using Metagenomics',
            'source': 'Science Journal',
            'year': 2024,
            'url': 'https://www.science.org/doi/10.1126/science.adh0001'
        }
    ]

    # Determine dataset based on academic discipline keywords
    query_lower = query.lower()
    if any(kw in query_lower for kw in ['ai', 'computer', 'algorithm', 'quantum']):
        return cs_publications[:max_results]
    elif any(kw in query_lower for kw in ['bio', 'genetic', 'crispr', 'microbiome']):
        return bio_publications[:max_results]

    return []

@tool
def initial_pdf_screening(articles: List[Dict[str, Union[str, List[str], int]]], output_format: Literal['summary', 'brief']) -> str:
    """
    Overview: Generates summary PDF from multiple article metadata
    Returns structured string output with formatted article data based on specified format.
    """
    # Handle empty input case
    if not articles:
        return "No articles provided for PDF screening"

    # Create PDF header
    output = [f"=== Academic Literature Screening Report ===\n"]
    output.append(f"Format: {output_format.title()}\n")
    output.append(f"Articles Processed: {len(articles)}\n\n")

    # Process each article
    for idx, article in enumerate(articles, 1):
        try:
            title = article.get('title', 'Untitled')
            authors = ', '.join(article['authors']) if isinstance(article.get('authors'), list) else article.get('authors', 'Unknown')
            year = article.get('year', 'Unknown')
            url = article.get('url', 'No URL available')
            summary = article.get('summary', 'No summary available')

            if output_format == 'summary':
                article_section = f"""
Article {idx}: {title}
---------------------------------
Authors: {authors}
Year: {year}
URL: {url}

Summary:
{summary}

"""
            else:  # brief format
                article_section = f"""
[{idx}] {title} ({year})
- Authors: {authors}
- Link: {url}
- Key Points: {summary[:100]}{'...' if len(summary) > 100 else ''}
"""

            output.append(article_section)

        except Exception as e:
            print(f'{RED}[TOOL] [ERR]{RESET} Processing article {idx}: {e}') if DEBUG else None
            continue

    # Add final separator
    output.append("\n=== End of Report ===")

    return ''.join(output)

@tool
def detailed_pdf_analysis(article: Dict[str, Union[str, List[str], int, Dict]], analysis_depth: Literal['standard', 'deep']) -> str:
    """
    Overview: Generates detailed analysis PDF for single article
    Generates mock analysis with methodology critique, significance assessment,
    and reproducibility score based on article metadata and analysis depth.

    Args:
    - article: Dict - Contains keys: 'title', 'authors', 'year', 'abstract', 'full_text' (optional)
    - analysis_depth: 'standard' or 'deep' level of analysis

    Returns:
    - str: Formatted analysis document with headers and sections
    """
    # Validate required input structure
    required_keys = ['title', 'authors', 'year']
    if not all(key in article for key in required_keys):
        raise ValueError("Invalid article structure - missing required fields")

    # Extract and type core metadata with safe handling
    title: str = article.get('title', 'Untitled')
    authors: Union[str, List[str]] = article.get('authors', ['Unknown'])
    year: int = article.get('year', 2023)
    abstract: str = article.get('abstract', '')
    full_text: str = article.get('full_text', '')

    # Normalize authors to list format
    if isinstance(authors, str):
        authors = [authors]

    # Extract methods section from full_text if available
    methods: str = ''
    if 'methods' in full_text.lower():
        methods_start = full_text.lower().find('methods')
        methods = full_text[methods_start:] if methods_start != -1 else ''

    # Generate analysis components
    analysis = [
        f"# Detailed Analysis: {title}\n",
        f"## Publication Metadata\n"
        f"- Authors: {', '.join(authors)}\n"
        f"- Publication Year: {year}\n"
    ]

    # Methodology Critique Section
    method_score = min(100, max(0, (year - 2000) * 2 + len(methods)//10))
    analysis.append(
        f"## Methodology Critique\n"
        f"**Rigor Score**: {method_score}/100\n"
        f"**Assessment**: {'Robust experimental design' if method_score > 75 else 'Adequate methodology' if method_score > 50 else 'Limited methodological details'}\n"
    )

    # Significance Assessment with safe abstract handling
    significance_factor = (year - 2010) * 0.5 + len(abstract)*0.1
    analysis.append(
        f"## Scientific Significance\n"
        f"**Impact Rating**: {'High' if significance_factor > 30 else 'Moderate' if significance_factor > 15 else 'Limited'}\n"
        f"**Key Contribution**: Breakthrough in {'novel methodology' if 'method' in (abstract or '').lower() else 'theoretical framework' if 'theory' in (abstract or '').lower() else 'empirical findings'}\n"
    )

    # Reproducibility Score
    repro_score = min(100, len(methods)*2 + (20 if 'data available' in (abstract or '').lower() else 0))
    analysis.append(
        f"## Reproducibility Evaluation\n"
        f"**Score**: {repro_score}/100\n"
        f"**Assessment**: {'Fully reproducible' if repro_score > 80 else 'Partially reproducible' if repro_score > 50 else 'Limited reproducibility'}\n"
    )

    # Deep Analysis Components
    if analysis_depth == 'deep':
        analysis.extend([
            f"## Literature Context\n"
            f"This work builds upon {len(authors)*2} key references in the field, "
            f"primarily focusing on {'machine learning' if 'ai' in title.lower() else 'biological systems' if 'bio' in title.lower() else 'general research'}.\n",
            
            f"## Future Research Directions\n"
            f"1. Extension to {'larger datasets' if 'data' in abstract else 'different domains'}\n"
            f"2. Improved {'validation methods' if method_score < 70 else 'theoretical foundations'}\n"
        ])

    # Add conclusion section
    analysis.append(
        f"## Final Assessment\n"
        f"This work represents a {'significant' if significance_factor > 30 else 'moderate'} "
        f"contribution to the field with {'strong' if repro_score > 70 else 'adequate'} "
        "reproducibility potential."
    )

    return '\n'.join(analysis)
chat_llm = myChatOpenAI(
	temperature= 0.3
).bind_tools([query_researcher_publications, google_scholar_api, tavily_search, initial_pdf_screening, detailed_pdf_analysis])




''' Helpful Functions '''

# TODO: Add Helpful Functions (if needed)



''' Nodes '''
def chat(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Analyzes user query, checks researcher DB first, then executes prioritized searches using google_scholar_api, tavily_search, initial_pdf_screening and detailed_pdf_analysis tools. May loop back for additional processing if tools are called. 
	Processes academic queries through prioritized research workflows and structured response generation.
	
	Steps:
	1. Extract latest user query from state['messages']
	2. Execute strict search sequence:
	   a) query_researcher_publications tool (priority)
	   b) google_scholar_api (first fallback)
	   c) tavily_search (final fallback)
	3. Append formatted response as AIMessage to state['messages']
	
	Inputs from state:
	- messages: List[BaseMessage] - conversation history containing latest HumanMessage
	
	Outputs to state:
	- messages: Updated with new AIMessage containing formatted response
	
	Tools (called via LLM.bind_tools()):
	- query_researcher_publications
	- google_scholar_api
	- tavily_search
	- initial_pdf_screening
	- detailed_pdf_analysis
	
	Note: Tool execution is handled by a ToolNode which appends ToolMessages to state
	"""

    print_function_name()
    try:
        # Extract latest user query with type annotation
        messages: List[BaseMessage] = state['messages']
        latest_query = next(
            (msg.content for msg in reversed(messages) 
             if isinstance(msg, HumanMessage)),
            ''
        )

        # Handle empty query case
        if not latest_query:
            error_msg = AIMessage(content="Error: No user query detected")
            return {'messages': [*messages, error_msg]}

        # Format prompt with query and current timestamp
        prompt = prompts.CHAT_PROMPT.format(query=latest_query)

        # Create message list avoiding system message duplication
        llm_messages = [SystemMessage(content=prompt)] + messages

        # Invoke LLM with tools and conversation history
        result = safe_invoke(chat_llm, messages=llm_messages)

        # Return new state with immutable message list update
        return {'messages': messages + [result]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state



''' Conditional Functions '''
def from_chat_to(state: AgentSchema) -> Literal["__end__", "chat_tools"]:
    """Routes to tools node if last message contains tool calls."""
    print_function_name()
    last_message: BaseMessage = state['messages'][-1]
    return "chat_tools" if will_tool_call([last_message]) else "__end__"



''' Graph '''
academic_query_responder_graph = StateGraph(AgentSchema)

academic_query_responder_graph.add_node("chat", chat)
academic_query_responder_graph.add_node("chat_tools", ToolNode([query_researcher_publications, google_scholar_api, tavily_search, initial_pdf_screening, detailed_pdf_analysis]))

academic_query_responder_graph.add_edge(START, "chat")
academic_query_responder_graph.add_conditional_edges(
    "chat",
    from_chat_to,
    {
        "__end__": END,
        "chat_tools": "chat_tools",
    }
)
academic_query_responder_graph.add_edge("chat_tools", "chat")


academic_query_responder_app = academic_query_responder_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(academic_query_responder_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/academic_query_responder_app.png', 'wb') as f:
        f.write(academic_query_responder_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'academic_query_responder'
    os.environ['LANGSMITH_PROJECT'] = 'academic_query_responder'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'academic_query_responder',
            'run_name': 'academic_query_responder',
            'thread_id': 'academic_query_responder', 
        }
    }

    user = '' # TODO: add
    response = academic_query_responder_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')