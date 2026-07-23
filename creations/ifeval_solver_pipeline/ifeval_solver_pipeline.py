''' Imports '''
# Langchain imports
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

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
from creations.ifeval_solver_pipeline import ifeval_solver_pipeline_prompts as prompts

import requests
import re



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} ifeval_solver_pipeline') if DEBUG else None



""" Schemas """

class AgentSchema(MessagesState):
	"""
	AgentSchema is the state of the ifeval_solver_pipeline graph (extends MessagesState for message history). It holds the IFEval prompt, conversation messages, mode routing, review loop counter, generated answer, review feedback, and pass status. Used by start (init), generate_response (write answer/mode), review_response (verify, increment loop, set passed), and end (parse output). All keys from node docstrings are included: ifeval_prompt, messages, mode, review_loops, answer, review_feedback, passed_latest_review.
	"""
	ifeval_prompt: str # The IFEval prompt string with explicit verifiable constraints (e.g., from google/IFEval 541-sample set). Loaded at start from external invocation payload; used by generate_response and review_response.
	mode: Literal['generate', 'review', 'done'] # Current workflow mode. Values: 'generate' (loop back for fix), 'review' (after generation), 'done' (finalized). Set by nodes to route edges.
	review_loops: int # Integer count of completed review loops; initialized to 0, incremented max up to 2 in review_response. Bounded by max 2 review loops constraint.
	answer: str # The generated response text satisfying constraints (best-effort). Produced by generate_response, verified by review_response.
	review_feedback: Optional[str] # Feedback from previous review if review_loops > 0; describes violations for fix loop. Used by generate_response to guide fixes.
	passed_latest_review: Optional[bool] # Boolean indicating if latest review passed. Set by review_response; False if loop limit reached with violations. Part of metadata output.




''' Tools '''
# Tavily, to search and gather information from the web
tavily_search = TavilySearch(
    tavily_api_key=TAVILY_API_KEY,
    search_depth="advanced",
    max_results=5,
    include_answer=True
)


@tool
def web_search_tool(query: str) -> dict:
    """
    Overview: Search the web with Tavily and return the results in a compact
    dictionary for use by the generation and review LLMs.
    Caller LLM: generate_response_llm and review_response_llm.
    Outside-the-Tool Work: ToolNode appends the returned ToolMessage.
    Inside-the-Tool Work: Invoke Tavily with the supplied query.
    Args: query (str): The web-search query.
    Returns: dict: {'results': Any} or an explicit error payload.
    """
    try:
        results = tavily_search.invoke({'query': query})
        return {'results': results}
    except Exception as e:
        return {'results': [], 'error': str(e)}


# Wikipedia, to search and gather information from the web
@tool
def wikipedia_search_tool(query: str) -> dict:
    """
    Overview: Search Wikipedia through its MediaWiki API and return up to
    three compact results for use by the generation and review LLMs.
    Caller LLM: generate_response_llm and review_response_llm.
    Outside-the-Tool Work: ToolNode appends the returned ToolMessage.
    Inside-the-Tool Work: Send one guarded HTTP request and parse JSON only
    when Wikipedia returns a JSON response.
    Args: query (str): The Wikipedia search query.
    Returns: dict: {'results': List[Dict[str, str]]} or an error payload.
    """
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 3,
                "format": "json",
                "utf8": 1,
            },
            headers={
                "User-Agent": "ifeval-solver-pipeline/1.0"
            },
            timeout=10,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "json" not in content_type.lower():
            return {
                "results": [],
                "error": (
                    "Wikipedia returned a non-JSON response: "
                    f"HTTP {response.status_code}, "
                    f"Content-Type={content_type or 'unknown'}"
                ),
            }

        data = response.json()
        results: List[Dict[str, str]] = []

        for item in data.get("query", {}).get("search", [])[:3]:
            page_id = item.get("pageid")
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))

            results.append({
                "title": item.get("title", ""),
                "url": (
                    f"https://en.wikipedia.org/?curid={page_id}"
                    if page_id is not None
                    else ""
                ),
                "snippet": snippet,
            })

        return {"results": results}

    except Exception as e:
        return {
            "results": [],
            "error": f"{type(e).__name__}: {e}",
        }


@tool
def substring_check_tool(candidate_text: str, required_substrings: List[str], forbidden_substrings: List[str]) -> dict:
    """
    Overview: Programmatically check whether a candidate text contains all required substrings and none of the forbidden substrings. Returns a structured report of missing required and present forbidden substrings. Used by LLMs to verify keyword constraints from IFEval prompts.
    Caller LLM: generate_response_llm and review_response_llm (both nodes bind this tool).
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None (ToolNode appends ToolMessage; no state change).
    Inside-the-Tool Work (Tool Responsibilities): Perform 'in' checks for each substring, compile results, return dict.
    Instructions: 1. For each s in required_substrings, check if s in candidate_text. 2. For each s in forbidden_substrings, check if s in candidate_text. 3. Build missing and violated lists. 4. Return dict with booleans and lists.
    State Updates (on the caller function): None.
    Args: candidate_text (str): Text to check. required_substrings (List[str]): Must appear. forbidden_substrings (List[str]): Must not appear.
    Returns: dict: {'all_required_present': bool, 'no_forbidden_present': bool, 'missing_required': List[str], 'present_forbidden': List[str]}.
    """
    missing_required: List[str] = [s for s in required_substrings if s not in candidate_text]
    present_forbidden: List[str] = [s for s in forbidden_substrings if s in candidate_text]
    
    return {
        'all_required_present': not missing_required,
        'no_forbidden_present': not present_forbidden,
        'missing_required': missing_required,
        'present_forbidden': present_forbidden
    }

@tool
def length_check_tool(candidate_text: str, max_words: Optional[int], min_words: Optional[int], exact_words: Optional[int]) -> dict:
    """
    Overview: Programmatically check length constraints (word count) on candidate text. Supports max, min, or exact word limits as specified by IFEval prompts. Returns a report with actual count and compliance booleans.
    Caller LLM: generate_response_llm and review_response_llm (both nodes bind this tool).
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None.
    Inside-the-Tool Work (Tool Responsibilities): Split text into words, count, compare to constraints, return dict.
    Instructions: 1. Count words via split(). 2. If exact_words given, check equality. 3. If max_words given, check <=. 4. If min_words given, check >=. 5. Return compliance dict.
    State Updates (on the caller function): None.
    Args: candidate_text (str): Text to measure. max_words (Optional[int]): Upper bound. min_words (Optional[int]): Lower bound. exact_words (Optional[int]): Exact required count.
    Returns: dict: {'word_count': int, 'meets_max': bool, 'meets_min': bool, 'meets_exact': bool}.
    """
    try:
        wc: int = len(candidate_text.split())
        return {
            'word_count': wc,
            'meets_max': True if max_words is None else wc <= max_words,
            'meets_min': True if min_words is None else wc >= min_words,
            'meets_exact': True if exact_words is None else wc == exact_words,
        }
    except Exception:
        return {'word_count': 0, 'meets_max': True, 'meets_min': True, 'meets_exact': True}

@tool
def count_substring_tool(candidate_text: str, substring: str) -> dict:
    """
    Overview: Count the number of non-overlapping occurrences of a specific substring within a candidate text. This tool supports IFEval constraints that require a certain count of symbols, characters, or phrases (e.g., 'Include exactly 3 exclamation marks'). Returns the count for LLM verification or guidance.
    Caller LLM: generate_response_llm and review_response_llm (both nodes bind this tool).
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None (ToolNode appends ToolMessage; no state change).
    Inside-the-Tool Work (Tool Responsibilities): Use string count method to compute occurrences and return compact dict.
    Instructions: 1. Receive candidate_text and substring. 2. Call candidate_text.count(substring). 3. Return dict with the integer count.
    State Updates (on the caller function): None.
    Args: candidate_text (str): Text to search within. substring (str): Substring to count (e.g., '!').
    Returns: dict: {'substring': str, 'count': int}.
    """
    count: int = candidate_text.count(substring)
    return {'substring': substring, 'count': count}

@tool
def capitalization_check_tool(candidate_text: str, expected_case: Literal['upper', 'lower', 'title', 'sentence']) -> dict:
	"""
	Overview: Programmatically check whether a candidate text matches a specified capitalization constraint (all uppercase, all lowercase, title case, or sentence case). Returns compliance boolean and details. Supports IFEval instructions like 'Write in ALL CAPS'.
	Caller LLM: generate_response_llm and review_response_llm (both nodes bind this tool).
	Outside-the-Tool Work (Tool Handler Function Responsibilities): None (ToolNode appends ToolMessage; no state change).
	Inside-the-Tool Work (Tool Responsibilities): Apply Python string methods (isupper, islower, istitile, etc.) and return structured report.
	Instructions: 1. Receive candidate_text and expected_case. 2. For 'upper' check candidate_text.isupper(). 3. For 'lower' check islower(). 4. For 'title' check istitile(). 5. For 'sentence' check first char upper and rest lower per sentence. 6. Return dict with compliance.
	State Updates (on the caller function): None.
	Args: candidate_text (str): Text to check. expected_case (Literal['upper', 'lower', 'title', 'sentence']): Capitalization mode required.
	Returns: dict: {'expected_case': str, 'matches': bool}.
	"""
	try:
		matches: bool = False
		if expected_case == 'upper':
			matches = candidate_text.isupper()
		elif expected_case == 'lower':
			matches = candidate_text.islower()
		elif expected_case == 'title':
			matches = candidate_text.istitle()
		elif expected_case == 'sentence':
			if not candidate_text:
				matches = False
			else:
				# Basic sentence split on '.!?' and check each sentence's first alpha upper, rest alpha lower
				  # local import to avoid top-level change; safe inside try
				sentences: List[str] = re.split(r'[.!?]+', candidate_text)
				matches = True
				for sent in sentences:
					# strip leading non-alphabetic chars
					stripped: str = sent.strip()
					if not stripped:
						continue
					# find first alphabetic character
					first_alpha_idx: int = -1
					for i, ch in enumerate(stripped):
						if ch.isalpha():
							first_alpha_idx = i
							break
					if first_alpha_idx == -1:
						continue  # no alphabetic chars in sentence, skip
					# check first alpha is upper
					if not stripped[first_alpha_idx].isupper():
						matches = False
						break
					# check remaining alphabetic chars are lower
					for ch in stripped[first_alpha_idx + 1:]:
						if ch.isalpha() and not ch.islower():
							matches = False
							break
					if not matches:
						break
		else:
			matches = False
	except Exception:
		matches = False

	return {'expected_case': expected_case, 'matches': matches}

@tool
def symbol_check_tool(candidate_text: str, required_symbols: List[str], forbidden_symbols: List[str]) -> dict:
    """
    Overview: Programmatically verify presence of required symbols/punctuation and absence of forbidden ones in a candidate text. Useful for IFEval prompts demanding specific punctuation (e.g., 'End with a question mark'). Returns violation report.
    Caller LLM: generate_response_llm and review_response_llm (both nodes bind this tool).
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None (ToolNode appends ToolMessage; no state change).
    Inside-the-Tool Work (Tool Responsibilities): Check each char in required/forbidden lists via 'in' and return dict.
    Instructions: 1. For each sym in required_symbols, check if sym in candidate_text. 2. For each sym in forbidden_symbols, check if sym in candidate_text. 3. Compile missing and present lists. 4. Return dict.
    State Updates (on the caller function): None.
    Args: candidate_text (str): Text to check. required_symbols (List[str]): Symbols that must appear. forbidden_symbols (List[str]): Symbols that must not appear.
    Returns: dict: {'all_required_present': bool, 'no_forbidden_present': bool, 'missing_required': List[str], 'present_forbidden': List[str]}.
    """
    try:
        missing_required: List[str] = [s for s in required_symbols if s not in candidate_text]
        present_forbidden: List[str] = [s for s in forbidden_symbols if s in candidate_text]
        return {
            'all_required_present': len(missing_required) == 0,
            'no_forbidden_present': len(present_forbidden) == 0,
            'missing_required': missing_required,
            'present_forbidden': present_forbidden
        }
    except Exception:
        return {
            'all_required_present': True,
            'no_forbidden_present': True,
            'missing_required': [],
            'present_forbidden': []
        }
# TODO: Add Tools (if needed)



''' LLM '''
generate_response_llm = myChatOpenAI(
	temperature= 0.2
).bind_tools([web_search_tool, wikipedia_search_tool, substring_check_tool, length_check_tool, count_substring_tool, capitalization_check_tool, symbol_check_tool])

review_response_llm = myChatOpenAI(
	temperature= 0.0
).bind_tools([web_search_tool, wikipedia_search_tool, substring_check_tool, length_check_tool, count_substring_tool, capitalization_check_tool, symbol_check_tool])



''' Nodes '''
def generate_response(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Generate response using primary LLM based on IFEval prompt and extracted constraints (constraints parsed via code or simple tools like substring/length checks). May use web search or basic string verification tools. Tools can be used by both LLMs. Store in state.answer, set state.mode='review'. 
	Overview: This node generates a constraint-compliant response for the given IFEval prompt using a primary LLM (generate_response_llm). It extracts explicit verifiable constraints from the prompt (via code or simple tools like substring/length checks) and instructs the LLM to produce an answer that strictly follows every constraint (format, length, keywords, language, etc.). The LLM may use tools such as web search or basic string verification during generation. The generated answer is stored in state.answer, and state.mode is set to 'review' to hand off to the review node. This is an LLM+TOOLS node; tools are bound to the LLM and executed via a ToolNode loop within the node's invocation. The same tool definitions are used by both generate_response and review_response nodes (web_search_tool, substring_check_tool, length_check_tool).
	
	Step-by-step instructions:
	1. Read the IFEval prompt from state (e.g., state.ifeval_prompt or state.messages) and any extracted constraints.
	2. If review_loops > 0, include previous feedback from the last review (state.review_feedback) to guide fixes.
	3. Format the generation prompt (prompts.GENERATE_RESPONSE_PROMPT) with the IFEval prompt and constraint details.
	4. Bind the shared tools (web_search_tool, substring_check_tool, length_check_tool) to generate_response_llm and invoke via safe_invoke with SystemMessage and conversation messages.
	5. A ToolNode executes any tool calls requested by the LLM; results are appended as ToolMessages and the LLM continues until a final answer is produced.
	6. Extract the final answer text from the LLM result and store it in state.answer.
	7. Set state.mode = 'review' to route to review_response.
	8. Return the updated state.
	
	Possible inputs needed by the node (state keys):
	- ifeval_prompt (or messages): the IFEval prompt string with explicit constraints.
	- review_loops: integer count of completed review loops; used to decide if feedback is included.
	- review_feedback (optional): feedback from previous review if review_loops > 0.
	- mode: current workflow mode (should be 'generate' or 'review' after a loop).
	
	Possible outputs from the node (state keys):
	- answer: the generated response text satisfying constraints (best-effort).
	- mode: set to 'review' for next node.
	- messages: appended with System/AI/Tool messages from the generation step.
	
	Possible tools (used via LLM.bind_tools and ToolNode, same definitions as in review_response):
	- web_search_tool: for fetching external info if needed for content.
	- substring_check_tool: programmatic check for required/forbidden substrings in a candidate.
	- length_check_tool: programmatic check for length constraints (e.g., word count).
	Note: A ToolNode must be configured to execute these shared tools and feed results back to the LLM within the node.
	"""

    print_function_name()
    try:
        ifeval_prompt: str = state['ifeval_prompt']
        review_loops: int = state.get('review_loops', 0)
        review_feedback: Optional[str] = state.get('review_feedback', None)
        prev_messages: List[BaseMessage] = state.get('messages', [])

        prompt: str = prompts.GENERATE_RESPONSE_PROMPT.format(
            ifeval_prompt=ifeval_prompt,
            review_loops=review_loops,
            review_feedback=review_feedback or ''
        )

        msgs: List[BaseMessage] = [SystemMessage(content=prompt)] + prev_messages
        result: BaseMessage = safe_invoke(generate_response_llm, messages=msgs)

        # MessagesState appends returned messages automatically. Return only the
        # new AI message; returning the whole history duplicates ToolMessages
        # and breaks their tool_call_id relationship.
        if isinstance(result, AIMessage) and result.tool_calls:
            return {'messages': [result]}

        return {
            'messages': [result],
            'answer': result.content,
            'mode': 'review'
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state


def review_response(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Verify state.answer against constraints using llm_checker and programmatic_checker tools (e.g., 'in' substring, len). Tools can be used by both LLMs. Log attempt metadata via log_service code. If violations and review_loops<2, increment counter and loop; else set passed_latest_review and route to end. 
	Overview: This node verifies the generated answer (state.answer) against all extracted IFEval constraints using programmatic checks via the shared tools (e.g., 'in' substring, len) and the LLM's own reasoning. It is an LLM+TOOLS node: the review_response_llm may call the same shared tools as generate_response to validate compliance. The node logs attempt metadata via a log_service code function. If violations are detected and review_loops < 2, it increments review_loops, stores feedback, and loops back to generate_response; otherwise it sets passed_latest_review (boolean) and routes to end. The node handles the loop decision and logging internally before returning state. The same tool definitions are used by both review_response and generate_response nodes (web_search_tool, substring_check_tool, length_check_tool). The llm_checker_tool and programmatic_checker_tool are removed; the LLM within this node itself performs the checking using the shared tools.
	
	Step-by-step instructions:
	1. Read state.answer, state.ifeval_prompt (constraints), and state.review_loops from state.
	2. Format the review prompt (prompts.REVIEW_RESPONSE_PROMPT) with answer and constraints.
	3. Bind the shared tools (web_search_tool, substring_check_tool, length_check_tool) to review_response_llm and invoke via safe_invoke with SystemMessage and context.
	4. ToolNode executes checker tools; results appended as ToolMessages; LLM synthesizes a verification result (pass/fail + violations) using its own reasoning (no separate llm_checker_tool or programmatic_checker_tool).
	5. Use log_service code to record attempt metadata (review_loops, timestamp, pass/fail) internally.
	6. If violations found and state.review_loops < 2: increment state.review_loops, set state.review_feedback with violation details, keep state.mode='generate' (loop back).
	7. Else: set state.passed_latest_review to True if no violations, or False if limit reached with violations; set state.mode='done'.
	8. Return updated state for conditional edge routing.
	
	Possible inputs needed by the node (state keys):
	- answer: the response text to verify.
	- ifeval_prompt (or messages): the IFEval prompt with explicit constraints.
	- review_loops: current integer count of reviews done.
	- mode: should be 'review' when entering.
	
	Possible outputs from the node (state keys):
	- review_loops: incremented if violation and <2.
	- passed_latest_review: boolean indicating if latest review passed.
	- review_feedback: description of violations for fix loop (if looping).
	- mode: 'generate' if looping, 'done' if ending.
	- messages: appended with review LLM and tool messages.
	
	Possible tools (used via LLM.bind_tools and ToolNode, same definitions as in generate_response):
	- web_search_tool: for fetching external info if needed for content.
	- substring_check_tool: programmatic check for required/forbidden substrings in a candidate.
	- length_check_tool: programmatic check for length constraints (e.g., word count).
	Note: A ToolNode must be configured to run these shared tools and return results to the LLM. The LLM itself acts as the checker; no llm_checker_tool or programmatic_checker_tool.
	"""

    print_function_name()
    try:
        # Read state values via dictionary access (MessagesState/TypedDict)
        ifeval_prompt: str = state['ifeval_prompt']
        answer: str = state['answer']
        review_loops: int = state.get('review_loops', 0)
        messages: List[BaseMessage] = state.get('messages', [])

        # Format the review prompt
        prompt: str = prompts.REVIEW_RESPONSE_PROMPT.format(
            ifeval_prompt=ifeval_prompt,
            answer=answer,
            review_loops=review_loops,
            review_feedback=state.get('review_feedback', None) or 'None'
        )

        # Build messages: SystemMessage with prompt + existing conversation
        msgs: List[BaseMessage] = [SystemMessage(content=prompt)] + messages

        # Invoke the review LLM
        result: BaseMessage = safe_invoke(review_response_llm, messages=msgs)

        if not result.content: result.content = 'PASS'

        # MessagesState appends returned messages automatically. Return only the
        # new AI message so tool-call messages are not duplicated.
        if isinstance(result, AIMessage) and result.tool_calls:
            return {
                'messages': [result],
                'review_loops': review_loops,
                'mode': state.get('mode', 'review')
            }

        # Parse result content for pass/fail with stricter heuristic
        content: str = result.content if hasattr(result, 'content') else str(result)
        content_upper: str = content.upper()
        # Stricter: if FAIL appears anywhere, it's a fail. Otherwise require explicit PASS token.
        if 'FAIL' in content_upper:
            passed: bool = False
        else:
            passed = 'PASS' in content_upper

        # Loop back if not passed and under review limit (max 2 loops, so review_loops can be 0 or 1)
        if not passed and review_loops < 2:
            new_loops: int = review_loops + 1
            log_service(new_loops, False, content)
            return {
                'messages': [result],
                'review_loops': new_loops,
                'review_feedback': content,
                'mode': 'generate'
            }

        # Finalize: passed or loops exhausted (review_loops is the count of completed reviews)
        # Log with the actual attempt number (review_loops reflects attempts done)
        log_service(review_loops, passed, None if passed else content)
        return {
            'messages': [result],
            'review_loops': review_loops,
            'passed_latest_review': passed,
            'mode': 'done'
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state





''' Conditional Functions '''
def from_review_response_to(state: AgentSchema) -> Literal["generate_response", "__end__", "review_response_tools_all"]:
    """Route after the review_response node in the ifeval_solver_pipeline graph.

    Logic:
    - First, inspect the last message in state['messages']. If it is an AIMessage
      that contains non-empty tool_calls, the review LLM requested verification tools,
      so route to "review_response_tools_all" (ToolNode) to execute them.
    - Otherwise, read state['mode'] which is set by review_response:
        * If mode == 'generate': violations found and review_loops < 2, so loop back
          to "generate_response" for a fix.
        * If mode == 'done': verification passed or loop limit reached, so route to
          "__end__".
    No side effects, no LLM calls. Pure conditional routing.
    """
    print_function_name()
    messages: List[BaseMessage] = state.get('messages', [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "review_response_tools_all"

    mode: Optional[str] = state.get('mode')
    if mode == 'generate':
        return "generate_response"
    # mode == 'done' or any unexpected value defaults to end
    return "__end__"

def from_generate_response_to(state: AgentSchema) -> Literal["review_response", "generate_response_tools_all"]:
    """ 
    Route from the generate_response node to the correct next step.
    If the last message in state['messages'] is an AIMessage with non-empty tool_calls 
    (i.e., the LLM requested tool execution), route to "generate_response_tools_all" 
    so the ToolNode can execute them. Otherwise (no tool calls, meaning the LLM produced 
    a final answer), route to "review_response" for verification.
    """
    print_function_name() if DEBUG else None
    
    messages: List[BaseMessage] = state['messages']
    if not messages:
        return "review_response"
    
    last_msg: BaseMessage = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, 'tool_calls', None):
        return "generate_response_tools_all"
    
    return "review_response"



''' Graph '''
ifeval_solver_pipeline_graph = StateGraph(AgentSchema)

ifeval_solver_pipeline_graph.add_node("generate_response", generate_response)
ifeval_solver_pipeline_graph.add_node("review_response", review_response)
ifeval_solver_pipeline_graph.add_node("generate_response_tools_all", ToolNode([web_search_tool, wikipedia_search_tool, substring_check_tool, length_check_tool, count_substring_tool, capitalization_check_tool, symbol_check_tool]))
ifeval_solver_pipeline_graph.add_node("review_response_tools_all", ToolNode([web_search_tool, wikipedia_search_tool, substring_check_tool, length_check_tool, count_substring_tool, capitalization_check_tool, symbol_check_tool]))

ifeval_solver_pipeline_graph.add_edge(START, "generate_response")
ifeval_solver_pipeline_graph.add_conditional_edges(
    "generate_response",
    from_generate_response_to,
    {
        "review_response": "review_response",
        "generate_response_tools_all": "generate_response_tools_all",
    }
)
ifeval_solver_pipeline_graph.add_edge("generate_response_tools_all", "generate_response")
ifeval_solver_pipeline_graph.add_conditional_edges(
    "review_response",
    from_review_response_to,
    {   # Not needed just for clarity
        "generate_response": "generate_response",
        "__end__": "__end__",
        "review_response_tools_all": "review_response_tools_all",
    }
)
ifeval_solver_pipeline_graph.add_edge("review_response_tools_all", "review_response")


ifeval_solver_pipeline_app = ifeval_solver_pipeline_graph.compile(checkpointer= MemorySaver())



''' Testing '''
''' Helpful Functions '''
def log_service(review_loops: int, passed: bool, feedback: Optional[str] = None) -> None:
    """
    Log attempt metadata for the ifeval_solver_pipeline review step.
    Records review loop count, pass/fail status, and optional feedback to stdout (or a file).
    No state changes; pure logging side-effect.
    Args:
        review_loops (int): Current count of completed review loops.
        passed (bool): Whether the latest review passed.
        feedback (Optional[str]): Violation details if not passed.
    Returns:
        None
    """
    print(f'{GREEN}[LOG] [REVIEW]{RESET} loops={review_loops} passed={passed} feedback={feedback}') if DEBUG else None


if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(ifeval_solver_pipeline_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/ifeval_solver_pipeline_app.png', 'wb') as f:
        f.write(ifeval_solver_pipeline_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'ifeval_solver_pipeline'
    os.environ['LANGSMITH_PROJECT'] = 'ifeval_solver_pipeline'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'ifeval_solver_pipeline',
            'run_name': 'ifeval_solver_pipeline',
            'thread_id': 'ifeval_solver_pipeline', 
        }
    }

    user = '' # TODO: add
    response = ifeval_solver_pipeline_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')