''' Imports '''

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from utils.utils import myChatOpenAI, safe_invoke, will_tool_call

from experiments.ablation_study.ablate_all import receipt_agent_prompts as prompts
from experiments.ablation_study.ablate_all.receipt_processor import receipt_processor_app
from experiments.ablation_study.ablate_all.receipt_storage import (
    filter_records_by_date,
    load_records,
    parse_items,
)

''' Constants '''

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG', 'false').strip().lower() in ('1', 'true', 'yes')

''' Schemas '''

class ReceiptAgentSchema(MessagesState):
    # Path to the receipt photo the user sent; empty/absent when the input is a text question.
    image_path: Optional[str]
    # The user's spending question, extracted from the latest message.
    question: Optional[str]
    # Final natural-language answer produced for a spending question.
    answer: Optional[str]
    # Path of the Excel file receipts are stored in.
    excel_path: Optional[str]
    # High-level outcome of the run: 'receipt_saved' | 'question_answered' | 'error'.
    status: Optional[str]
    # Operational error description; empty when everything succeeded.
    error: Optional[str]

''' Tools '''

@tool
def get_all_records() -> str:
    """Return every receipt record currently stored in the Excel file.

    Use this tool when the user's question needs the full detail of past
    purchases, such as questions about specific stores, locations, dates or
    individual receipts, or when the aggregated tools do not fit the question.

    Args:
        None.

    Returns:
        A JSON string shaped as {"count": <int>, "records": [{"cost": number,
        "date": "YYYY-MM-DD", "items": "item (quantity x price CURRENCY), ...",
        "location": str, "category": str}, ...]}.
    """
    records = load_records()
    return json.dumps({'count': len(records), 'records': records}, ensure_ascii=False)


@tool
def get_category_totals(start_date: str = '', end_date: str = '') -> str:
    """Return total spending per category, optionally within an inclusive date range.

    Use this tool for questions like "how much did I spend on groceries this
    month?" — build the ISO date range from today's date first.

    Args:
        start_date: Optional inclusive range start as ISO "YYYY-MM-DD". Empty means no lower bound.
        end_date: Optional inclusive range end as ISO "YYYY-MM-DD". Empty means no upper bound.

    Returns:
        A JSON string shaped as {"start_date": str, "end_date": str,
        "categories": [{"category": str, "total_cost": number, "receipt_count": int,
        "currencies": [str]}, ...]} sorted from highest to lowest total cost.
    """
    records = filter_records_by_date(load_records(), start_date, end_date)
    totals: Dict[str, Dict[str, Any]] = {}
    for record in records:
        category = record.get('category') or 'Uncategorized'
        entry = totals.setdefault(
            category,
            {'category': category, 'total_cost': 0.0, 'receipt_count': 0, 'currencies': set()},
        )
        entry['total_cost'] += float(record.get('cost') or 0.0)
        entry['receipt_count'] += 1
        for item in parse_items(record.get('items', '')):
            if item.get('currency'):
                entry['currencies'].add(item['currency'])
    ranked = sorted(totals.values(), key=lambda entry: entry['total_cost'], reverse=True)
    for entry in ranked:
        entry['currencies'] = sorted(entry['currencies'])
    return json.dumps(
        {'start_date': start_date, 'end_date': end_date, 'categories': ranked},
        ensure_ascii=False,
    )


@tool
def get_monthly_totals(category: str = '') -> str:
    """Return total spending per calendar month, optionally filtered to one category.

    Use this tool for trend questions like "how much did I spend per month on
    dining?" or "what were my biggest spending months?".

    Args:
        category: Optional category name (case-insensitive) to filter by. Empty means all categories.

    Returns:
        A JSON string shaped as {"category": str, "months": [{"month": "YYYY-MM",
        "total_cost": number, "receipt_count": int, "currencies": [str]}, ...]}
        sorted from earliest to latest month.
    """
    records = load_records()
    wanted = category.strip().lower()
    totals: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if wanted and (record.get('category') or '').strip().lower() != wanted:
            continue
        month = (record.get('date') or '')[:7]
        if not month:
            continue
        entry = totals.setdefault(
            month, {'month': month, 'total_cost': 0.0, 'receipt_count': 0, 'currencies': set()}
        )
        entry['total_cost'] += float(record.get('cost') or 0.0)
        entry['receipt_count'] += 1
        for item in parse_items(record.get('items', '')):
            if item.get('currency'):
                entry['currencies'].add(item['currency'])
    months = [totals[key] for key in sorted(totals)]
    for entry in months:
        entry['currencies'] = sorted(entry['currencies'])
    return json.dumps({'category': category, 'months': months}, ensure_ascii=False)


@tool
def get_top_items(limit: int = 3, start_date: str = '', end_date: str = '') -> str:
    """Return the items the user spent the most money on, ranked by total spend.

    Each recorded line item contributes (quantity x unit price) to its item
    total, and identical item names are merged across receipts. Use this tool
    for questions like "what are the top 3 items I spent the most on?".

    Args:
        limit: How many top items to return (minimum 1, default 3).
        start_date: Optional inclusive range start as ISO "YYYY-MM-DD".
        end_date: Optional inclusive range end as ISO "YYYY-MM-DD".

    Returns:
        A JSON string shaped as {"limit": int, "top_items": [{"name": str,
        "total_spend": number, "total_quantity": number, "currencies": [str]}, ...]}
        sorted from highest to lowest spend.
    """
    records = filter_records_by_date(load_records(), start_date, end_date)
    totals: Dict[str, Dict[str, Any]] = {}
    for record in records:
        for item in parse_items(record.get('items', '')):
            key = item['name'].strip().lower()
            if not key:
                continue
            entry = totals.setdefault(
                key,
                {
                    'name': item['name'].strip(),
                    'total_spend': 0.0,
                    'total_quantity': 0.0,
                    'currencies': set(),
                },
            )
            entry['total_spend'] += item['quantity'] * item['unit_price']
            entry['total_quantity'] += item['quantity']
            if item.get('currency'):
                entry['currencies'].add(item['currency'])
    safe_limit = max(1, int(limit or 3))
    ranked = sorted(totals.values(), key=lambda entry: entry['total_spend'], reverse=True)[:safe_limit]
    for entry in ranked:
        entry['currencies'] = sorted(entry['currencies'])
    return json.dumps({'limit': safe_limit, 'top_items': ranked}, ensure_ascii=False)


SPENDING_TOOLS = [get_all_records, get_category_totals, get_monthly_totals, get_top_items]

''' LLM '''

# Tool-bound model that decides which spending queries to run, then answers.
answer_llm = myChatOpenAI(temperature=0.0).bind_tools(SPENDING_TOOLS)

''' Helpful Functions '''

def _message_text(message: BaseMessage) -> str:
    """Extract the plain-text portion of a message, joining multimodal parts."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'text':
                parts.append(str(block.get('text', '')))
            elif isinstance(block, str):
                parts.append(block)
        return ' '.join(parts).strip()
    return ''


def _last_human_text(messages: List[BaseMessage]) -> str:
    """Return the text of the most recent human message, or '' when none exists."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ''

''' Nodes '''

def prepare_input_node(state: ReceiptAgentSchema) -> Dict[str, Any]:
    """Normalize the incoming request: keep photos as-is, extract question text."""
    updates: Dict[str, Any] = {'error': ''}
    if state.get('image_path'):
        return updates
    question = (state.get('question') or '').strip()
    if not question:
        question = _last_human_text(state.get('messages', []))
    if question:
        updates['question'] = question
    else:
        updates['error'] = 'No receipt photo or spending question was found in the request.'
    return updates


def process_receipt_node(state: ReceiptAgentSchema) -> Dict[str, Any]:
    """Run the receipt-processing subgraph on the supplied photo."""
    try:
        result = receipt_processor_app.invoke({'image_path': state.get('image_path') or ''})
    except Exception:
        if DEBUG:
            import traceback

            traceback.print_exc()
        return {'error': 'The receipt processing pipeline failed unexpectedly.', 'status': 'error'}
    return {
        'messages': result.get('messages', []),
        'excel_path': result.get('excel_path', ''),
        'status': result.get('status', ''),
        'error': result.get('error', '') or '',
    }


def answer_question_node(state: ReceiptAgentSchema) -> Dict[str, Any]:
    """Ask the tool-bound LLM to query spending data or produce the final answer."""
    question = state.get('question', '')
    today = date.today().isoformat()
    prompt = prompts.ANSWER_QUESTION_PROMPT.format(question=question, today=today)
    messages = [SystemMessage(content=prompt)] + list(state.get('messages', []))
    try:
        response = safe_invoke(answer_llm, messages)
    except Exception:
        if DEBUG:
            import traceback

            traceback.print_exc()
        return {
            'messages': [AIMessage(content='⚠️ Sorry, I could not analyze your spending right now. Please try again in a moment.')],
            'error': 'The spending assistant failed to respond.',
        }
    return {'messages': [response]}


def finalize_answer_node(state: ReceiptAgentSchema) -> Dict[str, Any]:
    """Copy the final assistant message into the dedicated answer field."""
    answer = ''
    for message in reversed(state.get('messages', [])):
        if isinstance(message, AIMessage):
            answer = _message_text(message)
            break
    status = 'error' if state.get('error') else 'question_answered'
    return {'answer': answer, 'status': status}


def handle_error_node(state: ReceiptAgentSchema) -> Dict[str, Any]:
    """Produce the user-facing message for the failure recorded in state."""
    error = state.get('error') or 'Something went wrong while handling your request.'
    if DEBUG:
        print(f'[receipt_agent] run failed: {error}')
    return {
        'messages': [AIMessage(content=f'⚠️ Sorry, I could not process your request: {error}')],
        'status': 'error',
    }

''' Conditional Functions '''

def route_input(state: ReceiptAgentSchema) -> Literal['process_receipt', 'answer_question', 'handle_error']:
    """Send photos to the receipt pipeline and text to the spending assistant."""
    if state.get('image_path'):
        return 'process_receipt'
    if state.get('error'):
        return 'handle_error'
    return 'answer_question'


def route_after_receipt(state: ReceiptAgentSchema) -> Literal['handle_error', 'done']:
    """Fail visibly when the receipt pipeline reported a problem."""
    return 'handle_error' if state.get('error') else 'done'


def route_after_answer(state: ReceiptAgentSchema) -> Literal['spending_tools', 'finalize_answer']:
    """Loop through tool execution until the assistant produces its final answer."""
    if will_tool_call(state.get('messages', [])):
        return 'spending_tools'
    return 'finalize_answer'

''' Graph '''

receipt_agent_graph = StateGraph(ReceiptAgentSchema)

receipt_agent_graph.add_node('prepare_input', prepare_input_node)
receipt_agent_graph.add_node('process_receipt', process_receipt_node)
receipt_agent_graph.add_node('answer_question', answer_question_node)
receipt_agent_graph.add_node('spending_tools', ToolNode(SPENDING_TOOLS))
receipt_agent_graph.add_node('finalize_answer', finalize_answer_node)
receipt_agent_graph.add_node('handle_error', handle_error_node)

receipt_agent_graph.add_edge(START, 'prepare_input')
receipt_agent_graph.add_conditional_edges(
    'prepare_input',
    route_input,
    {
        'process_receipt': 'process_receipt',
        'answer_question': 'answer_question',
        'handle_error': 'handle_error',
    },
)
receipt_agent_graph.add_conditional_edges(
    'process_receipt',
    route_after_receipt,
    {'handle_error': 'handle_error', 'done': END},
)
receipt_agent_graph.add_conditional_edges(
    'answer_question',
    route_after_answer,
    {'spending_tools': 'spending_tools', 'finalize_answer': 'finalize_answer'},
)
receipt_agent_graph.add_edge('spending_tools', 'answer_question')
receipt_agent_graph.add_edge('finalize_answer', END)
receipt_agent_graph.add_edge('handle_error', END)

receipt_agent_app = receipt_agent_graph.compile()

''' Testing '''

if __name__ == '__main__':
    import uuid

    config = {
        'recursion_limit': 40,
        'configurable': {
            'user_id': 'ablate_all_test',
            'run_name': 'ablate_all_test',
            'thread_id': f'ablate_all_test:{uuid.uuid4()}',
        }
    }

    print(
        'Commands:\n'
        '  re:<path>  Process a receipt image\n'
        '  q          Quit\n'
        '  anything else is treated as a spending question\n'
    )

    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'

    user_in = input(f'{GREEN}[USER INPUT]{RESET} > ')

    while user_in.lower() != 'q':
        if user_in.startswith('re:'):
            image_path = user_in[3:].strip()
            response = receipt_agent_app.invoke({
                'image_path': image_path,
                'messages': [HumanMessage(content='Please process and save this receipt.')],
            }, config= config)

        else:
            response = receipt_agent_app.invoke({
                'messages': [HumanMessage(content= user_in)]
            }, config= config)

        print(f'\n{BLUE}[STATUS]{RESET} {response.get("status")}')

        if response.get('answer'):
            print(f'{BLUE}[ANSWER]{RESET} {response["answer"]}')

        else:
            # Receipt processing and errors return their output through messages.
            for message in reversed(response.get('messages', [])):
                if isinstance(message, AIMessage) and message.content:
                    print(f'{BLUE}[ANSWER]{RESET} {message.content}')
                    break

        if response.get('error'):
            print(f'{RED}[ERROR]{RESET} {response["error"]}')

        user_in = input(f'\n{GREEN}[USER INPUT]{RESET} > ')