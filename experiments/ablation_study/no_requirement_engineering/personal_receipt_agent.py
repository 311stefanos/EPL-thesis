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
try:
    from experiments.ablation_study.no_requirement_engineering.process_receipt_subgraph import process_receipt_subgraph_app
except Exception:  # Subgraph module not available (yet) — the receipt branch degrades gracefully.
    process_receipt_subgraph_app = None
from experiments.ablation_study.no_requirement_engineering import personal_receipt_agent_prompts as prompts

from langchain_core.runnables import RunnableConfig
from datetime import datetime
import pandas as pd
import math

import tempfile



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'

# Default receipts Excel ledger (columns: cost, date, items, location, category)
DEFAULT_EXCEL_PATH = os.getenv('RECEIPTS_EXCEL_PATH', str(Path(__file__).resolve().parent / 'receipts.xlsx'))

# Hard bound for the answer_spending_question <-> tools loop: after this many
# LLM passes that emit tool_calls within one run, answer_spending_question
# force-finalizes its reply instead of returning more tool calls, so the loop
# can never execute unboundedly.
MAX_TOOL_ITERATIONS = 4



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} personal_receipt_agent') if DEBUG else None



""" Schemas """

class IntentClassification(BaseModel):
    """
    Structured-output schema for the classify_intent node (root-graph router). Used exactly as: structured_llm = classify_intent_llm.with_structured_output(IntentClassification), invoked via safe_invoke with [SystemMessage(content=CLASSIFY_INTENT_PROMPT), HumanMessage(content=message_text)] — deliberately without conversation history, since classification only needs the current message. Because the label is schema-enforced, classify_intent reads result.intent directly (no free-text parsing or label normalization) and maps it into state['intent'] and state['next_action']. On LLM failure the node falls back to intent='unknown' / next_action='clarify' so the conditional edge can still route and the user eventually gets a reply.
    """
    intent: Literal['receipt', 'spending_question', 'unknown'] = Field(..., description="The classified label for the newest inbound message: 'receipt' = the message carries a photo of a receipt; 'spending_question' = a text query about spending; 'unknown' = anything else.") # Schema-enforced label read directly by classify_intent via result.intent (no free-text parsing or label normalization).

class ReceiptRecord(TypedDict):
    """
    Single schema for one receipt everywhere (merges the former ReceiptData and ReceiptRecord per user feedback — the only difference was the category field, now Optional). Used in three places: (1) state['receipt_data'] — the structured receipt the process_receipt node maps back from the process_receipt_subgraph result (cost, date, items, location; category may be None there because the subgraph returns it as a separate key that the node mirrors into state['category']); (2) the element type of the 'records' list returned by the query_receipts_excel and filter_receipt_records tools bound in answer_spending_question (category always set, read from the Excel row); (3) the row shape append_receipt_to_excel appends, creating the Excel file with the standard columns cost, date, items, location, category if missing. 'items' must already be formatted as 'item1 (quantity x price currency), item2 (quantity x price currency), ...' to match the required Excel items format, and 'date' must be normalized to an ISO string (e.g. 'YYYY-MM-DD') so the query_receipts_excel tool can apply inclusive date_from/date_to filters. All values are JSON-serializable so tool outputs stay compact dicts attachable as ToolMessage content; the LLM must ground every reported figure strictly in these records and never invent numbers. Also used to compose the user-facing confirmation written to state['latest'] (e.g. 'Saved receipt from <location> on <date> — <cost>. Category: <category>.').
    """
    cost: float # Total receipt amount as printed on the receipt; becomes the Excel 'cost' column, feeds total_cost aggregates and top-N sorting, and appears in the user confirmation.
    date: str # Receipt date normalized to an ISO string (e.g. 'YYYY-MM-DD'); becomes the Excel 'date' column and enables inclusive date_from/date_to filtering and 'this month' style ranges.
    items: str # Items formatted exactly as the Excel 'items' column requires: 'item1 (quantity x price currency), item2 (quantity x price currency), ...'; searched by the item_keyword filter.
    location: str # Store/merchant name or location printed on the receipt; becomes the Excel 'location' column.
    category: Optional[str] # Spending category (e.g. 'groceries', 'dining', 'transport'). None while the record is freshly extracted by the subgraph (the category is mirrored separately in state['category']); always a str for records read from or written to the Excel 'category' column; used for category filtering and per-category totals.

class AgentSchema(MessagesState):
    """
    Root state schema for the personal_receipt_agent graph (StateGraph(AgentSchema)), persisted across runs by the MemorySaver checkpointer — this persisted state is the only memory between WhatsApp-style turns. The messages key is inherited from MessagesState (with the add_messages reducer) and is deliberately NOT redeclared: it is the ordered conversation log that classify_intent reads (newest HumanMessage text + optional image attachment), that process_receipt and answer_spending_question extend with the final AIMessage(latest), and that answer_spending_question also extends with intermediate AIMessage/ToolMessage exchanges from its bounded tool loop. Key details: (1) intent — label for the newest inbound message, written by classify_intent via structured output (IntentClassification) and read by the conditional edge from_classify_intent_to. (2) next_action — routing/await hint for turn-based resumption, persisted via the checkpointer. (3) excel_path — optional explicit path to the receipts Excel file (columns: cost, date, items, location, category); all Excel access falls back to the configured default when absent. (4) receipt_data — structured receipt mapped back from the process_receipt subgraph (see ReceiptRecord); absent until the receipt branch runs. (5) category — spending category determined from the items (receipt branch only). (6) ocr_text — raw OCR text if the subgraph exposes it. (7) latest — the single user-visible reply for the current run; every run ends at END immediately after latest is produced (conversational node contract).
    """
    intent: Literal['receipt', 'spending_question', 'unknown'] # Classification of the newest inbound message, written by classify_intent via structured output (IntentClassification) and read by the conditional edge from_classify_intent_to ('receipt' -> process_receipt, 'spending_question' -> answer_spending_question, 'unknown' -> clarify fallback on LLM failure).
    next_action: Literal['process_receipt', 'answer_question', 'clarify', 'done', 'await_receipt_photo', 'retry_receipt'] # Routing/await hint persisted via the checkpointer for turn-based resumption. classify_intent sets 'process_receipt'|'answer_question'|'clarify'; process_receipt sets 'done'|'await_receipt_photo'|'retry_receipt'; answer_spending_question always sets 'done' (terminal branch).
    excel_path: Optional[str] # Explicit path to the receipts Excel file (columns: cost, date, items, location, category). When absent, process_receipt and answer_spending_question fall back to the configured default path.
    receipt_data: Optional[ReceiptRecord] # Structured receipt mapped back from the process_receipt subgraph: {cost, date, items, location} with category possibly None (mirrored separately in state['category']). Absent (None) until the receipt branch runs. See the ReceiptRecord schema (merged ReceiptData+ReceiptRecord per user feedback).
    category: Optional[str] # Spending category (e.g. 'groceries', 'dining', 'transport') determined from the items; set only by the receipt branch and written to the Excel 'category' column.
    ocr_text: Optional[str] # Raw OCR text of the receipt photo, if the subgraph exposes it (useful for debugging); absent otherwise.
    latest: str # The single user-visible reply for the current run (receipt confirmation/failure, spending answer, or clarify prompt). Also appended to messages as an AIMessage; the run transitions to END right after.
    tool_iterations: int # Bounded tool-loop counter for the answer_spending_question <-> tools loop: answer_spending_question reads it on entry (resetting it to 0 whenever the newest message is still the run's inbound HumanMessage, i.e. the first pass of this branch in the current run, so stale checkpointer values never leak across turns) and writes it incremented by 1 each time it returns an AIMessage carrying tool_calls (one ToolNode execution per increment); once the counter reaches MAX_TOOL_ITERATIONS the node force-finalizes the user-visible reply instead of returning more tool calls, guaranteeing the loop terminates after a bounded number of ToolNode executions.




''' Tools '''
@tool
def query_receipts_excel(excel_path: str, category: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, item_keyword: Optional[str] = None) -> dict:  # {'records': List[dict], 'count': int, 'total_cost': float, 'error': Optional[str]}:
    """
    Read-only Excel query tool (NON-TERMINAL context updater). Loads the receipts Excel file (columns: cost, date, items, location, category) with pandas/openpyxl, applies optional filters, and returns the matching records plus their count and total cost as one compact JSON-serializable dict.
    
    Overview:
        Primary data-access tool for the spending-question branch. It performs NO
        writes and NO LLM calls: it reads the Excel ledger at excel_path, coerces
        every row into a ReceiptRecord-shaped dict ({cost: float, date: str,
        items: str, location: str, category: str}), applies the optional filters —
        category equality (case-insensitive), inclusive date range on the ISO
        'date' column (date_from <= date <= date_to), and case-insensitive
        substring match of item_keyword against 'items' — and returns
        {'records': [...], 'count': int, 'total_cost': float, 'error': None}.
        'total_cost' is the sum of 'cost' over the RETURNED (filtered) records
        rounded to 2 decimals, so 'how much did I spend on groceries this month?'
        can be answered directly from it. The caller LLM must ground every figure
        in its final answer strictly in these records and never invent numbers.
        The tool never raises: if the file is missing, unreadable or malformed it
        returns {'records': [], 'count': 0, 'total_cost': 0.0, 'error': '<short
        reason>'} so the LLM can tell the user no matching data was found. After
        this tool runs, control returns to the caller LLM in the same run
        (non-terminal): the LLM either chains another tool (e.g.
        filter_receipt_records for top-N refinements) or produces the final
        user-facing answer.
    
    Caller LLM:
        answer_spending_question_llm (temperature 0) in the answer_spending_question
        node, bound via answer_spending_question_llm.bind_tools([query_receipts_excel,
        append_receipt_to_excel, filter_receipt_records]). Typically the FIRST tool
        called for any spending question.
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities):
        Owned by the bounded inline tool-execution loop in answer_spending_question
        (ToolNode-style, max ~4 iterations): it detects AIMessage.tool_calls,
        executes this tool with the LLM-supplied arguments, appends the result to
        state['messages'] as a ToolMessage (content = json.dumps of the returned
        dict, tool_call_id matched to the request), and re-invokes the tool-bound
        LLM. The handler — never this tool — writes state['latest'], appends the
        final AIMessage(latest) and sets next_action='done' once a final
        no-tool-call AIMessage is produced.
    
    Inside-the-Tool Work (Tool Responsibilities):
        1. Validate excel_path (non-empty string); on invalid input return the
           error-shaped dict without touching the filesystem.
        2. Load the workbook READ-ONLY (pandas.read_excel / openpyxl); never
           create, modify or lock the file.
        3. Normalize rows to the ReceiptRecord shape: cost -> float, date -> ISO
           'YYYY-MM-DD' string (coerce datetime cells), items/location/category ->
           str; skip fully empty rows.
        4. Apply filters in order: category (case-insensitive equality), inclusive
           ISO date range (plain string comparison is safe for ISO dates),
           item_keyword (case-insensitive substring on 'items').
        5. Compute count = len(records) and total_cost = round(sum of costs, 2).
        6. Return the compact dict described under Returns; wrap ANY exception as
           {'records': [], 'count': 0, 'total_cost': 0.0, 'error': str(e)}.
    
    State Updates (on the caller function):
        None directly — a tool cannot see or mutate the graph state. The handler in
        answer_spending_question appends, on the LLM's behalf: messages +=
        [AIMessage(tool_calls=...), ToolMessage(content=result, tool_call_id=...)];
        after the loop it sets latest (final natural-language answer), appends it
        as an AIMessage, and sets next_action='done'. excel_path is NOT read from
        state by this tool: the node resolves it (state['excel_path'] or the
        configured default), places it in the system prompt, and the LLM passes it
        here as a plain argument. No other state keys are involved.
    
    Instructions:
        1. Call this tool FIRST for any spending question, passing the excel_path
           exactly as given in the system prompt.
        2. Translate relative time words into ISO bounds BEFORE calling, using the
           current date supplied in the system prompt: 'this month' -> date_from =
           first day of the current month, date_to = today; 'last month' -> first
           and last day of the previous month; 'this year' -> Jan 1 .. today. Omit
           date_from/date_to entirely when the question has no time constraint.
        3. Pass category only when the user names one (e.g. 'groceries'); pass
           item_keyword only when the user asks about a specific item or merchant.
           Never pass filters the user did not ask for.
        4. Use 'total_cost' for 'how much' totals, 'count' for 'how many'
           questions, and 'records' for per-item, per-store or top-N reasoning
           (chain filter_receipt_records when a subset/sort/top-N of THESE records
           is needed instead of re-reading the file).
        5. If 'error' is non-None or count == 0, reply that no matching receipts
           were found (mention the reason only when user-relevant, e.g. ledger
           file not found) — do NOT fabricate or estimate numbers.
        6. Keep total tool calls per run bounded (~4); once the records in hand
           answer the question, produce the final answer and stop calling tools.
    
    Args:
        excel_path (str): Path to the receipts Excel file with columns cost, date,
            items, location, category. Provided by the node via the system prompt
            (state['excel_path'] or the configured default). Required.
        category (Optional[str]): Spending category to keep, case-insensitive exact
            match (e.g. 'groceries', 'dining', 'transport'). None = no filter.
        date_from (Optional[str]): Inclusive lower bound on the receipt date, ISO
            'YYYY-MM-DD'. None = no lower bound.
        date_to (Optional[str]): Inclusive upper bound on the receipt date, ISO
            'YYYY-MM-DD'. None = no upper bound.
        item_keyword (Optional[str]): Case-insensitive substring matched against
            the 'items' column (e.g. 'coffee', 'Cafe X'). None = no filter.
    
    Returns:
        dict: {'records': List[dict], 'count': int, 'total_cost': float,
        'error': Optional[str]} where records is a list of ReceiptRecord-shaped
        dicts {cost: float, date: str, items: str, location: str, category: str},
        count is len(records), total_cost is the sum of their costs rounded to 2
        decimals (0.0 when empty), and error is None on success or a short reason
        string when the file is missing/unreadable (records then empty, totals 0).
    """
    # 1. Validate excel_path (non-empty string) WITHOUT touching the filesystem.
    if not isinstance(excel_path, str) or not excel_path.strip():
        return {'records': [], 'count': 0, 'total_cost': 0.0, 'error': 'invalid excel_path'}

    # Nested scalar-blank predicate: True for None/NaN/NaT and blank/whitespace strings.
    def _is_blank(value: Any) -> bool:
        return bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())

    try:
        # 2. Load the workbook READ-ONLY (never create, modify or write the file).
        df: pd.DataFrame = pd.read_excel(excel_path)
        rows: List[Dict[str, Any]] = df.to_dict('records')

        records: List[Dict[str, Any]] = []

        # 3. Normalize every row into a ReceiptRecord-shaped dict.
        for row in rows:
            raw_cost: Any = row.get('cost')
            raw_date: Any = row.get('date')
            raw_items: Any = row.get('items')
            raw_location: Any = row.get('location')
            raw_category: Any = row.get('category')

            # Skip fully empty rows (all values NaN or empty strings).
            if all(_is_blank(cell) for cell in (raw_cost, raw_date, raw_items, raw_location, raw_category)):
                continue

            # cost -> float; skip the row entirely if NaN/empty/unparseable.
            if _is_blank(raw_cost):
                continue
            try:
                cost: float = float(raw_cost)
            except Exception:
                continue
            if math.isnan(cost):
                continue

            # date -> ISO 'YYYY-MM-DD' string (pd.Timestamp subclasses datetime).
            if isinstance(raw_date, datetime):
                date_str: str = raw_date.strftime('%Y-%m-%d')
            else:
                try:
                    date_str = pd.to_datetime(raw_date).strftime('%Y-%m-%d')
                except Exception:
                    date_str = str(raw_date).strip()

            record: Dict[str, Any] = {
                'cost': cost,
                'date': date_str,
                'items': '' if _is_blank(raw_items) else str(raw_items),
                'location': '' if _is_blank(raw_location) else str(raw_location),
                'category': '' if _is_blank(raw_category) else str(raw_category),
            }

            # 4. Apply filters IN ORDER, each ONLY when its argument is truthy.
            # (a) category — case-insensitive exact equality.
            if category and record['category'].lower() != str(category).lower():
                continue
            # (b) inclusive ISO date range — plain string comparison (safe for ISO dates).
            if date_from and record['date'] < str(date_from):
                continue
            if date_to and record['date'] > str(date_to):
                continue
            # (c) item_keyword — case-insensitive substring match on 'items'.
            if item_keyword and str(item_keyword).lower() not in record['items'].lower():
                continue

            records.append(record)

        # 5. Aggregates over the RETURNED (filtered) records.
        count: int = len(records)
        total_cost: float = round(float(sum(rec['cost'] for rec in records)), 2)

        # 6. Success payload.
        return {'records': records, 'count': count, 'total_cost': total_cost, 'error': None}

    # 7. NEVER raise: wrap ANY exception (missing/unreadable/malformed file).
    except Exception as e:
        return {'records': [], 'count': 0, 'total_cost': 0.0, 'error': str(e)}

@tool
def append_receipt_to_excel(excel_path: str, cost: float, date: str, items: str, location: str, category: str) -> dict:  # {'success': bool, 'row_index': int, 'error': Optional[str]}:
    """
    Excel-append tool (TERMINAL finalizer). Appends exactly one receipt row to the Excel file at excel_path — creating the file with the header cost, date, items, location, category if it does not exist — and returns a compact {'success': bool, 'row_index': int, 'error': Optional[str]} payload.
    
    Overview:
        Side-effectful write tool for the spending-question branch, used when the
        user dictates a receipt in plain text (no photo) and asks the agent to
        save it (e.g. 'save this: 12.50 at Cafe X yesterday, two coffees and a
        croissant'). It writes ONE row with the five standard columns and reports
        success/failure; it performs no reads for answering questions and no LLM
        calls. This is a TERMINAL (finalizer) tool: once its ToolMessage result is
        back, the caller LLM must compose the single final user-visible
        confirmation (or failure apology) and end the run WITHOUT any further
        tool calls. The LLM is responsible for normalizing the dictated text
        BEFORE calling: cost as a float; date as ISO 'YYYY-MM-DD' (resolve words
        like 'yesterday' against the current date from the system prompt); items
        as 'item1 (quantity x price currency), item2 (quantity x price currency),
        ...' (the currency lives inside the items strings — there is no currency
        column); location as the store/merchant name; category as the
        LLM-inferred spending category from the items.
    
    Caller LLM:
        answer_spending_question_llm (temperature 0) in the answer_spending_question
        node, bound via answer_spending_question_llm.bind_tools([query_receipts_excel,
        append_receipt_to_excel, filter_receipt_records]). Called only when the newest user message asks to
        save/record a receipt described in text.
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities):
        None
    
    Inside-the-Tool Work (Tool Responsibilities):
        1. Validate arguments BEFORE any write: cost a finite number, date parses
           as 'YYYY-MM-DD', items/location/category non-empty strings. On invalid
           input return {'success': False, 'row_index': -1, 'error': '<reason>'}
           and write nothing.
        2. If the Excel file does not exist, create it with the header row: cost,
           date, items, location, category.
        3. Append exactly one data row: cost as float, date as the ISO string,
           items/location/category as strings — matching the ReceiptRecord/Excel
           column order.
        4. Save the workbook atomically (pandas/openpyxl): write the complete new
           workbook to a temp file in the target's directory, then os.replace
           that temp file over excel_path in one atomic step, and return
           {'success': True, 'row_index': <0-based index of the appended data
           row>, 'error': None}.
        5. Wrap ANY exception during create/append/save as {'success': False,
           'row_index': -1, 'error': str(e)} — never raise, never leave a
           partially written row.
    
    State Updates (on the caller function):
        None directly — a tool cannot see or mutate the graph state; its only side
        effect is the new row in the Excel file on disk. On the caller side the
        handler appends messages += [AIMessage(tool_calls=...),
        ToolMessage(content=result, tool_call_id=...)] and, after the final
        AIMessage, sets latest (confirmation or failure text), appends it as an
        AIMessage and sets next_action='done'. excel_path is NOT read from state
        by this tool: the node resolves it (state['excel_path'] or the configured
        default), surfaces it in the system prompt, and the LLM passes it here as
        a plain argument. The receipt_data/category state keys are NOT touched —
        those belong to the photo (process_receipt) branch only.
    
    Instructions:
        1. Use this tool ONLY when the user's message clearly asks to save/record
           a receipt they described in text. Never use it to duplicate a receipt
           the photo branch already saved, never for 'test' rows, and never as a
           side effect of answering a pure spending question.
        2. Normalize the dictated details BEFORE calling: cost -> float in the
           receipt's currency; date -> ISO 'YYYY-MM-DD' (resolve
           'yesterday'/'last Friday' etc. with the current date from the system
           prompt); items -> 'item1 (quantity x price currency), ...' exactly;
           location -> store/merchant; category -> your best inference from the
           items (e.g. 'groceries', 'dining', 'transport').
        3. If a required detail is missing or ambiguous (no total, no date,
           unclear items), do NOT call this tool — instead produce a final reply
           asking ONE clear question and end the run; the user's next message
           re-enters the graph via classify_intent.
        4. Call this tool exactly ONCE per receipt. When its ToolMessage returns:
           success=True -> reply with a short confirmation of what was saved
           (e.g. 'Saved: 12.5 at Cafe X on 2025-01-15. Category: dining.');
           success=False -> apologize, include the short error reason, and do NOT
           retry automatically.
        5. Make NO further tool calls after this tool (terminal): compose the
           final reply and end the run.
    
    Args:
        excel_path (str): Path to the receipts Excel file (columns: cost, date,
            items, location, category); created with this header if missing.
            The path must resolve inside the receipts ledger directory (the same
            folder as the configured default ledger) with an .xlsx extension,
            otherwise the tool refuses to write.
            Provided by the node via the system prompt. Required.
        cost (float): Total receipt amount as a number (e.g. 12.5). Required.
        date (str): Receipt date as ISO 'YYYY-MM-DD'. Required.
        items (str): Items formatted exactly as 'item1 (quantity x price
            currency), item2 (quantity x price currency), ...' (e.g. 'coffee (2 x
            3.5 USD), croissant (1 x 5.5 USD)'). Required.
        location (str): Store/merchant name or location for the receipt. Required.
        category (str): Spending category inferred from the items (e.g.
            'groceries', 'dining', 'transport'). Required.
    
    Returns:
        dict: {'success': bool, 'row_index': int, 'error': Optional[str]} —
        success is True iff the row was appended and the file saved; row_index is
        the 0-based index of the appended data row on success and -1 on failure;
        error is None on success or a short human-readable reason (invalid
        arguments, IO error) on failure.
    """
    try:
        # 1. Validate arguments BEFORE any write — each failure returns its own
        # short-reason payload and nothing is written to disk.
        if not isinstance(excel_path, str) or not excel_path.strip():
            return {'success': False, 'row_index': -1, 'error': 'invalid excel_path'}
        excel_path = excel_path.strip()

        # 1b. Path confinement (security allowlist) — immediately after the
        # excel_path validation and BEFORE any other validation or filesystem
        # access: the LLM-supplied path may only point at an .xlsx file directly
        # inside the receipts ledger directory (the folder holding the
        # configured default ledger), preventing the model from creating or
        # overwriting files anywhere else on disk.
        allowed_dir: Path = Path(DEFAULT_EXCEL_PATH).resolve().parent
        try:
            provided_path: Path = Path(excel_path).resolve()
        except Exception:
            return {'success': False, 'row_index': -1, 'error': 'invalid excel_path'}
        if provided_path.parent != allowed_dir or provided_path.suffix.lower() != '.xlsx':
            return {'success': False, 'row_index': -1, 'error': 'excel_path outside the allowed receipts ledger directory'}

        if isinstance(cost, bool) or not isinstance(cost, (int, float)):
            return {'success': False, 'row_index': -1, 'error': 'invalid cost'}
        if not math.isfinite(float(cost)):
            return {'success': False, 'row_index': -1, 'error': 'invalid cost'}

        try:
            datetime.strptime(str(date), '%Y-%m-%d')
        except Exception:
            return {'success': False, 'row_index': -1, 'error': 'invalid date, expected YYYY-MM-DD'}

        if not isinstance(items, str) or not items.strip():
            return {'success': False, 'row_index': -1, 'error': 'invalid items'}
        if not isinstance(location, str) or not location.strip():
            return {'success': False, 'row_index': -1, 'error': 'invalid location'}
        if not isinstance(category, str) or not category.strip():
            return {'success': False, 'row_index': -1, 'error': 'invalid category'}

        # 2. Row values — matching the ReceiptRecord/Excel column order.
        row: Dict[str, Any] = {
            'cost': float(cost),
            'date': str(date),
            'items': str(items),
            'location': str(location),
            'category': str(category),
        }
        columns: List[str] = ['cost', 'date', 'items', 'location', 'category']

        # 3. Create-or-append: missing file -> header + first data row
        # (row_index 0); existing file -> read it, the appended row's 0-based
        # index is the current row count, extend via pd.concat (never the
        # removed DataFrame.append method).
        df: pd.DataFrame
        row_index: int
        if not os.path.exists(excel_path):
            df = pd.DataFrame([row], columns=columns)
            row_index = 0
        else:
            df = pd.read_excel(excel_path)
            row_index = len(df)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        # 4. Save atomically — the complete new workbook is first written to a
        # temp file created in the SAME directory as the target (same
        # filesystem, so the final os.replace is an atomic rename), then
        # os.replace moves that temp file over excel_path in one indivisible
        # step: readers always see either the complete old workbook or the
        # complete new one, the original file is never truncated or left
        # partially written, and on any failure the temp file is deleted while
        # the original workbook stays untouched.
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix='receipts_', suffix='.xlsx', dir=str(allowed_dir))
            os.close(fd)  # pandas/openpyxl opens the path itself — release the fd immediately.
            df.to_excel(tmp_path, index=False)
            os.replace(tmp_path, excel_path)
        except Exception:
            # Best-effort cleanup of the leftover temp file — wrapped so cleanup
            # errors never mask the original error; then propagate into the
            # outer except below for the usual failure payload.
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass  # cleanup errors must never mask the original error
            raise  # propagate into the existing outer except below

        # 5. Success payload.
        return {'success': True, 'row_index': row_index, 'error': None}

    except Exception as e:
        # Never raise: wrap ANY exception during create/append/save.
        return {'success': False, 'row_index': -1, 'error': str(e)}

@tool
def filter_receipt_records(records_json: str, category: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, item_keyword: Optional[str] = None, top_n: Optional[int] = None, sort_by: Optional[str] = None) -> dict:  # {'records': List[dict], 'count': int, 'error': Optional[str]}:
    """
    In-memory filter/sort tool (NON-TERMINAL context updater) over records the LLM already retrieved. Pure function: no file I/O, no side effects — parses a JSON string of previously fetched receipt records, applies optional filters, optional sort and top-N truncation, and returns the refined list.
    
    Overview:
        Follow-up refinement tool for the spending-question branch. Use it when a
        question can be answered from records already returned by
        query_receipts_excel in this run (passed here as a JSON string) instead of
        re-reading the Excel file — e.g. 'of those, what are the top 3?' or 'now
        only the dinners'. It applies the same filter semantics as
        query_receipts_excel (category case-insensitive equality, inclusive ISO
        date range, case-insensitive item_keyword substring on 'items'), then
        optionally sorts (sort_by: 'cost_desc' | 'cost_asc' | 'date_desc' |
        'date_asc') and truncates to top_n AFTER sorting (so top_n=3 +
        sort_by='cost_desc' yields the 3 most expensive records). It returns
        {'records': [...], 'count': int, 'error': None}. It NEVER touches the
        Excel file and never invents records: only records present in
        records_json can appear in the output, and the caller LLM must ground its
        final answer strictly in them. After this tool runs, control returns to
        the caller LLM in the same run (non-terminal): the LLM produces the final
        answer or chains at most one more refinement.
    
    Caller LLM:
        answer_spending_question_llm (temperature 0) in the answer_spending_question
        node, bound via .bind_tools([query_receipts_excel, append_receipt_to_excel,
        filter_receipt_records]). Typically the SECOND step of a chain:
        query_receipts_excel -> filter_receipt_records -> final answer.
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities):
        Owned by the bounded inline tool-execution loop in answer_spending_question:
        it executes this tool on the LLM's request, appends the returned dict to
        state['messages'] as a ToolMessage (content = json.dumps of the result,
        tool_call_id matched), and re-invokes the tool-bound LLM. The handler —
        never this tool — writes state['latest'], appends the final
        AIMessage(latest) and sets next_action='done' when the loop ends.
    
    Inside-the-Tool Work (Tool Responsibilities):
        1. Parse records_json (expected: JSON array of {cost, date, items,
           location, category} dicts). On parse failure return {'records': [],
           'count': 0, 'error': 'invalid records_json'}.
        2. Coerce each entry to the ReceiptRecord shape (cost -> float,
           date/items/location/category -> str), skipping malformed entries.
        3. Apply filters: category (case-insensitive equality), inclusive
           date_from <= date <= date_to (ISO string comparison), item_keyword
           (case-insensitive substring on 'items').
        4. If sort_by is 'cost_desc'/'cost_asc' sort by float cost;
           'date_desc'/'date_asc' sort by ISO date string; any other value is
           ignored (incoming order kept).
        5. If top_n is a positive int, keep only the first top_n records AFTER
           sorting.
        6. Return {'records': [...], 'count': len(records), 'error': None}; wrap
           ANY other exception the same way as step 1 — never raise.
    
    State Updates (on the caller function):
        None directly — a tool cannot see or mutate the graph state, and its
        inputs come from the LLM (the JSON it received in an earlier ToolMessage),
        not from state. The handler in answer_spending_question appends messages
        += [AIMessage(tool_calls=...), ToolMessage(content=result,
        tool_call_id=...)] and, after the final no-tool-call AIMessage, sets
        latest, appends it as an AIMessage and sets next_action='done'. No state
        keys are read or written by or for this tool.
    
    Instructions:
        1. Call this ONLY with records actually received earlier in THIS run:
           serialize the 'records' list from a previous query_receipts_excel
           ToolMessage into records_json. Never paste invented, remembered or
           hallucinated records.
        2. Prefer this tool over re-calling query_receipts_excel when the needed
           refinement is a subset/sort/top-N of data you already hold — it avoids
           a redundant file read; re-query the file only when the earlier records
           cannot answer the question at all.
        3. For 'top N most expensive' questions pass top_n=N,
           sort_by='cost_desc'; for 'most recent N' pass top_n=N,
           sort_by='date_desc'; combine with category/date/item_keyword filters
           to narrow first.
        4. This tool filters and sorts WHOLE receipts, not individual line items:
           for per-item questions inside multi-item receipts (e.g. 'which single
           item cost the most?'), do the item-level reasoning yourself from the
           returned 'items' strings.
        5. Ground the final answer strictly in the returned records; if count ==
           0, say no matching receipts were found. After answering, make no
           further tool calls unless one more genuine refinement is needed (stay
           within the ~4-call bound).
    
    Args:
        records_json (str): JSON string encoding a list of previously retrieved
            records, each {cost: float, date: str, items: str, location: str,
            category: str} — i.e. the 'records' value from a query_receipts_excel
            result. Required.
        category (Optional[str]): Keep only records whose category matches
            (case-insensitive). None = no category filter.
        date_from (Optional[str]): Inclusive lower bound on date, ISO
            'YYYY-MM-DD'. None = no lower bound.
        date_to (Optional[str]): Inclusive upper bound on date, ISO 'YYYY-MM-DD'.
            None = no upper bound.
        item_keyword (Optional[str]): Case-insensitive substring matched against
            the 'items' string. None = no item filter.
        top_n (Optional[int]): Keep only the first N records AFTER sorting
            (positive int). None = keep all.
        sort_by (Optional[str]): One of 'cost_desc', 'cost_asc', 'date_desc',
            'date_asc'. None = keep the incoming order.
    
    Returns:
        dict: {'records': List[dict], 'count': int, 'error': Optional[str]} where
        records is the filtered/sorted/truncated list of ReceiptRecord-shaped
        dicts {cost: float, date: str, items: str, location: str, category: str},
        count is len(records), and error is None on success or a short reason
        string when records_json could not be parsed (records then empty, count
        0).
    """
    try:
        # Step 1: Parse records_json — on ANY parse failure return the exact
        # 'invalid records_json' payload (never raise).
        try:
            parsed: Any = json.loads(records_json)
        except Exception:
            return {'records': [], 'count': 0, 'error': 'invalid records_json'}

        if not isinstance(parsed, list):
            return {'records': [], 'count': 0, 'error': 'invalid records_json'}

        # Step 2: Coerce each entry to the ReceiptRecord shape — skip non-dict
        # entries and entries whose cost cannot be coerced to float.
        def _to_str(value: Any) -> str:
            return '' if value is None else str(value)

        records: List[dict] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            raw_cost: Any = entry.get('cost')
            try:
                cost: float = float('' if raw_cost is None else raw_cost)
            except (TypeError, ValueError):
                continue
            records.append({
                'cost': cost,
                'date': _to_str(entry.get('date')),
                'items': _to_str(entry.get('items')),
                'location': _to_str(entry.get('location')),
                'category': _to_str(entry.get('category')),
            })

        # Step 3: Apply filters — each ONLY when its argument is truthy.
        if category:
            category_value: str = str(category).lower()
            records = [record for record in records if record['category'].lower() == category_value]

        if date_from:
            lower_bound: str = str(date_from)
            records = [record for record in records if record['date'] >= lower_bound]

        if date_to:
            upper_bound: str = str(date_to)
            records = [record for record in records if record['date'] <= upper_bound]

        if item_keyword:
            keyword: str = str(item_keyword).lower()
            records = [record for record in records if keyword in record['items'].lower()]

        # Step 4: Sort — only the four supported values; ANY other sort_by is
        # ignored (incoming order kept). Python sorts are stable.
        if sort_by == 'cost_desc':
            records.sort(key= lambda record: record['cost'], reverse= True)
        elif sort_by == 'cost_asc':
            records.sort(key= lambda record: record['cost'])
        elif sort_by == 'date_desc':
            records.sort(key= lambda record: record['date'], reverse= True)
        elif sort_by == 'date_asc':
            records.sort(key= lambda record: record['date'])

        # Step 5: Truncate to top_n AFTER sorting — only for a positive int
        # that is not a bool.
        if isinstance(top_n, int) and not isinstance(top_n, bool) and top_n > 0:
            records = records[:top_n]

        # Step 6: Success payload.
        return {'records': records, 'count': len(records), 'error': None}

    except Exception as e:
        # Never raise: wrap ANY other exception as the error-shaped payload.
        return {'records': [], 'count': 0, 'error': str(e)}



''' LLM '''
classify_intent_llm = myChatOpenAI(
    temperature= 0.0
).with_structured_output(IntentClassification)

answer_spending_question_llm = myChatOpenAI(
    temperature= 0.0
).bind_tools([query_receipts_excel, append_receipt_to_excel, filter_receipt_records])




''' Helpful Functions '''

# TODO: Add Helpful Functions (if needed)



''' Nodes '''
def classify_intent(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM. Classifies the incoming WhatsApp message as either 'receipt' (photo with receipt) or 'spending_question' (text query about spending). 
    LLM node (structured output). Classifies the newest inbound user message as either a receipt photo ('receipt') or a text query about spending ('spending_question') using a structured-output call whose schema is a Pydantic model with a single Literal field.
    
    Overview:
        This is the router-decision node of the root graph and runs exactly once per
        inbound event (one WhatsApp-style message). It inspects the newest
        HumanMessage in state['messages'] — its text content and whether it carries
        an image attachment (photo of a receipt) — and asks classify_intent_llm
        (temperature 0) to classify it via STRUCTURED OUTPUT, not free text:
            structured_llm = classify_intent_llm.with_structured_output(IntentClassification)
        where IntentClassification is a Pydantic BaseModel defined in the Agent
        Schema section with a single field:
            intent: Literal['receipt', 'spending_question', 'unknown']
        Because the output is schema-enforced, no free-text parsing or label
        normalization is needed — the node reads result.intent directly. The label
        is written to state['intent'] and consumed by the conditional edge
        'from_classify_intent_to'. This node does NOT process the receipt or answer
        the question; it only classifies and routes, producing no user-visible reply.
    
    Step-by-step:
        1. Retrieve the newest HumanMessage inline (no helper function): scan
           state['messages'] from the end, e.g.
           next((m for m in reversed(state['messages']) if isinstance(m, HumanMessage)), None).
        2. Extract the message text and detect an image attachment (image path or
           base64 payload) from the message content blocks / additional_kwargs.
        3. Format prompts.CLASSIFY_INTENT_PROMPT with the message text and a flag
           indicating whether an image is attached
           (prompt.format(message_text=..., has_image=...)).
        4. Build the structured LLM inside the node:
           structured_llm = classify_intent_llm.with_structured_output(IntentClassification)
           and invoke it via safe_invoke with
           [SystemMessage(content=prompt), HumanMessage(content=message_text)].
           Do NOT include the full conversation history — classification only needs
           the current message.
        5. Read the structured result: intent = result.intent. The Literal-typed
           schema guarantees one of 'receipt' | 'spending_question' | 'unknown' —
           no post-hoc string parsing is required.
        6. Return an updated state dict containing:
           - intent: the classified label,
           - next_action: 'process_receipt' if 'receipt', 'answer_question' if
             'spending_question', 'clarify' if 'unknown',
           - all other existing state keys preserved (messages untouched — this
             node produces no user-visible reply).
        7. On exception: log via the DEBUG print block, set intent='unknown' and
           next_action='clarify', and return the state otherwise unchanged so the
           conditional edge can still route and the user eventually gets a reply.
    
    Inputs (state keys):
        - messages (List[BaseMessage]): ordered conversation log; this node reads
          the newest HumanMessage (text and optional image attachment).
    
    Outputs (state keys):
        - intent (str): 'receipt' | 'spending_question' | 'unknown' — read by the
          conditional function from_classify_intent_to.
        - next_action (str): routing hint persisted via the checkpointer
          ('process_receipt' | 'answer_question' | 'clarify').
        - messages: passed through unchanged.
    
    Tools:
        None. The LLM is used via .with_structured_output(IntentClassification) and
        safe_invoke — a plain structured-output LLM invocation, not a tool call.
    
    Helpful functions:
        None required. The last-user-message lookup and image-attachment detection
        are trivial inline operations (explicitly kept out of helpful functions per
        user feedback).
    """

    print_function_name()
    try:
        # Step 1: Retrieve the newest HumanMessage inline (no helper function).
        last_human: Optional[HumanMessage] = next(
            (m for m in reversed(state.get('messages', [])) if isinstance(m, HumanMessage)),
            None
        )
        if last_human is None:
            return {'intent': 'unknown', 'next_action': 'clarify'}

        # Step 2: Extract the message text from string / multimodal / other content.
        content: Any = last_human.content
        if isinstance(content, str):
            text: str = content
        elif isinstance(content, list):
            text = ' '.join(
                block.get('text', '')
                for block in content
                if isinstance(block, dict) and 'text' in block
            ).strip()
        else:
            text = str(content)

        # Step 2 (cont.): Detect an image attachment — from content blocks
        # (image-typed blocks or image-bearing keys) and/or additional_kwargs.
        has_image: bool = False
        if isinstance(content, list):
            image_types: set = {'image_url', 'image_path', 'image', 'input_image', 'media'}
            image_keys: set = {'image_url', 'image_path', 'image_base64'}
            has_image = any(
                isinstance(block, dict) and (
                    block.get('type') in image_types
                    or any(key in block for key in image_keys)
                )
                for block in content
            )
        additional_kwargs: Dict[str, Any] = getattr(last_human, 'additional_kwargs', {}) or {}
        if any(additional_kwargs.get(key) for key in ('image_path', 'image_base64', 'image', 'image_url')):
            has_image = True

        # Step 3: Format the classification prompt (readable format arguments).
        prompt: str = prompts.CLASSIFY_INTENT_PROMPT.format(
            message_text=text,
            has_image='yes' if has_image else 'no'
        )

        # Step 4: Invoke the module-level structured LLM directly — the
        # .with_structured_output(IntentClassification) wrapping is ALREADY applied
        # at module level (calling it again here would break the structured output).
        # Only the current message is passed — NO conversation history.
        result: IntentClassification = safe_invoke(
            classify_intent_llm,
            messages=[SystemMessage(content=prompt), HumanMessage(content=text)]
        )

        # Step 5: Read the schema-enforced label (ATTRIBUTE access on the Pydantic
        # model — no dict access, no free-text parsing or label normalization).
        intent: str = result.intent

        # Step 6: Map the label to the routing hint for the conditional edge
        # from_classify_intent_to ('unknown' and any unexpected value -> 'clarify').
        if intent == 'receipt':
            next_action: str = 'process_receipt'
        elif intent == 'spending_question':
            next_action = 'answer_question'
        else:
            next_action = 'clarify'

        # Return ONLY the routing keys — messages/latest untouched (this node
        # produces no user-visible reply; the conditional edge routes next).
        return {'intent': intent, 'next_action': next_action}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Never raise out of the node: fall back so the conditional edge can
        # still route and the user eventually gets a reply.
        return {'intent': 'unknown', 'next_action': 'clarify'}


def process_receipt(state: AgentSchema) -> AgentSchema:
    """ Execution: SUBGRAPH. Runs the receipt processing subgraph: OCR, extraction, categorization, and Excel insertion. 
    SUBGRAPH node. Delegates full receipt processing to the 'process_receipt_subgraph' app: OCR via vision model (OpenRouter), structured extraction (cost, date, items with quantity x price currency, location), category determination, and appending the row to the Excel file.
    
    Overview:
        This node is a thin wrapper around process_receipt_subgraph_app and does not
        call any LLM itself. Its responsibilities are: (1) extract the receipt image
        reference from the newest HumanMessage, (2) invoke the subgraph with a clean,
        minimal input dict, (3) map the subgraph's outputs back into the root state,
        and (4) compose the single user-visible confirmation (or failure) reply for
        this run, since the graph transitions to END immediately after this node.
    
    Step-by-step:
        1. Retrieve the newest HumanMessage inline (scan state['messages'] from the
           end for the last HumanMessage) and extract the image reference
           (image_path or image_base64) from its attachment/content blocks.
        2. If no image is found: skip the subgraph, set latest to a short message
           asking the user to attach a receipt photo, append it to messages as an
           AIMessage, set next_action='await_receipt_photo', and return (run ends).
        3. Resolve the Excel target path (state['excel_path'] if present, otherwise
           the default path from environment/config) — the subgraph's
           append_to_excel node writes there.
        4. Invoke the subgraph:
           result = process_receipt_subgraph_app.invoke(
               {'image_path': ..., 'image_base64': ..., 'excel_path': ...},
               config={'configurable': {...}}  # propagate thread_id/user_id if needed
           )
           The subgraph internally runs: ocr_image -> extract_receipt_data ->
           determine_category -> append_to_excel -> end.
        5. Read the subgraph result keys: receipt_data (dict with cost, date, items
           formatted as 'item1 (quantity x price currency), ...', location),
           category (str), and success/error flags (plus ocr_text if exposed).
        6. Compose the user-facing reply: on success, a short confirmation
           summarizing what was saved (e.g., 'Saved receipt from <location> on
           <date> — <cost>. Category: <category>.'); on failure, an apology plus
           the error reason.
        7. Return an updated state dict with:
           - receipt_data, category, ocr_text (if returned by the subgraph),
           - latest: the confirmation/failure text,
           - messages: previous messages + AIMessage(content=latest),
           - next_action: 'done' on success, 'await_receipt_photo' on failure,
           - all other existing keys preserved.
        8. On exception: log via the DEBUG block, set latest to a user-friendly
           failure message, append the AIMessage, set next_action='retry_receipt',
           and return the state (never raise out of the node).
    
    Inputs (state keys):
        - messages (List[BaseMessage]): the newest HumanMessage must carry the
          receipt photo (image_path or base64 attachment).
        - excel_path (Optional[str]): target Excel file path; falls back to the
          configured default when absent.
    
    Outputs (state keys):
        - receipt_data (dict): {cost, date, items, location} as extracted by the
          subgraph (items formatted 'item1 (quantity x price currency), ...').
        - category (str): spending category determined from the items.
        - ocr_text (Optional[str]): raw OCR text, if the subgraph exposes it.
        - latest (str): the single user-visible reply for this run.
        - messages (List[BaseMessage]): history + the new AIMessage(latest).
        - next_action (str): 'done' | 'await_receipt_photo' | 'retry_receipt'.
    
    Tools:
        None in this node. All LLM/tool usage (vision OCR via OpenRouter,
        extraction, categorization) happens inside process_receipt_subgraph_app.
    
    Helpful functions:
        None required. The last-user-message lookup and image extraction are simple
        inline operations.
    """

    print_function_name()
    try:
        # Guard: the subgraph module may be unavailable (its import failed at load time).
        if process_receipt_subgraph_app is None:
            latest: str = 'Sorry, the receipt processor is unavailable right now. Please try again later.'
            return {
                'latest': latest,
                'messages': [AIMessage(content=latest)],
                'next_action': 'retry_receipt',
            }

        # Step 1: Retrieve the newest HumanMessage inline (no helper function).
        last_human: Optional[HumanMessage] = next(
            (m for m in reversed(state.get('messages', [])) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            latest = 'Please attach a photo of the receipt so I can save it for you.'
            return {
                'latest': latest,
                'messages': [AIMessage(content=latest)],
                'next_action': 'await_receipt_photo',
            }

        # Step 2: Extract the image reference (path or base64) from the message.
        image_path: Optional[str] = None
        image_base64: Optional[str] = None
        if isinstance(last_human.content, list):
            for block in last_human.content:
                if not isinstance(block, dict):
                    continue
                if block.get('type') == 'image_url':
                    image_value: Any = block['image_url']
                    image_path = (
                        image_value.get('url')
                        if isinstance(image_value, dict)
                        else image_value
                    )
                if 'image_path' in block:
                    image_path = block['image_path']
                if 'image_base64' in block:
                    image_base64 = block['image_base64']
        if isinstance(last_human.additional_kwargs, dict):
            if 'image_path' in last_human.additional_kwargs:
                image_path = last_human.additional_kwargs['image_path']
            if 'image_base64' in last_human.additional_kwargs:
                image_base64 = last_human.additional_kwargs['image_base64']

        # Keep only str-or-None image references.
        image_path = image_path if isinstance(image_path, str) else None
        image_base64 = image_base64 if isinstance(image_base64, str) else None

        # Step 3: No image reference found -> ask the user to attach a receipt photo.
        if image_path is None and image_base64 is None:
            latest = 'Please attach a photo of the receipt so I can save it for you.'
            return {
                'latest': latest,
                'messages': [AIMessage(content=latest)],
                'next_action': 'await_receipt_photo',
            }

        # Step 4: Resolve the Excel target path (explicit path or configured default).
        excel_path: str = state.get('excel_path') or DEFAULT_EXCEL_PATH

        # Step 5: Build the minimal subgraph input (only non-None image values) and invoke it.
        subgraph_input: Dict[str, Any] = {'excel_path': excel_path}
        if image_path is not None:
            subgraph_input['image_path'] = image_path
        if image_base64 is not None:
            subgraph_input['image_base64'] = image_base64

        result: Any = process_receipt_subgraph_app.invoke(subgraph_input)

        # Step 6: Defensively read the subgraph outputs.
        result = result if isinstance(result, dict) else {}
        receipt_data: Optional[Dict[str, Any]] = result.get('receipt_data')
        category: Optional[str] = result.get('category')
        ocr_text: Optional[str] = result.get('ocr_text')
        error: Optional[str] = result.get('error')
        success: Any = result.get('success')
        if 'success' not in result:
            success = (receipt_data is not None) and not error

        # Step 7: Compose the single user-visible reply for this run.
        if success:
            location: str = (receipt_data or {}).get('location') or 'unknown location'
            date: str = (receipt_data or {}).get('date') or 'unknown date'
            cost: Any = (receipt_data or {}).get('cost', '?')
            latest = f'Saved receipt from {location} on {date} — {cost}.'
            if category:
                latest += f' Category: {category}.'
        else:
            latest = 'Sorry, I could not process that receipt.' + (f' Reason: {error}' if error else '')

        # Step 8: Return the state update (the graph transitions to END right after).
        return {
            'receipt_data': receipt_data,
            'category': category,
            'ocr_text': ocr_text,
            'latest': latest,
            'messages': [AIMessage(content=latest)],
            'next_action': 'done' if success else 'await_receipt_photo',
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        latest = 'Sorry, something went wrong while saving your receipt. Please try again.'
        return {
            'latest': latest,
            'messages': [AIMessage(content=latest)],
            'next_action': 'retry_receipt',
        }


def answer_spending_question(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Answers spending questions by querying the Excel file (e.g., monthly totals, top items) and generating a natural-language response. 
    LLM+TOOLS node. Answers natural-language questions about the user's spending (e.g., 'How much did I spend on groceries this month?', 'What are my top 3 most expensive items?') by querying the Excel receipt file with bound tools; it can also append a receipt dictated in plain text and filter already-retrieved records, then generates one natural-language reply.
    
    Overview:
        This node runs answer_spending_question_llm (temperature 0) bound to three
        Excel tools via .bind_tools([...]). The LLM decides which tools to call
        (almost always query_receipts_excel first); tool results come back as
        ToolMessages and the LLM grounds its final answer strictly in those records.
        It can chain calls (e.g., query -> filter_receipt_records for a 'top 3'
        follow-up on retrieved records) and can save a receipt the user dictated as
        text (no photo) via append_receipt_to_excel. This node produces the single
        user-visible reply for the run; the graph transitions to END immediately
        after. It never invents numbers — every figure must come from tool output.
    
    Step-by-step:
        1. Retrieve the newest HumanMessage (the user's message) inline (scan
           state['messages'] from the end for the last HumanMessage) and resolve
           the Excel path (state['excel_path'] or the configured default).
        2. Format prompts.ANSWER_SPENDING_QUESTION_PROMPT with the current date (so
           relative ranges like 'this month' resolve correctly) and the excel_path.
        3. Bind the tools:
           llm_with_tools = answer_spending_question_llm.bind_tools(
               [query_receipts_excel, append_receipt_to_excel, filter_receipt_records])
        4. Invoke llm_with_tools with [SystemMessage(content=prompt)] + the recent
           conversation messages (so follow-ups like 'and last month?' work).
        4b. If the newest message is a ToolMessage produced by an
           append_receipt_to_excel call, the append is terminal: compose the final
           confirmation/failure reply directly from the ToolMessage content and the
           original tool-call arguments, set next_action=done and return WITHOUT
           invoking the LLM or any further tool.
        5. Tool loop (bounded, max ~4 iterations): if the AIMessage contains
           tool_calls, execute them ToolNode-style — run each requested tool, append
           each result as a ToolMessage to the message list — then re-invoke
           llm_with_tools with the updated messages until a final AIMessage with no
           tool_calls is produced. Typical chains: query_receipts_excel ->
           filter_receipt_records (top-N / refined follow-ups on already-retrieved
           records) -> final answer; or append_receipt_to_excel when the user
           dictates a receipt in text and asks to save it.
        6. Post-process the final AIMessage with clean_llm_output. The answer must
           be computed ONLY from the returned records — never invent numbers. If
           the records are empty, reply that no matching receipts were found.
        7. Return an updated state dict with:
           - latest: the final answer text,
           - messages: previous messages + all intermediate AIMessage/ToolMessage
             exchanges + the final AIMessage,
           - next_action: 'done',
           - all other existing keys preserved.
        8. On exception: log via the DEBUG block, set latest to a friendly failure
           message, append the AIMessage, set next_action='done', and return state.
    
    Inputs (state keys):
        - messages (List[BaseMessage]): conversation log; the newest HumanMessage is
          the spending question (or a text-described receipt to save); earlier
          messages give context for follow-up questions.
        - excel_path (Optional[str]): path to the receipts Excel file with columns
          cost, date, items, location, category; falls back to the configured
          default when absent.
    
    Outputs (state keys):
        - latest (str): the single user-visible natural-language answer.
        - messages (List[BaseMessage]): history + tool-loop AIMessage/ToolMessage
          exchanges + final AIMessage(latest).
        - next_action (str): always 'done' after this node (terminal branch).
    
    Tools (bound via .bind_tools; NOTE: tool calls are executed by a ToolNode or
    an equivalent inline tool-execution loop — the ToolNode runs the requested
    tools, appends their results as ToolMessages, and control returns to this node
    to continue reasoning):
        - query_receipts_excel(excel_path: str, category: Optional[str] = None,
          date_from: Optional[str] = None, date_to: Optional[str] = None,
          item_keyword: Optional[str] = None) -> dict: read-only tool that loads
          the Excel file (pandas/openpyxl), optionally filters rows by category,
          inclusive date range, and item keyword, and returns a compact
          JSON-serializable dict: {'records': [{cost, date, items, location,
          category}, ...], 'count': int, 'total_cost': float}.
        - append_receipt_to_excel(excel_path: str, cost: float, date: str,
          items: str, location: str, category: str) -> dict: appends one row to
          the Excel file (creating it with the standard columns if missing) and
          returns {'success': bool, 'row_index': int, 'error': Optional[str]}.
          Used when the user dictates a receipt in text (no photo) and asks to
          save it.
        - filter_receipt_records(records_json: str, category: Optional[str] = None,
          date_from: Optional[str] = None, date_to: Optional[str] = None,
          item_keyword: Optional[str] = None, top_n: Optional[int] = None,
          sort_by: Optional[str] = None) -> dict: pure in-memory filter/sort over
          records the LLM already retrieved (passed as a JSON string) — used when
          a follow-up question can be answered from previously fetched records
          without re-reading the file; returns {'records': [...], 'count': int}.
    
    Helpful functions:
        None required. The last-user-message lookup is a trivial inline operation;
        the tools themselves handle all Excel I/O and record filtering.
    """

    print_function_name()
    try:
        # <append-terminal short-circuit> — inspect the newest message BEFORE any
        # prompt resolution or LLM invocation: an append_receipt_to_excel result
        # is TERMINAL and must never reach the LLM again.
        messages: List[BaseMessage] = list(state.get('messages', []))
        last: Optional[BaseMessage] = messages[-1] if messages else None

        if isinstance(last, ToolMessage):
            # Search BACKWARDS for the AIMessage that requested this tool result
            # (tool_calls entries are dicts with keys 'name', 'args', 'id').
            matching_call: Optional[Dict[str, Any]] = None
            for message in reversed(messages[:-1]):
                if not isinstance(message, AIMessage):
                    continue
                message_tool_calls: Any = getattr(message, 'tool_calls', None)
                if not isinstance(message_tool_calls, list):
                    continue
                matching_call = next(
                    (
                        call for call in message_tool_calls
                        if isinstance(call, dict) and call.get('id') == last.tool_call_id
                    ),
                    None,
                )
                if matching_call is not None:
                    break

            # The append has ALREADY executed and is TERMINAL: compose the final
            # user-visible reply directly — no LLM call, no further tool calls.
            if matching_call is not None and matching_call.get('name') == 'append_receipt_to_excel':
                print(f'{BLUE}[NODE] [INFO]{RESET} append-terminal short-circuit engaged') if DEBUG else None

                # Parse the ToolMessage content (fall back to {} on any failure).
                try:
                    append_result: Any = json.loads(last.content)
                    if not isinstance(append_result, dict):
                        append_result = {}
                except Exception:
                    append_result = {}

                # Original tool-call arguments of the append request.
                call_args: Any = matching_call.get('args', {})
                call_args = call_args if isinstance(call_args, dict) else {}

                if append_result.get('success'):
                    cost: Any = call_args.get('cost')
                    date: Any = call_args.get('date')
                    location: Any = call_args.get('location')
                    category: Any = call_args.get('category')
                    cost = cost if cost is not None else 'unknown'
                    date = date if date is not None else 'unknown'
                    location = location if location is not None else 'unknown'
                    category = category if category is not None else 'unknown'
                    latest: str = (
                        f'Saved receipt: {cost} at {location} on {date}. '
                        f'Category: {category}.'
                    )
                else:
                    error: Any = append_result.get('error')
                    if error:
                        latest = f'Sorry, I could not save that receipt. Reason: {error}.'
                    else:
                        latest = 'Sorry, I could not save that receipt.'

                # Terminal return: the final AIMessage carries no tool_calls, so the
                # existing conditional edge routes straight to END.
                return {
                    'latest': latest,
                    'messages': [AIMessage(content= latest)],
                    'next_action': 'done',
                }

        # <bounded-loop counter> — the loop counter for the
        # answer_spending_question <-> tools loop (max MAX_TOOL_ITERATIONS
        # ToolNode executions per run). Read from the persisted state and
        # sanitized; reset on the first pass of this branch in the current run.
        tool_iterations: int = state.get('tool_iterations', 0)
        if not isinstance(tool_iterations, int) or isinstance(tool_iterations, bool):
            tool_iterations = 0
        # Reset on the first pass of this branch in the current run: the newest
        # message is still the inbound HumanMessage, so stale checkpointer values
        # from previous runs never leak into this run's bound.
        if isinstance(last, HumanMessage):
            tool_iterations = 0

        # <preprocess> — resolve the Excel target path (explicit state value or the
        # configured default) and the current ISO date so relative ranges like
        # 'this month' resolve correctly inside the tools.
        excel_path: str = state.get('excel_path') or DEFAULT_EXCEL_PATH
        today_iso: str = datetime.now().strftime('%Y-%m-%d')

        # The system prompt carries ONLY the current date + excel_path; the full
        # conversation history (which on later passes already contains the
        # intermediate AIMessage(tool_calls)/ToolMessage exchanges appended by the
        # ToolNode) is passed separately — never formatted into the prompt too.
        prompt: str = prompts.ANSWER_SPENDING_QUESTION_PROMPT.format(current_date= today_iso, excel_path= excel_path)

        # Exactly ONE LLM invocation per node run: the module-level
        # answer_spending_question_llm is ALREADY bound to the three tools, so it
        # is invoked directly — never re-bound, never executed, never looped here.
        history: List[BaseMessage] = list(state.get('messages', []))
        response: BaseMessage = safe_invoke(answer_spending_question_llm, messages= [SystemMessage(content= prompt)] + history)

        # <postprocess> — intermediate step: the LLM requested tools. Return ONLY
        # the AIMessage so the conditional edge (from_answer_spending_question_to)
        # routes it to the ToolNode; do NOT set latest/next_action yet. The graph
        # re-enters this node with the ToolMessages added to the history.
        tool_calls: Optional[List[Dict[str, Any]]] = getattr(response, 'tool_calls', None)
        if tool_calls:
            # Bounded-loop cap check: the counter is incremented exactly once per
            # AIMessage carrying tool_calls (one ToolNode execution per increment),
            # so once tool_iterations reaches MAX_TOOL_ITERATIONS no further
            # ToolNode execution can ever happen — the loop is guaranteed to
            # terminate within a bounded number of executions per run.
            if tool_iterations >= MAX_TOOL_ITERATIONS:
                # The unexecuted tool-call AIMessage (`response`) is deliberately
                # DISCARDED: an AIMessage carrying tool_calls that is never followed
                # by matching ToolMessages would break the LLM invocation of any
                # later turn, so it must never enter the persisted history.
                reply: str = (
                    'Sorry, I could not complete that request within the allowed '
                    'number of data lookups. Please rephrase or narrow your question '
                    'and try again.'
                )

                # Ground the fallback reply in the most recent ToolMessage if
                # possible: scan backwards for the first ToolMessage whose content
                # parses into a dict carrying an int 'count'.
                for message in reversed(messages):
                    if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
                        continue
                    try:
                        payload: Any = json.loads(message.content)
                    except Exception:
                        continue

                    if not isinstance(payload, dict) or not isinstance(payload.get('count'), int):
                        continue

                    count: int = payload['count']
                    total_cost: Any = payload.get('total_cost')
                    if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool):
                        reply = (
                            f'I found {count} matching receipts totalling {total_cost}, '
                            'but I reached the limit of follow-up lookups for this run. '
                            'Please rephrase or narrow your question (e.g. by date range '
                            'or category) if you need more detail.'
                        )
                    else:
                        reply = (
                            f'I found {count} matching receipts, but I reached the '
                            'limit of follow-up lookups for this run. Please rephrase '
                            'or narrow your question if you need more detail.'
                        )
                    break

                # Force-finalize WITHOUT returning 'tool_iterations': the
                # reset-on-HumanMessage logic handles staleness on the next run.
                return {
                    'latest': reply,
                    'next_action': 'done',
                    'messages': [AIMessage(content= reply)]
                }

            # Under the cap: hand the tool-call AIMessage to the conditional edge
            # (which routes it to the ToolNode) and increment the counter by
            # exactly one — one ToolNode execution per increment.
            return {'messages': [response], 'tool_iterations': tool_iterations + 1}

        # Final answer: ground it strictly in the tool outputs (per the system
        # prompt), clean it, and publish the single user-visible reply.
        final_answer: str = clean_llm_output(response.content)
        return {
            'latest': final_answer,
            'next_action': 'done',
            'messages': [AIMessage(content= final_answer)]
        }
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Never raise out of the node — return a friendly failure reply instead.
        latest: str = 'Sorry, I had trouble answering that. Please try again.'
        return {
            'latest': latest,
            'next_action': 'done',
            'messages': [AIMessage(content= latest)]
        }





def clarify(state: AgentSchema) -> AgentSchema:
    """ Execution: STATIC REPLY (no LLM, no tools). Asks the user to rephrase when the newest inbound message could not be classified ('unknown' intent fallback).
    No-LLM terminal node for the 'unknown'-intent fallback path. When classify_intent's structured classification fails or returns 'unknown', the conditional edge from_classify_intent_to routes here instead of answer_spending_question, so an unclassifiable message gets a clarification request rather than an unrelated spending answer.

    Overview:
        Runs exactly once per misclassified inbound message and produces the single
        user-visible reply for the run; the graph transitions to END immediately
        after (conversational node contract). It performs NO LLM call and NO tool
        call: the reply is a static, friendly clarification text.

    Step-by-step:
        1. Compose the clarification text (static string, no LLM): briefly say the
           message was not understood, ask the user to rephrase, and list the two
           supported abilities — (a) send a photo of a receipt to save it, and
           (b) ask a question about spending (e.g. 'How much did I spend on
           groceries this month?').
        2. Return {'latest': <text>, 'messages': [AIMessage(content=<text>)],
           'next_action': 'done'} — all other state keys are preserved by being
           absent from the returned dict.
        3. On exception: log via the DEBUG print block (RED [NODE] [ERR] prefix +
           traceback.print_exc()) and return the same shape with a generic
           clarification text — never raise out of the node.

    Inputs (state keys):
        - messages: untouched (the reply is static; no message parsing needed).

    Outputs (state keys):
        - latest (str): the clarification prompt (single user-visible reply).
        - messages (List[BaseMessage]): history + the new AIMessage(latest).
        - next_action (str): always 'done' (terminal node).

    Tools:
        None.

    Helpful functions:
        print_function_name() at the top of the body, as in the other nodes.
    """
    print_function_name()
    try:
        # Static clarification text — no LLM call, no tool call, no state access.
        text: str = (
            "Sorry, I didn't quite understand that — could you rephrase? "
            "You can send a photo of a receipt to save it, or ask a question "
            "about your spending (e.g. 'How much did I spend on groceries this month?')."
        )
        return {
            'latest': text,
            'messages': [AIMessage(content= text)],
            'next_action': 'done',
        }
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Generic fallback clarification — never raise out of the node.
        text = (
            "Sorry, I couldn't process that. Please rephrase, send a photo of a "
            "receipt to save it, or ask a question about your spending."
        )
        return {
            'latest': text,
            'messages': [AIMessage(content= text)],
            'next_action': 'done',
        }




''' Conditional Functions '''
def from_classify_intent_to(state: AgentSchema) -> Literal["process_receipt", "answer_spending_question", "clarify"]:
    """Conditional edge after classify_intent: routes 'receipt' intents to process_receipt, 'spending_question' intents to answer_spending_question, and everything else (the 'unknown'/clarify fallback from classify_intent) to the clarify node, so unclassifiable messages get a clarification request instead of an unrelated spending answer."""
    print_function_name()
    intent: str = state.get('intent', 'unknown')
    if intent == 'receipt':
        return "process_receipt"
    if intent == 'spending_question':
        return "answer_spending_question"
    return "clarify"


def from_answer_spending_question_to(state: AgentSchema) -> Literal["tools", "answer_done"]:
    """Route tool calls to the ToolNode or finish the run.

    Conditional edge after the 'answer_spending_question' node. It inspects the
    newest message in state['messages']:
      - If it is an AIMessage carrying a non-empty tool_calls list, return 'tools'
        (routes to the ToolNode named 'tools', which executes the bound tools
        query_receipts_excel, append_receipt_to_excel and filter_receipt_records,
        appends their results as ToolMessages, and the graph then loops back to
        'answer_spending_question').
      - Otherwise (a final AIMessage with no tool calls, a HumanMessage or
        ToolMessage, or an empty history) return 'answer_done', which maps to END
        in the graph's conditional-edge mapping — the node has already written
        state['latest'] and next_action='done' for the final reply.
    Always returns exactly one of the two route strings — never None — and never
    raises (any unexpected error falls back to 'answer_done').
    """
    print_function_name()
    try:
        # Dict-style access — AgentSchema is a MessagesState (TypedDict).
        messages: List[BaseMessage] = state.get('messages', [])
        last: Optional[BaseMessage] = messages[-1] if messages else None

        # Route to the shared ToolNode ONLY for an AIMessage whose tool_calls is a
        # non-empty list — this covers ALL bound tools, including
        # append_receipt_to_excel (there is no separate handler node in this graph).
        tool_calls: Any = getattr(last, 'tool_calls', None)
        if isinstance(last, AIMessage) and isinstance(tool_calls, list) and tool_calls:
            return "tools"

        # Final AIMessage without tool calls, a HumanMessage/ToolMessage, or an
        # empty/malformed history -> the run is done (maps to END).
        return "answer_done"
    except Exception as e:
        # A conditional edge must never raise — fall back to END.
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        return "answer_done"



''' Graph '''
personal_receipt_agent_graph = StateGraph(AgentSchema)

personal_receipt_agent_graph.add_node("classify_intent", classify_intent)
personal_receipt_agent_graph.add_node("process_receipt", process_receipt)
personal_receipt_agent_graph.add_node("answer_spending_question", answer_spending_question)
personal_receipt_agent_graph.add_node("clarify", clarify)
personal_receipt_agent_graph.add_node("tools", ToolNode([query_receipts_excel, append_receipt_to_excel, filter_receipt_records]))

personal_receipt_agent_graph.add_edge(START, "classify_intent")
personal_receipt_agent_graph.add_conditional_edges(
    "classify_intent",
    from_classify_intent_to,
    {   # Not needed just for clarity
        "process_receipt": "process_receipt",
        "answer_spending_question": "answer_spending_question",
        "clarify": "clarify",
    }
)
personal_receipt_agent_graph.add_edge("process_receipt", END)
personal_receipt_agent_graph.add_edge("clarify", END)
personal_receipt_agent_graph.add_conditional_edges(
    "answer_spending_question",
    from_answer_spending_question_to,
    {
        "tools": "tools",
        "answer_done": END,
    },
)
personal_receipt_agent_graph.add_edge("tools", "answer_spending_question")


personal_receipt_agent_app = personal_receipt_agent_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    import uuid

    os.environ['EXCEL_FILE_PATH'] = str(Path(__file__).resolve().parent / 'receipts.xlsx')

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'no_req_eng_test',
            'run_name': 'no_req_eng_test',
            'thread_id': f'no_req_eng_test:{uuid.uuid4()}',
        }
    }

    print(
        'Commands:\n'
        '  re:<path>          Process a receipt image\n'
        '  q                  Quit\n'
        '  anything else is sent as a normal conversation message\n'
    )

    user_in = input(f'{GREEN}[USER INPUT]{RESET} > ')

    while user_in.lower() != 'q':

        if user_in.startswith('re:'):
            receipt_input = user_in[3:].strip()

            image_path = receipt_input.strip()

            human_message = HumanMessage(
                content='Receipt photo attached.',
                additional_kwargs={
                    'image_path': image_path
                }
            )

        else:
            human_message = HumanMessage(content=user_in)

        response = personal_receipt_agent_app.invoke(
            {
                'messages': [human_message]
            },
            config=config,
        )

        messages = response.get('messages', [])

        last_human_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_index = i
                break

        for message in messages[last_human_index + 1:]:
            if message.content:
                print(f'\n{BLUE}[ANSWER]{RESET} {message.content}')
            else:
                print(f'\n{BLUE}[ANSWER]{RESET} {message}')

        if response.get('latest'):
            print(f'\n{GREEN}[LATEST]{RESET} {response["latest"]}')

        user_in = input(f'\n{GREEN}[USER INPUT]{RESET} > ')