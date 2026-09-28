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
from experiments.ablation_study.full_framework import whatsapp_receipt_expense_tracker_prompts as prompts

from openpyxl import load_workbook, Workbook
from datetime import datetime, timedelta
from pydantic import ValidationError
import base64
import csv
import math
import re
import requests



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} whatsapp_receipt_expense_tracker') if DEBUG else None



""" Schemas """

class ReceiptItem(BaseModel):
    """
    General Schema (tool I/O). One line item OCR'd from a receipt photo by the extract_receipt_data tool (exactly one OpenRouter vision call per photo). It is an element of ExtractedReceipt.items and is rendered into the Excel 'items' column token 'name (qty x price currency)' via format_item() (e.g. 'Milk (2 x 1.19 EUR)'). quantity supports fractional amounts (e.g. 0.5 kg of produce); unit_price is the per-unit price; currency is the ISO code as printed on the receipt and may be non-EUR — only the receipt TOTAL is converted to EUR via the convert_to_eur tool, line items keep their printed currency in the items string.
    """
    name: str # Item name exactly as printed on the receipt (used verbatim in the items string).
    quantity: float # Units purchased; float to support fractional amounts like 0.5 kg of produce.
    unit_price: float # Price per single unit, in `currency`.
    currency: str # ISO-4217 code as printed on the receipt (e.g. 'EUR', 'USD'); may be non-EUR — only the receipt TOTAL is converted to EUR via the convert_to_eur tool, line items keep their printed currency in the items string.

    def format_item(self) -> str:
            """
        Return the canonical items-column token for this line item: f'{name} ({quantity} x {unit_price} {currency})', e.g. 'Milk (2 x 1.19 EUR)'. Composed by ExtractedReceipt.format_items().
            """
            return f'{self.name} ({self.quantity} x {self.unit_price} {self.currency})'


class ExtractedReceipt(BaseModel):
    """
    General Schema (tool I/O). Structured extraction of ONE receipt by extract_receipt_data; the tool returns {'receipts': [ExtractedReceipt, ...]} — one entry per receipt when a single photo contains several, and each entry becomes exactly one Excel row (one row per receipt). The chat node applies validate-before-write: needs_confirmation() is True when confidence=='low' or a required field (cost/date/items) is missing — then it asks ONE combined confirmation question, writes nothing, and the correction is applied in the next run from history; otherwise it converts non-EUR totals via the convert_to_eur tool and appends via append_expense_rows. 'date' is deterministic 'YYYY-MM-DD HH:MM' — HH:MM from the receipt when printed, otherwise the time the photo was sent; date falls back to today. 'category' is picked from the known categories pre-loaded from the .xlsx into the chat prompt (reuse an existing one whenever it fits; create a new one only when the receipt clearly fits none — prevents near-duplicates like 'Supermarket' vs 'Groceries'). 'currency' is the original receipt currency; when it is not EUR the convert_to_eur tool is called and this original currency must be recorded in the row's comments (note format documented on ExpenseRow.comments).
    """
    total_cost: Optional[float] # Total amount paid INCLUDING tax, in `currency`; None when unreadable (missing required field → confirmation required before any write).
    currency: Optional[str] # Original receipt currency (ISO-4217); None when unreadable. When not 'EUR', the convert_to_eur tool is applied and this original currency must be recorded in the row's comments (e.g. 'Original: 12.50 USD ...').
    date: Optional[str] # Deterministic 'YYYY-MM-DD HH:MM'; HH:MM from the receipt when printed, otherwise the time the photo was sent; date falls back to today. None only when the receipt is unparseable (→ confirmation).
    items: List[ReceiptItem] # Line items with qty/unit price/currency (sub-schema ReceiptItem); defaults to an empty list when nothing could be read (empty = missing required field → confirmation). Rendered to the Excel items string via format_items().
    location: str # Merchant location as printed on the receipt; '' when not present (not a required field for writing a row). Defaults to ''.
    category: str # Suggested category, chosen from the known categories pre-loaded from the .xlsx into the chat prompt; reuse an existing category whenever it fits, create a new one only when the receipt clearly fits none (prevents near-duplicates like 'Supermarket' vs 'Groceries').
    confidence: Literal['high', 'low'] # The vision model's self-assessed confidence flag; 'low' (or any missing required field) forces the validate-before-write confirmation flow (see needs_confirmation()).

    def format_items(self) -> str:
        """
    Build the exact Excel 'items' string: ', '.join(item.format_item() for item in items) → 'item1 (qty x price currency), item2 (qty x price currency), ...'. Returns '' when items is empty (missing required field → confirmation flow).
        """
        return ', '.join(item.format_item() for item in self.items)

    def missing_required_fields(self) -> List[str]:
            """
        Return the names of missing required fields among cost/date/items: 'cost' when total_cost is None, 'date' when date is None, 'items' when items is empty. Empty list means all required fields are present.
            """
            missing: List[str] = []
            if self.total_cost is None:
                missing.append('cost')
            if not self.date:
                missing.append('date')
            if not self.items:
                missing.append('items')
            return missing

    def needs_confirmation(self) -> bool:
        """
    True when confidence == 'low' OR missing_required_fields() is non-empty — i.e. no row may be written yet and the agent must ask the user to confirm or correct before append_expense_rows is called (in the next run, resumed from history).
        """
        return self.confidence == 'low' or bool(self.missing_required_fields())

    def to_expense_row(self, cost_eur: float, comments: str) -> 'ExpenseRow':
        """
        Map this extraction to an Excel-ready ExpenseRow. cost_eur must be the final EUR cost — pass the eur_amount returned by the convert_to_eur tool when the receipt currency was not EUR (and include the original-currency note in comments, e.g. 'Original: 12.50 USD (rate 0.9234, 2025-01-15, source: csv_fallback)'), or total_cost when it was already EUR. comments carries user/agent notes. Uses format_items() for the items string and keeps location/category as extracted.
        """
        return ExpenseRow(
            cost= float(cost_eur),
            date= self.date if self.date else datetime.now().strftime('%Y-%m-%d %H:%M'),
            items= self.format_items(),
            location= self.location or '',
            category= self.category or '',
            comments= comments or '',
        )


class ExpenseRow(BaseModel):
    """
    General Schema (tool I/O + data shape). One row of the local Excel file (default ./expenses.xlsx, created on first use with headers cost|date|items|location|category|comments). Defines the dict shape of each element of append_expense_rows' `rows` argument (validate with ExpenseRow.model_validate / pass model_dump()) and the shape of the updated row returned by update_row_comment. The sheet is append-only — existing rows are never modified except the comments column via update_row_comment. 'cost' is the final EUR amount including tax (already converted when the receipt was in another currency); 'date' is 'YYYY-MM-DD HH:MM' in the machine's configured local timezone; 'items' is the formatted string built by ExtractedReceipt.format_items(); 'comments' holds free-text notes from the user or the agent — the agent logs the original (pre-conversion) receipt currency there whenever a conversion was applied, and users add/amend notes on request.
    """
    cost: float # Final EUR amount paid including tax; already converted via the convert_to_eur tool when the receipt was in another currency.
    date: str # Deterministic 'YYYY-MM-DD HH:MM' in the machine's configured local timezone.
    items: str # 'item1 (qty x price currency), item2 (qty x price currency), ...' — build with ExtractedReceipt.format_items() so the format is always spec-compliant.
    location: str # Merchant location; '' when unknown. Defaults to ''.
    category: str # Deduplicated category (existing category reused from the spreadsheet, new only when nothing fits).
    comments: str # Free-text notes from the user or the agent; the agent logs the original (pre-conversion) receipt currency here whenever a conversion was applied — e.g. 'Original: 12.50 USD (rate 0.9234, 2025-01-15, source: csv_fallback)', built from the dict returned by the convert_to_eur tool (from_currency, original_amount, eur_amount, rate, rate_date, source 'api'|'csv_fallback'); users add/amend notes via update_row_comment. Defaults to ''.

    def to_row_values(self) -> List[Union[float, str]]:
            """
        Return [cost, date, items, location, category, comments] — the values in the exact header order of expenses.xlsx (cost|date|items|location|category|comments) for openpyxl's append-only ws.append(...).
            """
            return [self.cost, self.date, self.items, self.location, self.category, self.comments]


class AgentSchema(MessagesState):
    """
    AgentSchema is the sole LangGraph state schema for the whatsapp_receipt_expense_tracker graph (StateGraph(AgentSchema), compiled with a MemorySaver checkpointer so state persists across runs of the same thread). Per user feedback it subclasses MessagesState with NO declared arguments — MessagesState already provides the `messages` key (Annotated[List[BaseMessage], add_messages]) with the append reducer, so `messages` must NOT be redeclared here; if the framework ever rejects a field-less subclass, add a single dummy argument (e.g. `placeholder: Optional[str] = None`) and nothing else. There are NO extra routing/state keys — next_action, pending_question, pending_extractions and latest were all removed per user feedback; the reactive_conversational flow needs nothing else because the single 'chat' node is a plain tool-bound LLM invocation and the multi-turn confirmation flow resumes purely from conversation history (the previous run's AIMessage confirmation question and the new HumanMessage answer both live in messages).
    
    `messages` (inherited from MessagesState) — INPUT: the newest HumanMessage is this run's single inbound WhatsApp-bridged message (one message per run, batch I/O, no streaming) containing receipt image file path(s) and/or free text; earlier HumanMessages/AIMessages/ToolMessages provide full context, including any open confirmation question. OUTPUT: the chat node appends ToolMessage(s) (one per executed tool call: extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses) and exactly one final AIMessage — the single user-visible English reply (append summary, aggregate answer, comment-update confirmation, combined confirmation question, or normal chat). The add_messages reducer appends returned messages to the persisted history instead of replacing it, which is what carries the confirmation question across runs. On exception the node returns state unchanged.
    """
    pass


''' Tools '''
@tool
def extract_receipt_data(image_path: str, user_text: Optional[str] = None) -> Dict[str, Union[List[ExtractedReceipt], str]]:
    """
    Overview: OCRs ONE receipt photo and returns the structured extraction for every receipt found in that photo — exactly ONE OpenRouter vision-model call per photo (cost-efficient; multiple receipts in one photo become separate entries, and each entry later becomes exactly one Excel row). For each receipt it extracts: total cost including tax with the original ISO-4217 currency; purchase date & time as a deterministic 'YYYY-MM-DD HH:MM' string (HH:MM taken from the receipt when printed, otherwise the time the photo was sent — approximated by the tool's processing time in the machine's local timezone; the date falls back to today); line items as ReceiptItem-shaped entries (name, quantity float supporting fractional amounts like 0.5 kg, unit_price, currency exactly as printed); the merchant location ('' when not printed); a suggested category chosen from the known categories already stored in the spreadsheet (the tool reads them itself via the load_known_categories() helper so it reuses an existing category whenever the receipt fits and creates a new one only when the receipt clearly fits none — preventing near-duplicates like 'Supermarket' vs 'Groceries'); and a self-assessed confidence flag 'high'|'low'. Every entry matches the provided ExtractedReceipt schema (total_cost: Optional[float], currency: Optional[str], date: Optional[str] 'YYYY-MM-DD HH:MM', items: List[ReceiptItem], location: str, category: str, confidence: Literal['high', 'low']). This is a NON-TERMINAL context-updater tool: it writes NOTHING to Excel and converts NOTHING — after it runs, control returns to the caller LLM, which decides per validate-before-write (ExtractedReceipt.needs_confirmation()) whether to convert non-EUR totals (convert_to_eur) and append rows (append_expense_rows), or to ask ONE combined confirmation question and write nothing (when confidence is 'low' or cost/date/items are missing).
    
    Caller LLM: chat_llm — the single tool-bound LLM of the chat node (myChatOpenAI, temperature=0), bound via chat_llm.bind_tools([extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses]). The OpenRouter vision-model call happens INSIDE this tool as an implementation detail; it is not a separate bound tool and is not visible to chat_llm.
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None — the ToolNode executes the function and appends the returned dict as a ToolMessage to state['messages']; the caller LLM then continues reasoning in the same run (convert_to_eur for non-EUR totals, append_expense_rows for confident/confirmed rows, or a single combined confirmation question whose AIMessage is the run's only outbound reply — the pending data then lives in the conversation history and the next run resumes from it, since no extra state keys exist). No handler-side state mutation beyond that automatic messages append.
    
    Inside-the-Tool Work (Tool Responsibilities): Validates the image path; loads and base64-encodes the image; calls load_known_categories() (strictly read-only) to seed the vision prompt with the current deduplicated spreadsheet categories; builds ONE vision prompt (image + optional user_text + categories + extraction contract incl. the date/time fallback rules and the one-entry-per-receipt rule); makes exactly ONE OpenRouter vision call for the photo; parses the vision output into ExtractedReceipt-shaped entries (items default [], location '', total_cost/currency/date None when unreadable, confidence 'high'|'low'); returns the JSON-serializable dict. Side effects: none on local files (never creates or modifies expenses.xlsx or conversions.csv); one outbound HTTPS call to OpenRouter.
    
    Instructions: 1. Validate image_path: a non-empty string pointing to an existing, readable image file; on failure return {'receipts': [], 'error': '<reason>'} — never raise. 2. Read the known categories with load_known_categories() so the suggested category reuses existing spreadsheet categories (an empty list on first use means the vision model may create the first category). 3. Base64-encode the image and compose ONE vision prompt containing: the image, the optional user_text as context, the known categories list, and the extraction contract — total cost INCLUDING tax + ISO currency; date 'YYYY-MM-DD HH:MM' with HH:MM from the receipt when printed, otherwise the current local time (photo-send time), date fallback today; line items with name/quantity/unit_price/currency exactly as printed (fractional quantities allowed); merchant location ('' when absent); suggested category from the provided list, a new one only when nothing fits; confidence 'high'|'low'; ONE entry per receipt when several appear in the same photo. 4. Make exactly ONE OpenRouter vision-model call for this photo (never one call per receipt). 5. Parse the model output into entries matching the provided ExtractedReceipt schema field names and types exactly (total_cost Optional[float], currency Optional[str], date Optional[str] 'YYYY-MM-DD HH:MM', items List[ReceiptItem] each with name str / quantity float / unit_price float / currency str, location str, category str, confidence Literal['high', 'low']); keep unreadable required fields as None and items as [] so the caller's validate-before-write check (needs_confirmation) triggers. 6. Return {'receipts': [<ExtractedReceipt-shaped entries>]}; do NOT convert currency, do NOT write to Excel, do NOT ask the user anything — those are the caller LLM's next steps in the same run.
    
    State Updates (on the caller function): None — a tool cannot see or mutate graph state, and its arguments come only from the LLM's tool call. The ToolMessage built from the return value is appended to state['messages'] automatically by the ToolNode; the caller LLM uses it in the same run to either append rows or send the single combined confirmation question (which, if sent, is the run's one outbound AIMessage; the pending data then lives in the conversation history and the next run resumes from it — no extra state keys exist).
    
    Args: image_path (str): Local file path of the receipt photo (the user bridges WhatsApp to this file-path-based interface themselves); required. user_text (Optional[str]): Optional free text the user sent with the photo (e.g. 'team lunch'), passed to the vision model as context; default None.
    
    Returns: Dict[str, Union[List[ExtractedReceipt], str]] — success: {'receipts': [ExtractedReceipt, ...]} with one schema-shaped entry per receipt found in the photo; failure: {'receipts': [], 'error': '<reason>'}.
    """
    try:
        # (1) Validate image_path: a non-empty string pointing to an existing, readable file.
        if not isinstance(image_path, str) or not image_path or not Path(image_path).is_file():
            return {'receipts': [], 'error': f'image file not found: {image_path}'}

        # (2) Load + base64-encode the image; resolve the mime type from the file suffix.
        image_file: Path = Path(image_path)
        data: bytes = image_file.read_bytes()
        b64: str = base64.b64encode(data).decode('ascii')
        suffix: str = image_file.suffix.lower()
        mime: str = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
        }.get(suffix, 'image/jpeg')

        # (3) Read-only category lookup + current local time (approximates the photo-send time).
        categories: List[str] = load_known_categories()
        now_str: str = datetime.now().strftime('%Y-%m-%d %H:%M')

        # (4) Build ONE English vision prompt: extraction contract + categories + optional user context + strict output shape.
        categories_line: str = ', '.join(categories) if categories else 'none stored yet — you may create the first category'
        # NOTE: plain (non-f) string on purpose — it contains literal braces.
        output_shape: str = ('{"receipts": [{"total_cost": number|null, "currency": str|null, '
            '"date": "YYYY-MM-DD HH:MM"|null, "items": [{"name": str, "quantity": number, '
            '"unit_price": number, "currency": str}], "location": str, "category": str, '
            '"confidence": "high"|"low"}]}')
        prompt_parts: List[str] = [
            'You are a receipt-OCR extraction engine. Analyze the attached receipt photo and extract EVERY receipt it contains — output ONE entry per receipt when several appear in the same photo.',
            'Extraction contract for every entry:',
            "- total_cost: the total amount paid INCLUDING tax, as a number, with the original ISO-4217 currency code in 'currency'.",
            f"- date: a string in 'YYYY-MM-DD HH:MM' format. Take the HH:MM from the receipt when it is printed there; otherwise use EXACTLY the current local time provided here: {now_str}. When the receipt's date is unreadable, the date falls back to today (the date part of the provided time).",
            '- items: the line items, each with name, quantity (a float; fractional amounts like 0.5 are allowed), unit_price and currency exactly as printed on the receipt.',
            "- location: the merchant location as printed on the receipt; use '' when it is not printed.",
            '- category: a suggested category chosen from the known categories provided below — reuse an existing category whenever it fits, and create a new one ONLY when the receipt clearly fits none of them.',
            "- confidence: your self-assessed extraction confidence flag, 'high' or 'low'.",
            '- Unreadable required fields (total_cost, currency, date) must be null, and items must be an empty list when nothing could be read.',
            f'Known categories: {categories_line}',
        ]
        if user_text is not None:
            prompt_parts.append(f'Additional context from the user: {user_text}')
        prompt_parts.append('Reply with ONLY a JSON object — no markdown fences and no commentary — of exactly this shape:')
        prompt_parts.append(output_shape)
        prompt: str = '\n'.join(prompt_parts)

        # (5) EXACTLY ONE OpenRouter vision call with a FRESH LLM (chat_llm has tools bound — never use it here).
        vision_llm = myChatOpenAI(temperature=0.0)
        response = safe_invoke(vision_llm, messages=[
            HumanMessage(content=[
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}},
            ])
        ])
        if response is None:
            return {'receipts': [], 'error': 'vision model call failed'}

        # (6) Extract the text and parse the JSON payload (strip fences, slice first '{' .. last '}').
        content = response.content
        if isinstance(content, list):
            joined_parts: List[str] = []
            for part in content:
                if isinstance(part, str):
                    joined_parts.append(part)
                elif isinstance(part, dict):
                    joined_parts.append(str(part.get('text', '')))
                else:
                    joined_parts.append(str(part))
            content = ' '.join(joined_parts)
        text: str = str(content).strip()
        text = re.sub(r'^```[a-zA-Z]*\s*', '', text)   # leading ```json / ``` fence
        text = re.sub(r'\s*```$', '', text).strip()    # trailing fence
        start: int = text.find('{')
        end: int = text.rfind('}')
        if start == -1 or end == -1 or end <= start:
            return {'receipts': [], 'error': 'could not parse vision model output'}
        try:
            parsed = json.loads(text[start:end + 1])
        except Exception:
            return {'receipts': [], 'error': 'could not parse vision model output'}
        if not isinstance(parsed, dict):
            return {'receipts': [], 'error': 'could not parse vision model output'}

        # (7) Validate every entry into ExtractedReceipt-shaped plain dicts; low-confidence fallback on failure.
        receipts_out: List[Dict[str, Any]] = []
        entries = parsed.get('receipts', [])
        if not isinstance(entries, list):
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                validated: ExtractedReceipt = ExtractedReceipt.model_validate(entry)
                receipts_out.append(validated.model_dump())
            except Exception:  # ValidationError or any field-coercion failure -> minimal low-confidence entry
                receipts_out.append(ExtractedReceipt(
                    total_cost=None,
                    currency=None,
                    date=None,
                    items=[],
                    location=str(entry.get('location') or ''),
                    category=str(entry.get('category') or ''),
                    confidence='low',
                ).model_dump())

        # (8) Return plain JSON-serializable dicts (NOT pydantic objects).
        return {'receipts': receipts_out}

    except Exception as e:
        print(f'{RED}[TOOL] [ERR] extract_receipt_data: {e}{RESET}') if DEBUG else None
        return {'receipts': [], 'error': str(e)[:200] or 'extraction failed'}

@tool
def convert_to_eur(amount: float, from_currency: str) -> Dict[str, Union[float, Literal['api', 'csv_fallback'], str]]:
    """
    Overview: The SINGLE currency-conversion tool for the whole workflow: converts a non-EUR receipt total to EUR and encapsulates ALL currency logic — the free keyless FX API call (e.g. Frankfurter, ECB rates, EUR base, no key), the fallback to the most recent stored rate for that currency in the local conversions.csv when the API fails, and the refresh of conversions.csv (created on first use, always kept current with the latest rate per currency) when the API succeeds. Returns the EUR amount, the rate used, its date, and the source ('api' | 'csv_fallback') so the caller can both set the row's final EUR cost and record the original (pre-conversion) currency in the comments column (e.g. 'Original: 12.50 USD (rate 0.9234, 2025-01-15, source: csv_fallback)'). NON-TERMINAL context updater: after it runs, control returns to the caller LLM, which composes the final ExpenseRow (cost_eur + comments note) and calls append_expense_rows in the same run. Only receipt TOTALS are converted; line items keep their printed currency in the items string.
    
    Caller LLM: chat_llm — the tool-bound LLM of the chat node (myChatOpenAI, temperature=0), part of chat_llm.bind_tools([extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses]).
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None — the ToolNode appends the returned dict as a ToolMessage to state['messages']; the caller LLM then builds the comments note from (from_currency, original_amount, eur_amount, rate, rate_date, source) and calls append_expense_rows with the final EUR cost in the same run. No handler-side state mutation beyond the automatic messages append.
    
    Inside-the-Tool Work (Tool Responsibilities): Validates inputs; short-circuits EUR amounts (rate 1.0, no API/csv touch); otherwise calls the FX API with a short timeout; on success appends/refreshes the latest rate for the currency in conversions.csv (creating the file on first use) and returns source 'api'; on API failure reads the most recent rate for that currency from conversions.csv and returns source 'csv_fallback' (no csv write on fallback); when both API and csv fail, returns {'error': ...} so the caller asks the user instead of writing a row. Side effects: may create/append conversions.csv; one outbound HTTPS call to the FX API on the success path.
    
    Instructions: 1. Validate amount (finite number > 0) and from_currency (non-empty currency code, normalized to upper case, e.g. 'USD'); on invalid input return {'error': '<reason>'} without any I/O. 2. If from_currency == 'EUR', return {'eur_amount': amount, 'rate': 1.0, 'rate_date': <today 'YYYY-MM-DD'>, 'source': 'api', 'from_currency': 'EUR', 'original_amount': amount} immediately — no API call, no csv write. 3. Call the free keyless FX API (e.g. Frankfurter) for from_currency→EUR with a short timeout. 4. On success: append/refresh the latest rate for from_currency in conversions.csv (create the file on first use; keep the most recent rate per currency current) and return {'eur_amount': float, 'rate': float, 'rate_date': 'YYYY-MM-DD', 'source': 'api', 'from_currency': str, 'original_amount': float}. 5. On API failure: read the most recent stored rate for from_currency from conversions.csv; if found, return the same shape with 'source': 'csv_fallback' and do NOT write to the csv. 6. If both the API and the csv fallback fail, return {'error': 'no EUR rate available for <from_currency>'} — the caller LLM must then ask the user instead of writing a row. 7. Never raise; always return a compact JSON-serializable dict.
    
    State Updates (on the caller function): None — a tool cannot see or mutate graph state. The ToolMessage is appended to state['messages'] automatically by the ToolNode; the caller LLM uses the returned dict to set the row's final EUR cost and to compose the comments note recording the original (pre-conversion) currency, as required by the ExpenseRow.comments contract, before calling append_expense_rows.
    
    Args: amount (float): Receipt total amount paid INCLUDING tax, in from_currency (ExtractedReceipt.total_cost); required. from_currency (str): Original receipt currency as an ISO-4217 code (ExtractedReceipt.currency), e.g. 'USD'; required.
    
    Returns: Dict[str, Union[float, Literal['api', 'csv_fallback'], str]] — success: {'eur_amount': float, 'rate': float, 'rate_date': 'YYYY-MM-DD', 'source': 'api'|'csv_fallback', 'from_currency': str, 'original_amount': float}; no rate available: {'error': '<reason>'}.
    """
    try:
        # (2) Validate BEFORE any I/O
        if not isinstance(amount, (int, float)) or isinstance(amount, bool) \
                or not math.isfinite(amount) or amount <= 0:
            return {'error': f'invalid amount: {amount!r} (must be a finite number > 0)'}
        if not isinstance(from_currency, str) or not from_currency.strip():
            return {'error': f'invalid from_currency: {from_currency!r} (must be a non-empty string)'}
        currency: str = from_currency.strip().upper()

        # (3) EUR short-circuit: no API call, no csv touch
        if currency == 'EUR':
            return {
                'eur_amount': float(amount),
                'rate': 1.0,
                'rate_date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'api',
                'from_currency': 'EUR',
                'original_amount': float(amount),
            }

        csv_path: Path = Path('conversions.csv')

        # (4a) Read helper: {} when the file is missing (never created on read); skips malformed rows
        def _read_stored_rates() -> Dict[str, Dict[str, Any]]:
            if not csv_path.exists():
                return {}
            entries: Dict[str, Dict[str, Any]] = {}
            with csv_path.open('r', newline='', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    try:
                        row_currency: str = str(row.get('currency') or '').strip().upper()
                        row_rate: float = float(row.get('rate'))
                        row_date: str = str(row.get('date') or '')
                    except (TypeError, ValueError):
                        continue  # malformed row -> skip
                    if not row_currency or not math.isfinite(row_rate) or row_rate <= 0:
                        continue
                    entries[row_currency] = {'rate': row_rate, 'date': row_date}
            return entries

        # (4b) Upsert helper: keep the LATEST rate per currency; rewrite the WHOLE file (creates it on first use)
        def _upsert_rate(upsert_currency: str, upsert_rate: float, upsert_rate_date: str) -> None:
            entries: Dict[str, Dict[str, Any]] = _read_stored_rates()  # {} when file missing
            entries[upsert_currency] = {'rate': upsert_rate, 'date': upsert_rate_date}
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['currency', 'rate', 'date'])
                for entry_currency, entry in entries.items():
                    writer.writerow([entry_currency, entry['rate'], entry['date']])

        # (5) FX API: ANY exception / non-200 / missing data['rates']['EUR'] -> API failure
        api_rate: Optional[float] = None
        api_data: Optional[Dict[str, Any]] = None
        try:
            response = requests.get('https://api.frankfurter.app/latest',
                                    params={'from': currency, 'to': 'EUR'}, timeout=10)
            if response.status_code != 200:
                raise ValueError(f'FX API returned HTTP status {response.status_code}')
            api_data = response.json()
            candidate_rate: float = float(api_data['rates']['EUR'])
            if not math.isfinite(candidate_rate) or candidate_rate <= 0:
                raise ValueError(f'FX API returned an invalid EUR rate: {candidate_rate!r}')
            api_rate = candidate_rate
        except Exception:
            api_rate = None
            api_data = None

        # (6) API success: refresh conversions.csv with the latest rate, return the 'api' dict
        if api_rate is not None:
            eur_amount: float = round(float(amount) * api_rate, 2)
            rate_date: str = str(api_data.get('date') or datetime.now().strftime('%Y-%m-%d'))
            _upsert_rate(currency, api_rate, rate_date)
            return {
                'eur_amount': eur_amount,
                'rate': api_rate,
                'rate_date': rate_date,
                'source': 'api',
                'from_currency': currency,
                'original_amount': float(amount),
            }

        # (7) API failure -> csv fallback (NO csv write on this path)
        stored_rates: Dict[str, Dict[str, Any]] = _read_stored_rates()
        if currency in stored_rates:
            fallback_rate: float = float(stored_rates[currency]['rate'])
            return {
                'eur_amount': round(float(amount) * fallback_rate, 2),
                'rate': fallback_rate,
                'rate_date': str(stored_rates[currency]['date']),
                'source': 'csv_fallback',
                'from_currency': currency,
                'original_amount': float(amount),
            }

        # (8) Both API and csv fallback failed
        return {'error': f'no EUR rate available for {currency}'}

    except Exception as e:
        print(f'{RED}[TOOL] [ERR] convert_to_eur: {e}{RESET}') if DEBUG else None
        return {'error': str(e) or type(e).__name__}

@tool
def append_expense_rows(rows: List[ExpenseRow]) -> Dict[str, Union[int, List[ExpenseRow], str]]:
    """
    Overview: Append-only write of one or more expense rows to the local Excel file (default ./expenses.xlsx, created on first use with the exact headers cost | date | items | location | category | comments). Each element of `rows` is ONE receipt → ONE row and must match the provided ExpenseRow schema: cost (float, final EUR amount including tax, already converted via convert_to_eur when the receipt was in another currency), date ('YYYY-MM-DD HH:MM' in the machine's configured local timezone), items ('item1 (qty x price currency), item2 (qty x price currency), ...' built with ExtractedReceipt.format_items(); line items keep their printed currency), location ('' when unknown), category (deduplicated — an existing spreadsheet category reused whenever it fits, a new one only when nothing fits), comments (free-text user/agent notes; MUST include the original-currency note whenever a conversion was applied, e.g. 'Original: 12.50 USD (rate 0.9234, 2025-01-15, source: csv_fallback)'). Plain dicts with exactly these six fields are accepted and validated with ExpenseRow.model_validate; values are appended in the exact header order via ExpenseRow.to_row_values() → [cost, date, items, location, category, comments]. Existing rows are never modified, reordered, or deleted — the sheet is append-only. This is the TERMINAL finalizer of the receipt-ingest branch: after it runs, the caller LLM must emit the run's single user-visible summary reply and end the run with no further tool calls.
    
    Caller LLM: chat_llm — the single tool-bound LLM of the chat node (myChatOpenAI, temperature=0), bound via chat_llm.bind_tools([extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses]).
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None — the ToolNode appends the returned summary dict as a ToolMessage to state['messages']; the caller LLM then writes the run's one outbound AIMessage (an append summary such as 'Logged 2 receipts: €4.30 Groceries, €12.50 Restaurants') and ends the run. No handler-side state mutation beyond the automatic messages append.
    
    Inside-the-Tool Work (Tool Responsibilities): Validates the rows payload (each element against the ExpenseRow schema via ExpenseRow.model_validate); creates ./expenses.xlsx (and parent directories if needed) with the six headers on first use; appends each row's values in exact header order [cost, date, items, location, category, comments] (the ExpenseRow.to_row_values() ordering) via openpyxl's append-only ws.append(...); saves the workbook; returns a compact summary dict. Side effects: the ONLY tool that writes to expenses.xlsx; appends rows, never rewrites existing ones.
    
    Instructions: 1. Validate `rows`: a non-empty list where each element matches the ExpenseRow schema — cost (float, EUR), date ('YYYY-MM-DD HH:MM'), items (str), location (str, may be ''), category (str), comments (str, may be ''); validate each element with ExpenseRow.model_validate and on invalid input return {'error': '<reason>'} without touching the file. 2. Validate-before-write is the CALLER's duty (extraction confident OR user confirmed/corrected in a prior turn) — this tool writes unconditionally whatever valid rows it receives, so the caller must never pass unconfirmed low-confidence extractions. 3. Open ./expenses.xlsx or create it with headers cost|date|items|location|category|comments on first use; never rewrite, reorder, or delete existing rows. 4. Append each row in list order using ExpenseRow.to_row_values() for the value order, then save the workbook. 5. Return {'appended': <int>, 'rows': [<ExpenseRow-shaped appended rows in list order>], 'file': './expenses.xlsx'}; never raise.
    
    State Updates (on the caller function): None — a tool cannot see or mutate graph state. The ToolMessage is appended to state['messages'] automatically by the ToolNode; the caller LLM then produces the run's single summary reply and ends the run (no further tool calls after this terminal finalizer).
    
    Args: rows (List[ExpenseRow]): One ExpenseRow per receipt, in processing order — each with exactly the schema fields {'cost': float, 'date': 'YYYY-MM-DD HH:MM', 'items': str, 'location': str, 'category': str, 'comments': str} (plain dicts with exactly these keys are accepted and validated via ExpenseRow.model_validate); required.
    
    Returns: Dict[str, Union[int, List[ExpenseRow], str]] — success: {'appended': int, 'rows': [ExpenseRow, ...], 'file': str}; invalid payload: {'error': '<reason>'}.
    """
    try:
        # 1. Validate FIRST — before touching any file.
        if not isinstance(rows, list) or not rows:
            return {'error': 'rows must be a non-empty list'}
        validated: List[ExpenseRow] = []
        for i, r in enumerate(rows):
            try:
                validated.append(ExpenseRow.model_validate(r))
            except ValidationError as ve:
                print(f'{RED}[TOOL] [ERR] append_expense_rows: row {i}: {ve}{RESET}') if DEBUG else None
                return {'error': f'invalid row at index {i}: {ve}'}

        # 2. Resolve the path; create parent directories only when the path is not a bare filename.
        path: Path = Path('expenses.xlsx')
        if str(path.parent) not in ('', '.'):
            path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Open the existing workbook, or create it with the exact headers on first use.
        wb: Workbook
        ws: Any
        if path.exists():
            wb = load_workbook(path)
            ws = wb.worksheets[0]
        else:
            wb = Workbook()
            ws = wb.active
            ws.append(['cost', 'date', 'items', 'location', 'category', 'comments'])

        # 4. APPEND-ONLY: append each validated row in list order; existing rows are never touched.
        for row in validated:
            ws.append(row.to_row_values())

        # 5. Save once — appends were in-memory only, so a save failure leaves the existing file untouched.
        wb.save(path)
        wb.close()

        # 6. Compact, JSON-serializable summary (plain dicts, NOT pydantic objects).
        return {
            'appended': len(validated),
            'rows': [r.model_dump() for r in validated],
            'file': 'expenses.xlsx',
        }
    except Exception as e:
        print(f'{RED}[TOOL] [ERR] append_expense_rows: {e}{RESET}') if DEBUG else None
        return {'error': str(e)}

@tool
def update_row_comment(comment: str, row_identifier: Optional[str] = None) -> Dict[str, Union[bool, ExpenseRow, Literal['latest', 'date', 'row_number'], str]]:
    """
    Overview: Writes to the comments column of ONE existing row in the local Excel file (default ./expenses.xlsx) — the ONLY tool permitted to modify an existing row, and even then only its comments cell (cost/date/items/location/category stay untouched; the sheet is otherwise append-only). Targets the most recent (last) data row by default; otherwise the row the user identifies in chat by date ('YYYY-MM-DD HH:MM' or just 'YYYY-MM-DD') or by Excel row number. Serves branch (d) note requests: the user adds or amends a free-text note on a row through chat (the agent logs original-currency notes at append time via append_expense_rows, not via this tool). Returns the updated row as an ExpenseRow so the caller can confirm precisely what changed. TERMINAL finalizer of the note branch: after it runs, the caller LLM must emit the run's single confirmation reply and end the run with no further tool calls.
    
    Caller LLM: chat_llm — the single tool-bound LLM of the chat node (myChatOpenAI, temperature=0), bound via chat_llm.bind_tools([extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses]).
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None — the ToolNode appends the returned dict as a ToolMessage to state['messages']; the caller LLM then writes the run's one outbound AIMessage confirming the update (e.g. 'Comment updated on the 2025-01-15 18:32 row.') and ends the run. No handler-side state mutation beyond the automatic messages append.
    
    Inside-the-Tool Work (Tool Responsibilities): Opens the existing workbook read-write (never creates the file — no rows means nothing to comment on); resolves the target row from row_identifier (None → most recent data row; date-like string → matching date column value, most recent on multiple matches; numeric string → Excel row number, header is row 1 so values < 2 are rejected); overwrites ONLY the comments cell of that row with `comment`; saves the workbook; returns the updated ExpenseRow plus how it was matched. Side effects: modifies exactly one comments cell in expenses.xlsx.
    
    Instructions: 1. Validate `comment` (str; '' allowed to clear a note) and `row_identifier` (None, a 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DD' string, or a numeric row-number string). 2. If ./expenses.xlsx does not exist or has no data rows, return {'error': 'no expense rows yet'} WITHOUT creating the file. 3. Resolve the target row: None → the most recent data row; date-like string → the row whose date column equals the full 'YYYY-MM-DD HH:MM' or whose date part equals 'YYYY-MM-DD' (most recent on multiple matches); numeric string → that Excel row number (row 1 is the header; reject < 2 or beyond the last data row). 4. Overwrite only the comments cell of the resolved row with `comment`; leave every other column untouched. 5. Save the workbook and return {'updated': True, 'row': <ExpenseRow>, 'matched_by': 'latest'|'date'|'row_number'}; when nothing matches return {'error': "no row matches '<row_identifier>'"}. 6. Never raise; always return a compact JSON-serializable dict.
    
    State Updates (on the caller function): None — a tool cannot see or mutate graph state. The ToolMessage built from the return value is appended to state['messages'] automatically by the ToolNode; the caller LLM then produces the run's single confirmation reply and ends the run (no further tool calls after this terminal finalizer).
    
    Args: comment (str): The new comment text to write into the comments column (user note or agent note; pass '' to clear an existing note); required. row_identifier (Optional[str]): None → most recent row; otherwise the date ('YYYY-MM-DD HH:MM' or 'YYYY-MM-DD') or the Excel row number, exactly as the user specified it in chat; default None.
    
    Returns: Dict[str, Union[bool, ExpenseRow, Literal['latest', 'date', 'row_number'], str]] — success: {'updated': True, 'row': ExpenseRow, 'matched_by': 'latest'|'date'|'row_number'}; failure: {'error': '<reason>'}.
    """
    try:
        # (2) Normalize the comment ('' is allowed — it clears the note).
        comment: str = str(comment)

        # (3) The workbook must already exist — this tool NEVER creates the file.
        path: Path = Path('expenses.xlsx')
        if not path.exists():
            return {'error': 'no expense rows yet'}

        # (4) Open read-write and build a case-insensitive header map from row 1.
        wb: Any = load_workbook(path)
        ws: Any = wb.worksheets[0]
        header_map: Dict[str, int] = {}
        for col_idx, header_cell in enumerate(ws[1], start= 1):
            if header_cell.value is not None:
                header_map[str(header_cell.value).strip().lower()] = col_idx
        if 'date' not in header_map or 'comments' not in header_map:
            return {'error': 'sheet is missing required columns'}

        # (5) Data rows = rows 2..max_row with any non-None value among the six known columns.
        known_names: List[str] = ['cost', 'date', 'items', 'location', 'category', 'comments']
        known_idx: List[int] = [header_map[name] for name in known_names if name in header_map]
        data_rows: List[int] = []
        for r in range(2, (ws.max_row or 1) + 1):
            if any(ws.cell(row= r, column= c).value is not None for c in known_idx):
                data_rows.append(r)
        if not data_rows:
            return {'error': 'no expense rows yet'}
        last_data_row: int = data_rows[-1]

        # (6) Resolve the target row from row_identifier.
        target: int
        matched_by: str
        if row_identifier is None:
            # Most recent (last) data row.
            target = last_data_row
            matched_by = 'latest'
        elif re.fullmatch(r'\d+', str(row_identifier).strip()):
            # Digits-only string (also covers a plain int row_identifier) -> Excel row number; header is row 1.
            n: int = int(str(row_identifier).strip())
            if n < 2 or n > last_data_row:
                return {'error': f'row number {n} is out of range'}
            target = n
            matched_by = 'row_number'
        else:
            # Date identifier: full 'YYYY-MM-DD HH:MM' equality OR 'YYYY-MM-DD' prefix match on the date column.
            ident: str = str(row_identifier).strip()
            date_col: int = header_map['date']
            best_date: Optional[str] = None
            best_row: Optional[int] = None
            for r in data_rows:
                raw_date: Any = ws.cell(row= r, column= date_col).value
                date_str: str = '' if raw_date is None else str(raw_date).strip()
                if date_str == ident or date_str.startswith(ident):
                    # The fixed 'YYYY-MM-DD HH:MM' format sorts lexicographically, so a strict '>' keeps the most recent match (ties keep the earlier row, mirroring max() semantics).
                    if best_date is None or date_str > best_date:
                        best_date = date_str
                        best_row = r
            if best_row is None:
                return {'error': f"no row matches '{row_identifier}'"}
            target = best_row
            matched_by = 'date'

        # (7) Read the six cells of the target row and validate as ExpenseRow BEFORE any write.
        def _cell_text(col_name: str) -> str:
            if col_name not in header_map:
                return ''
            raw: Any = ws.cell(row= target, column= header_map[col_name]).value
            return '' if raw is None else str(raw).strip()

        cost_val: float = 0.0
        if 'cost' in header_map:
            raw_cost: Any = ws.cell(row= target, column= header_map['cost']).value
            if raw_cost is not None:
                try:
                    cost_val = float(raw_cost)
                except (TypeError, ValueError):
                    cost_val = 0.0

        try:
            updated: ExpenseRow = ExpenseRow(
                cost= cost_val,
                date= _cell_text('date'),
                items= _cell_text('items'),
                location= _cell_text('location'),
                category= _cell_text('category'),
                comments= comment,
            )
        except ValidationError as ve:
            return {'error': str(ve)} # Nothing written, nothing saved.

        # (8) Overwrite ONLY the comments cell of the target row; every other cell stays untouched.
        ws.cell(row= target, column= header_map['comments']).value = comment
        wb.save(path)

        # (9) Return a plain JSON-serializable dict (NOT a pydantic object) for the ToolMessage.
        return {'updated': True, 'row': updated.model_dump(), 'matched_by': matched_by}

    except Exception as e:
        print(f'{RED}[TOOL] [ERR] update_row_comment: {e}{RESET}') if DEBUG else None
        return {'error': str(e)}

@tool
def query_expenses(question: str, group_by: Optional[Literal['category', 'item', 'month']] = None, period: Optional[str] = None, category: Optional[str] = None, location: Optional[str] = None, days: Optional[int] = None, top_n: Optional[int] = None, bottom_n: Optional[int] = None) -> Dict[str, Union[Dict[str, Any], int, str, None]]:
    """
    Overview: Reads the local Excel expenses file (default ./expenses.xlsx) strictly read-only — never creates or modifies it — and returns computed aggregates answering the user's spending question, with EXPLICIT filter arguments (per user requirement) instead of relying only on free text: filter to one `category`, one `location`, a `days` window, and/or rank `top_n` (highest spend) or `bottom_n` (lowest spend). Aggregation dimensions via `group_by`: 'category' → total EUR cost per category plus grand total; 'item' → cumulative spend per item name computed from the items-string tokens 'name (qty x price currency)' (quantity × unit price as printed; line items keep their printed currency, so when one item name appears in multiple currencies the tool returns separate per-currency subtotals and flags 'mixed_currencies': True for the caller to caveat); 'month' → total EUR cost per 'YYYY-MM'. Spec-fixed semantics: 'this month' = the current calendar month in the machine's configured local timezone; 'top items' = highest cumulative spend per item across receipts. Returns a compact JSON dict of aggregates; the caller LLM turns it into the single natural-language English answer. NON-TERMINAL context updater: after it runs, control returns to the caller LLM, which formats the final reply (or, rarely, decides a follow-up tool call).
    
    Caller LLM: chat_llm — the single tool-bound LLM of the chat node (myChatOpenAI, temperature=0), bound via chat_llm.bind_tools([extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses]). The OpenRouter vision-model call happens INSIDE this tool as an implementation detail; it is not a separate bound tool and is not visible to chat_llm.
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None — the ToolNode appends the returned aggregates dict as a ToolMessage to state['messages']; the caller LLM then composes the run's single user-visible English answer (e.g. 'You spent €84.20 on Groceries this month.') and ends the run. No handler-side state mutation beyond the automatic messages append.
    
    Inside-the-Tool Work (Tool Responsibilities): Read-only openpyxl load (read_only=True, data_only=True) of ./expenses.xlsx; parses data rows into {cost, date, items, location, category, comments} dicts; applies the row filters (category, location, days/period) using the machine's local timezone; aggregates per `group_by`; sorts descending by spend and applies top_n/bottom_n ranking; returns a compact JSON-serializable dict. Side effects: none (never writes any file).
    
    Instructions: 1. If ./expenses.xlsx does not exist or has no data rows, return {'answer_data': {}, 'rows_considered': 0, 'note': 'no expenses recorded yet'} WITHOUT creating the file. 2. Load the first worksheet read-only (data_only=True) and parse each data row into {'cost': float, 'date': 'YYYY-MM-DD HH:MM', 'items': str, 'location': str, 'category': str, 'comments': str}. 3. Apply the row filters: `category` (Optional[str]) → keep only rows whose category equals it case-insensitively (categories are deduplicated, so exact-insensitive match; no match → empty aggregation with a note); `location` (Optional[str]) → keep only rows whose location contains it case-insensitively (substring match, e.g. 'Rewe' matches 'Rewe Berlin Mitte'); time window → `days` (Optional[int]) keeps rows within the last N days including today in the machine's local timezone, else `period` (None → all rows; 'this_month' → current calendar month local timezone; 'last_month' → previous calendar month; 'YYYY-MM' → that month; 'YYYY-MM-DD..YYYY-MM-DD' → inclusive date range); when both `days` and `period` are given, `days` takes precedence; an unparseable period → return {'error': '<reason>'} without aggregating. 4. Aggregate per `group_by`: None → infer from `question` (mentions items → 'item', months/periods → 'month', otherwise 'category'); 'category' → total EUR cost per category plus grand total; 'item' → cumulative spend per item name from items-string tokens (qty × unit price), keeping each token's printed currency — when one item name spans multiple currencies, emit separate per-currency subtotals and set 'mixed_currencies': True; 'month' → total EUR cost per 'YYYY-MM'. Sort each aggregation descending by spend. 5. Apply ranking: `top_n` (Optional[int]) → keep only the N highest-spend entries and expose them under 'top_n' in answer_data; `bottom_n` (Optional[int]) → keep only the N lowest-spend entries under 'bottom_n'; when both are given, top_n takes precedence. 6. Return {'answer_data': <aggregates dict>, 'rows_considered': <int>, 'period': <str|None>, 'group_by': <str|None>} — compact and easy for the caller to phrase; never raise and never write to any file.
    
    State Updates (on the caller function): None — a tool cannot see or mutate graph state. The ToolMessage is appended to state['messages'] automatically by the ToolNode; the caller LLM then produces the run's single natural-language English answer from the aggregates and ends the run.
    
    Args: question (str): The user's spending question, verbatim or condensed (e.g. 'how much did I spend on groceries this month?', 'what are the top 3 items I spent the most on?'); used to infer group_by when None and to phrase the answer; required. group_by (Optional[Literal['category', 'item', 'month']]): Aggregation dimension — 'category' | 'item' | 'month'; None means infer from `question`, defaulting to category totals; default None. period (Optional[str]): Time filter — None (all time), 'this_month' (current calendar month, local timezone), 'last_month', 'YYYY-MM', or 'YYYY-MM-DD..YYYY-MM-DD' (inclusive); ignored when `days` is given; default None. category (Optional[str]): Keep only rows in this category (case-insensitive exact match on the deduplicated category column); default None (no category filter). location (Optional[str]): Keep only rows whose location contains this text (case-insensitive substring match); default None. days (Optional[int]): Keep only rows within the last N days including today, in the machine's configured local timezone; takes precedence over `period` when both are given; default None. top_n (Optional[int]): Return only the N highest-spend entries after aggregation (e.g. top 3 items); takes precedence over bottom_n; default None. bottom_n (Optional[int]): Keep only the N lowest-spend entries after aggregation; default None.
    
    Returns: Dict[str, Union[Dict[str, Any], int, str, None]] — success: {'answer_data': dict (per-category EUR totals / per-item cumulative spend with currency and optional 'mixed_currencies': True / per-month EUR totals, sorted descending, grand total, and 'top_n'/'bottom_n' ranked lists where requested), 'rows_considered': int, 'period': str|None, 'group_by': str|None}; no data: {'answer_data': {}, 'rows_considered': 0, 'note': str}; bad period: {'error': str}.
    """
    try:
        # ----------------------------------------------------------------------
        # Nested helpers (self-contained; no module pollution, no new imports).
        # ----------------------------------------------------------------------
        def _parse_date_str(date_s: str) -> Optional[datetime]:
            # Try 'YYYY-MM-DD HH:MM' first, then 'YYYY-MM-DD'; None when unparseable.
            for fmt in ('%Y-%m-%d %H:%M', '%Y-%m-%d'):
                try:
                    return datetime.strptime(date_s, fmt)
                except Exception:
                    continue
            return None

        def _parse_item_token(part: str) -> Optional[tuple]:
            # 'name (qty x price currency)' -> (name, qty, price, CURRENCY); None -> skip token.
            m: Optional[re.Match] = re.match(r'^(.*?)\s*\((.+?)\s+x\s+(.+?)\)\s*$', part.strip())
            if not m:
                return None
            bits: List[str] = m.group(3).strip().rsplit(' ', 1)
            if len(bits) != 2:
                return None
            try:
                return (m.group(1).strip(), float(m.group(2)), float(bits[0]), bits[1].strip().upper())
            except Exception:
                return None

        def _cell(row: tuple, idx: Optional[int]) -> Any:
            # Cell value at header index `idx`; None when the column/cell is absent.
            if idx is None or idx >= len(row):
                return None
            return row[idx]

        # ----------------------------------------------------------------------
        # 1) Locate the workbook — strictly read-only; NEVER create the file.
        # ----------------------------------------------------------------------
        path: Path = Path('expenses.xlsx')
        if not path.exists():
            return {'answer_data': {}, 'rows_considered': 0, 'note': 'no expenses recorded yet'}

        # ----------------------------------------------------------------------
        # 2) Read-only load and row parsing.
        # ----------------------------------------------------------------------
        rows: List[Dict[str, Any]] = []
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.worksheets[0] if wb.worksheets else None
            if ws is None:
                return {'answer_data': {}, 'rows_considered': 0, 'note': 'no expenses recorded yet'}
            rows_iter: Any = ws.iter_rows(values_only=True)
            header: Optional[tuple] = next(rows_iter, None)
            if header is None:
                return {'answer_data': {}, 'rows_considered': 0, 'note': 'no expenses recorded yet'}
            # Case-insensitive header name -> column index map (cost|date|items|location|category|comments).
            col: Dict[str, int] = {}
            for col_idx, h in enumerate(header):
                if h is not None:
                    col[str(h).strip().lower()] = col_idx
            idx_cost: Optional[int] = col.get('cost')
            idx_date: Optional[int] = col.get('date')
            idx_items: Optional[int] = col.get('items')
            idx_location: Optional[int] = col.get('location')
            idx_category: Optional[int] = col.get('category')
            idx_comments: Optional[int] = col.get('comments')

            for row in rows_iter:
                # Skip fully empty rows (every cell None or whitespace-only).
                if not any(v is not None and str(v).strip() != '' for v in row):
                    continue
                try:
                    cost_val: float = float(_cell(row, idx_cost))
                except Exception:
                    cost_val = 0.0
                date_raw: Any = _cell(row, idx_date)
                date_val: str = '' if date_raw is None else str(date_raw).strip()
                items_raw: Any = _cell(row, idx_items)
                location_raw: Any = _cell(row, idx_location)
                category_raw: Any = _cell(row, idx_category)
                comments_raw: Any = _cell(row, idx_comments)
                rows.append({
                    'cost': cost_val,
                    'date': date_val,
                    'items': '' if items_raw is None else str(items_raw),
                    'location': '' if location_raw is None else str(location_raw),
                    'category': '' if category_raw is None else str(category_raw),
                    'comments': '' if comments_raw is None else str(comments_raw),
                    '_parsed': _parse_date_str(date_val),
                })
        finally:
            wb.close()

        if not rows:
            return {'answer_data': {}, 'rows_considered': 0, 'note': 'no expenses recorded yet'}

        # ----------------------------------------------------------------------
        # 3) Filters, in order: category -> location -> time window.
        # ----------------------------------------------------------------------
        if category is not None:
            cat_filter: str = category.strip().lower()
            rows = [r for r in rows if r['category'].strip().lower() == cat_filter]
            if not rows:
                return {'answer_data': {}, 'rows_considered': 0, 'note': f"no rows in category '{category}'"}
        if location is not None:
            loc_filter: str = location.lower()
            rows = [r for r in rows if loc_filter in r['location'].lower()]

        if days is not None:
            # `days` takes precedence over `period`.
            if days <= 0:
                return {'error': 'days must be a positive integer'}
            cutoff: datetime = (datetime.now() - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
            rows = [r for r in rows if r['_parsed'] is not None and r['_parsed'] >= cutoff]
        elif period is not None:
            now: datetime = datetime.now()
            if period == 'this_month':
                rows = [r for r in rows if r['_parsed'] is not None and r['_parsed'].year == now.year and r['_parsed'].month == now.month]
            elif period == 'last_month':
                prev_month: datetime = (now.replace(day=1) - timedelta(days=1))
                rows = [r for r in rows if r['_parsed'] is not None and r['_parsed'].year == prev_month.year and r['_parsed'].month == prev_month.month]
            elif re.match(r'^\d{4}-\d{2}$', period):
                year_i: int = int(period[:4])
                month_i: int = int(period[5:7])
                rows = [r for r in rows if r['_parsed'] is not None and r['_parsed'].year == year_i and r['_parsed'].month == month_i]
            elif re.match(r'^\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2}$', period):
                start_s, end_s = period.split('..', 1)
                try:
                    start_dt: datetime = datetime.strptime(start_s, '%Y-%m-%d')
                    end_dt: datetime = datetime.strptime(end_s, '%Y-%m-%d')
                except Exception:
                    return {'error': f"unparseable period '{period}'"}
                # Inclusive calendar-day range: receipts ON the end date count.
                rows = [r for r in rows if r['_parsed'] is not None and start_dt.date() <= r['_parsed'].date() <= end_dt.date()]
            else:
                return {'error': f"unparseable period '{period}'"}

        # ----------------------------------------------------------------------
        # 4) group_by: infer from the question when None.
        # ----------------------------------------------------------------------
        if group_by is None:
            q_lower: str = question.lower()
            if 'item' in q_lower:
                group_by = 'item'
            elif 'month' in q_lower:
                group_by = 'month'
            else:
                group_by = 'category'

        # ----------------------------------------------------------------------
        # 5) Aggregations (2-decimal rounding, sorted descending by spend).
        # ----------------------------------------------------------------------
        answer_data: Dict[str, Any] = {}
        primary: List[Dict[str, Any]] = []
        spend_key: str = 'total_eur'

        if group_by == 'category':
            totals: Dict[str, float] = {}
            for r in rows:
                cat_key: str = r['category'].strip() or 'Uncategorized'
                totals[cat_key] = totals.get(cat_key, 0.0) + r['cost']
            primary = [{'category': c, 'total_eur': round(v, 2)} for c, v in totals.items()]
            primary.sort(key=lambda d: d['total_eur'], reverse=True)
            answer_data = {'by_category': primary, 'grand_total_eur': round(float(sum(totals.values())), 2)}
        elif group_by == 'item':
            spend: Dict[tuple, float] = {}
            for r in rows:
                for part in r['items'].split(','):
                    token: Optional[tuple] = _parse_item_token(part)
                    if token is None:
                        continue
                    t_name, t_qty, t_price, t_cur = token
                    spend[(t_name, t_cur)] = spend.get((t_name, t_cur), 0.0) + (t_qty * t_price)
            primary = [{'item': n, 'currency': c, 'total': round(v, 2)} for (n, c), v in spend.items()]
            primary.sort(key=lambda d: d['total'], reverse=True)
            answer_data = {'items': primary}
            spend_key = 'total'
            currencies_by_name: Dict[str, set] = {}
            for (n, c) in spend.keys():
                currencies_by_name.setdefault(n, set()).add(c)
            if any(len(curs) > 1 for curs in currencies_by_name.values()):
                answer_data['mixed_currencies'] = True
            if any(c != 'EUR' for c in {c for (_, c) in spend.keys()}):
                answer_data['note'] = 'item totals are in the currency printed on the receipts'
        elif group_by == 'month':
            month_totals: Dict[str, float] = {}
            for r in rows:
                if r['_parsed'] is None:
                    continue  # unparseable dates are skipped in the month aggregation
                m_key: str = r['_parsed'].strftime('%Y-%m')
                month_totals[m_key] = month_totals.get(m_key, 0.0) + r['cost']
            primary = [{'month': m, 'total_eur': round(v, 2)} for m, v in month_totals.items()]
            primary.sort(key=lambda d: d['total_eur'], reverse=True)
            answer_data = {'by_month': primary, 'grand_total_eur': round(float(sum(month_totals.values())), 2)}
        else:
            return {'error': f"unsupported group_by '{group_by}'"}

        # ----------------------------------------------------------------------
        # 6) Ranking on the sorted primary list (top_n takes precedence).
        # ----------------------------------------------------------------------
        if top_n is not None and top_n > 0:
            answer_data['top_n'] = primary[:top_n]
        elif bottom_n is not None and bottom_n > 0:
            answer_data['bottom_n'] = sorted(primary, key=lambda d: d[spend_key])[:bottom_n]

        # ----------------------------------------------------------------------
        # 7) Compact, JSON-serializable result; no file was written.
        # ----------------------------------------------------------------------
        return {'answer_data': answer_data, 'rows_considered': len(rows), 'period': period, 'group_by': group_by}
    except Exception as e:
        print(f'{RED}[TOOL] [ERR] query_expenses: {e}{RESET}') if DEBUG else None
        return {'error': str(e)}
# TODO: Add Tools (if needed)



''' LLM '''
chat_llm = myChatOpenAI(
    temperature= 0.0
).bind_tools([extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses])




''' Helpful Functions '''
def load_known_categories() -> List[str]:
    """
    Overview: Reads the local Excel expenses file (default ./expenses.xlsx — the same file append_expense_rows writes to and query_expenses / update_row_comment read) and returns the deduplicated list of category values already present in the sheet's 'category' column. The result is embedded directly into the chat system prompt (prompts.CHAT_PROMPT) on every run, so the tool-bound LLM reuses an existing category whenever a receipt fits and creates a new category only when the receipt clearly fits none — preventing near-duplicates like 'Supermarket' vs 'Groceries'. This function replaces the removed read_categories_from_excel tool: category lookup is prompt-level preprocessing performed by node code, never an LLM tool call. It must be strictly read-only and side-effect free: it never creates the .xlsx or its directories (creation on first use is append_expense_rows' responsibility) and never modifies existing rows (the sheet is append-only). When the file does not exist yet, is empty, or contains no category values, it returns an empty list, and the prompt then tells the model it may create the first category.
    
    Caller Node: chat — called once per run in the node's preprocessing step (STEP-BY-STEP step 2, the '<preprocess> / <make format arguments readable>' TODO), before prompts.CHAT_PROMPT.format(...), so the system prompt always carries the current, deduplicated category list from the spreadsheet.
    
    Instructions: 1. Resolve the Excel path to the module default './expenses.xlsx' (relative to the working directory), matching the default path used by the append_expense_rows / query_expenses / update_row_comment tools. 2. If the file does not exist (e.g., the very first run before any receipt was ever logged), return [] immediately — do NOT create the file or any directories. 3. Open the workbook read-only with openpyxl (load_workbook(path, read_only=True, data_only=True)) and select the first worksheet. 4. Read the header row and locate the 'category' column by name (expected headers: cost | date | items | location | category | comments); if the sheet has no header row or no 'category' column, return []. 5. Iterate the data rows below the header and collect each row's category cell value; convert cells to str and strip whitespace; skip None/empty cells. 6. Deduplicate while preserving first-seen order and first-seen casing: treat values as duplicates when they are equal case-insensitively (so 'Groceries' and 'groceries' collapse to the first-seen form), keeping the list stable so the prompt shows categories exactly as stored in the sheet. 7. Close/release the workbook and return the deduplicated List[str]; the chat node formats it into prompts.CHAT_PROMPT (e.g., as a comma-separated list) as part of its 'make format arguments readable' preprocessing.
    
    Args: none — the function takes no parameters, reads only the default Excel path, and must not receive or mutate graph state.
    
    Returns: List[str] — the deduplicated, first-seen-ordered category names currently stored in the spreadsheet's 'category' column; [] when the .xlsx does not exist yet, has no data rows, or contains no category values.
    """
    try:
        path: Path = Path('expenses.xlsx')
        if not path.exists():
            return []
        categories: List[str] = []
        seen: set = set()
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.worksheets[0] if wb.worksheets else None
            if ws is None:
                return []
            rows_iter = ws.iter_rows(values_only=True)
            header = next(rows_iter, None)
            if header is None:
                return []
            lowered = [str(h).strip().lower() if h is not None else '' for h in header]
            if 'category' not in lowered:
                return []
            cat_idx: int = lowered.index('category')
            for row in rows_iter:
                if cat_idx >= len(row):
                    continue
                val = row[cat_idx]
                if val is None:
                    continue
                val = str(val).strip()
                if not val:
                    continue
                if val.lower() not in seen:
                    seen.add(val.lower())
                    categories.append(val)
            return categories
        finally:
            wb.close()
    except Exception as e:
        print(f'{RED}[HELPER] [ERR] load_known_categories: {e}{RESET}') if DEBUG else None
        return []
# TODO: Add Helpful Functions (if needed)



''' Nodes '''
def chat(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Single tool-using conversational step that reacts to the newest Human message and the routing context and produces exactly one user-visible reply per run, always in English. The known expense categories are supplied directly in the prompt (pre-loaded from the .xlsx), so no category-lookup tool is called: the model reuses an existing category when it fits and creates a new category only if nothing fits. Tools available: extract_receipt_data (exactly one OpenRouter vision call per photo; multiple receipts in one photo become separate rows), convert_to_eur (the single currency-conversion tool: it calls the FX API, falls back to the latest-per-currency rate in conversions.csv when the API fails, refreshes that csv with the newest rate, and returns the EUR amount — all fallback/read/write logic happens inside the tool), append_expense_rows (append-only Excel writes with deterministic YYYY-MM-DD HH:MM timestamps in the machine's configured local timezone), update_row_comment (comments column; targets the most recent row unless the user identifies a specific row by date or row number), and query_expenses (aggregations; 'this month' = current calendar month in the configured local timezone, 'top items' = highest cumulative spend per item across receipts). Branching: (a) receipt image file path(s) present -> run one vision extraction per photo, match categories from the prompt-provided list, convert non-EUR totals via convert_to_eur, then either append one row per receipt and reply with a summary, or store the pending extraction(s) in state, set state.next_action='handle_confirmation' with a pending_question, send one confirmation question, and end the run; (b) confirmation context -> apply the user's corrections directly (no second vision call), append the corrected row(s), and clear the pending extraction state and next_action; (c) spending question -> answer with computed aggregates from query_expenses; (d) note request -> update the comments column on the target row; (e) anything else -> converse normally. Guards: no Excel row is written unless extraction is confident OR the user confirmed/corrected it in a prior run; the Excel file is append-only; at most one outbound message per run; if several valid file paths arrive in one message, each is processed in the same run and any needed confirmations are combined into that run's single reply with all pending extractions held in state. Receipt data stays in local files (.xlsx, conversions.csv); no external data sharing beyond the OpenRouter vision call and the FX API. 
    Execution: LLM+TOOLS. Single tool-using conversational step for the whatsapp_receipt_expense_tracker reactive_conversational workflow; runs once per run (one inbound user message bridged from WhatsApp by the user; batch input/output, no streaming) and produces exactly one user-visible English reply per run. As far as the code is concerned this node is a plain LLM invocation with tools bound — there are NO extra routing/state keys (no next_action, pending_question, pending_extractions, or latest; removed per user feedback). All domain behavior lives in the system prompt, and the multi-turn confirmation flow resumes purely from conversation history: the previous run's AIMessage question and the user's new HumanMessage answer are both in state['messages'], so the model sees the open question and finishes the job (e.g., appends the corrected row) without any state flags.
    
    OVERVIEW
    Reacts to the newest Human message using the conversation history restored from the LangGraph checkpointer. The known expense categories are pre-loaded from the local .xlsx and embedded directly in the system prompt (helper load_known_categories), so there is NO category-lookup tool: the model reuses an existing category whenever it fits and creates a new one only when the receipt clearly fits none (preventing near-duplicates like 'Supermarket' vs 'Groceries'). All domain work happens through five tools bound with .bind_tools(): extract_receipt_data (exactly one OpenRouter vision call per photo), convert_to_eur (the single currency tool — FX API call, conversions.csv latest-rate fallback, and csv refresh all happen inside the tool), append_expense_rows (append-only Excel writes), update_row_comment (comments column), and query_expenses (aggregations). The model itself decides the branch from message content and history: (a) receipt image file path(s) -> one vision extraction per photo, category matched against the prompt-provided list, non-EUR totals converted, then either append one row per receipt and reply with a summary, or — when extraction is low-confidence or a required field is missing — ask ONE combined confirmation question and write nothing (the next run resumes from history); (b) the newest Human message answers a previously asked confirmation question (visible in messages) -> apply the corrections directly (no second vision call) and append the corrected row(s); (c) spending question -> answer with computed aggregates; (d) note request -> update the comments column; (e) anything else -> converse normally.
    
    STEP-BY-STEP
    1. Restore context: read state['messages'] (persisted by the checkpointer). The newest HumanMessage is this run's single inbound message (receipt image file path(s) and/or free text); earlier messages — including any AIMessage that asked a confirmation question in a previous run — provide the full conversational context. There are no other state keys to inspect.
    2. Build the system prompt: prompts.CHAT_PROMPT.format(...) with the known categories from load_known_categories(). The prompt carries ALL behavioral rules (prompt-level, not state/code logic): respond in English; exactly one extract_receipt_data call per photo (multiple receipts in one photo -> separate entries/rows); reuse an existing category when it fits, create a new one only if nothing fits; convert non-EUR totals with convert_to_eur and record the original receipt currency in the comments whenever a conversion was applied; items formatted 'item1 (qty x price currency), item2 (qty x price currency), ...'; date 'YYYY-MM-DD HH:MM' (HH:MM from the receipt when printed, otherwise the time the photo was sent; date fallback: today); validate-before-write — if extraction is low-confidence or cost/date/items are missing, ask the user to confirm or correct and write NOTHING; Excel is append-only; 'this month' = current calendar month in the local timezone; 'top items' = highest cumulative spend per item across receipts; at most one outbound reply per run; several valid file paths in one message are all processed in the same run with any confirmations merged into that single reply.
    3. Bind the tools: chat_llm.bind_tools([extract_receipt_data, convert_to_eur, append_expense_rows, update_row_comment, query_expenses]).
    4. Invoke: result = safe_invoke(chat_llm, messages=[SystemMessage(content=prompt)] + state['messages']).
    5. Tool loop: if the response contains tool calls, the ToolNode executes them and appends each result as a ToolMessage; merge those into the message list and re-invoke the LLM (same node, same run) so it continues reasoning with the tool results. Repeat (bounded) until the model returns a final AIMessage with no tool calls. Typical chains: extract_receipt_data -> convert_to_eur -> append_expense_rows -> summary; query_expenses -> answer; update_row_comment -> confirmation.
    6. Finalize: append the final AIMessage to state['messages'] and return the updated state — the last AIMessage is this run's single user-visible reply. If a confirmation question was asked, it IS that reply; the pending receipt data lives in the conversation history, and the user's next message (a new run) lets the model apply corrections and call append_expense_rows. Guards (enforced by the prompt): no Excel row is written unless extraction is confident OR the user confirmed/corrected it in a prior turn; existing rows are never overwritten; receipt data stays in local files (.xlsx, conversions.csv) with no external sharing beyond the OpenRouter vision call and the FX API. On any exception, log it (DEBUG) and return state unchanged so the run ends gracefully without a write.
    
    INPUTS (state)
    - state['messages'] (List[BaseMessage]): the full conversation log persisted by the checkpointer; the newest HumanMessage is this run's inbound WhatsApp-bridged message (receipt image file path(s) and/or free text); an earlier AIMessage may be a confirmation question whose answer is the new HumanMessage — that history pair is the entire confirmation-resume mechanism (no extra keys). AgentSchema is a MessagesState subclass, so 'messages' is its only key (add-messages reducer).
    
    OUTPUTS (state)
    - state['messages']: appended with this run's ToolMessage(s) (produced by the ToolNode) and exactly one final AIMessage — the single user-visible English reply (append summary, aggregate answer, comment-update confirmation, combined confirmation question, or normal chat).
    
    TOOLS (bound via chat_llm.bind_tools([...]); the ToolNode executes the requested calls, appends each result as a ToolMessage, and hands control back to this node in the same run)
    - extract_receipt_data(image_path: str, user_text: Optional[str] = None) -> dict — exactly one OpenRouter vision call per photo; OCRs the receipt and returns total cost incl. tax + currency, purchase date & time, line items (name, qty, unit price, currency), merchant location, a suggested category, and a confidence flag; one entry per receipt when a photo contains several.
    - convert_to_eur(amount: float, from_currency: str) -> dict — the single currency tool: calls the free keyless FX API (e.g., Frankfurter); on API failure falls back to the most recent rate for that currency in conversions.csv; refreshes conversions.csv (created on first use) with the newest rate; returns the EUR amount, the rate used, its date, and the source ('api' | 'csv_fallback').
    - append_expense_rows(rows: List[Dict[str, Any]]) -> dict — append-only write to the local .xlsx (default ./expenses.xlsx; created with headers cost|date|items|location|category|comments on first use); never modifies existing rows; each row: cost (EUR), date 'YYYY-MM-DD HH:MM', items string, location, category, comments.
    - update_row_comment(comment: str, row_identifier: Optional[str] = None) -> dict — writes to the comments column of the targeted row (most recent row by default; otherwise the row matching the user-supplied date or row number) and returns the updated row.
    - query_expenses(question: str, group_by: Optional[str] = None, period: Optional[str] = None) -> dict — reads the Excel data and returns computed aggregates by category, item, or time period (e.g., month totals, top-N items by cumulative spend).
    Note: LLM.invoke(...) is not a tool; only the five tools above are bound and executed by the ToolNode.
    
    HELPFUL FUNCTIONS
    - load_known_categories() -> List[str]: reads the existing .xlsx (if present) and returns the deduplicated list of categories already in the sheet, to be embedded in the system prompt (replaces the removed read_categories_from_excel tool; the model reuses these and creates a new category only when nothing fits).
    """

    print_function_name()
    try:
        # <preprocess> Load the known expense categories from the spreadsheet so the system prompt
        # can steer the model to reuse existing categories (there is NO category-lookup tool).
        known_categories: List[str] = load_known_categories()
        categories_str: str = ', '.join(known_categories) if known_categories else 'No categories stored yet - you may create the first category when a receipt clearly fits none.'

        # <make format arguments readable> Superset of format kwargs: str.format silently ignores
        # extra kwargs, so the prompt renders regardless of which placeholders the template uses.
        format_kwargs: Dict[str, str] = {
            'known_categories': categories_str,
            'categories': categories_str,
            'current_date': datetime.now().strftime('%Y-%m-%d'),
            'current_time': datetime.now().strftime('%H:%M'),
            'current_datetime': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }

        # <formatting> Robust prompt rendering: fall back to plain concatenation if the template
        # cannot be formatted, keeping the node alive even if the template changes.
        prompt: str
        try:
            prompt = prompts.CHAT_PROMPT.format(**format_kwargs)
        except Exception:
            prompt = prompts.CHAT_PROMPT + f'\n\nKnown expense categories (reuse an existing one whenever it fits; create a new one only when the receipt clearly fits none): {categories_str}'

        # <inputs> SINGLE LLM invocation. The conversation history goes ONLY via the messages
        # argument (never embedded into the prompt text — that would double-pass the messages to
        # safe_invoke). No tool loop and no tool execution here: the graph's 'tools' ToolNode plus
        # the route_chat conditional edge re-enter this node automatically after tools run.
        result: Optional[BaseMessage] = safe_invoke(chat_llm, messages=[SystemMessage(content=prompt)] + list(state['messages']))

        # <postprocess> None -> graceful no-op; otherwise return the partial state update — the
        # add_messages reducer appends the AIMessage to the persisted history (with tool_calls it
        # routes to the ToolNode and back into this node; a plain final AIMessage ends the run as
        # the run's single user-visible reply).
        if result is None:
            return state
        return {'messages': [result]}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state



def route_chat(state: AgentSchema) -> Literal['tools', '__end__']:
    """
    Conditional tool routing after the chat LLM node. All tool calls route to 'tools' (the single ToolNode), which returns to chat; no tool calls → END.
    """
    # Read the last message from the persisted conversation history (AgentSchema is a MessagesState subclass -> dict-style access).
    last: Optional[BaseMessage] = state['messages'][-1] if state['messages'] else None
    # Empty history (e.g. chat hit its exception path and returned state unchanged) -> end the run gracefully.
    if last is None:
        return END
    # Any tool call on the final AIMessage (any of the five bound tools) -> the single ToolNode, which returns to chat.
    if isinstance(last, AIMessage) and getattr(last, 'tool_calls', None):
        return 'tools'
    # Plain final reply (no tool calls) or a non-AIMessage tail -> end the run; the final AIMessage is the run's single user-visible reply.
    return END


# Type A tools (non-terminal, ToolMessage-only) execute on a standard ToolNode and return control to chat.
tools = ToolNode([extract_receipt_data, convert_to_eur, query_expenses, update_row_comment, append_expense_rows])



''' Graph '''
whatsapp_receipt_expense_tracker_graph = StateGraph(AgentSchema)

whatsapp_receipt_expense_tracker_graph.add_node("chat", chat)
whatsapp_receipt_expense_tracker_graph.add_node("tools", tools)

whatsapp_receipt_expense_tracker_graph.add_edge(START, "chat")
whatsapp_receipt_expense_tracker_graph.add_conditional_edges(
    "chat",
    route_chat,
    {
        'tools': 'tools',
        '__end__': END,
    }
)
whatsapp_receipt_expense_tracker_graph.add_edge("tools", "chat")


whatsapp_receipt_expense_tracker_app = whatsapp_receipt_expense_tracker_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    import uuid

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'full_framework_test',
            'run_name': 'full_framework_test',
            'thread_id': f'full_framework_test:{uuid.uuid4()}',
        }
    }

    print(
        'Commands:\n'
        '  re:<path>          Process a receipt image\n'
        '  re:<path> | <text> Process a receipt image with accompanying text\n'
        '  q                  Quit\n'
        '  anything else is sent as a normal conversation message\n'
    )

    user_in = input(f'{GREEN}[USER INPUT]{RESET} > ')

    while user_in.lower() != 'q':

        if user_in.startswith('re:'):
            receipt_input = user_in[3:].strip()

            if '|' in receipt_input:
                image_path, user_text = receipt_input.split('|', 1)
                message = (
                    f'Receipt image path: {image_path.strip()}\n'
                    f'Accompanying text: {user_text.strip()}'
                )
            else:
                message = f'Receipt image path: {receipt_input}'

        else:
            message = user_in

        response = whatsapp_receipt_expense_tracker_app.invoke({
            'messages': [HumanMessage(content=message)]
        },config= config)

        messages = response.get('messages', [])

        # Find the index of the last HumanMessage.
        last_human_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_index = i
                break
        # Print all messages after the last HumanMessage, in original order.
        for message in messages[last_human_index + 1:]:
            if message.content:
                print(f'\n{BLUE}[ANSWER]{RESET} {message.content}')
            else:
                print(f'\n{BLUE}[ANSWER]{RESET} {message}')

        user_in = input(f'\n{GREEN}[USER INPUT]{RESET} > ')