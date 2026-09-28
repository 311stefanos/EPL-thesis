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
from typing import TypedDict, Literal, List, Optional, Annotated, Union, Dict, Any, Tuple
from pydantic import BaseModel, Field
from operator import add

# General imports
from dotenv import load_dotenv
from pathlib import Path
from time import sleep
from datetime import datetime
from openpyxl import Workbook, load_workbook
import base64
import csv
import math
import re
import requests
import traceback
import json
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS, read_state_file, clean_llm_output
from experiments.ablation_study.no_implementation import receipt_tracker_assistant_prompts as prompts



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'

PROJECT_DIR = Path(__file__).resolve().parent # Project folder: receipts.xlsx and conversions.csv live here
MAX_TOOL_ROUNDS = 10 # Bounded tool-loop guard: max tool-call rounds per run before a final text reply is forced
_CURRENCY_SYMBOLS = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY', '₹': 'INR'} # Symbols the vision model may return instead of ISO codes



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} receipt_tracker_assistant') if DEBUG else None



""" Schemas """

class AgentSchema(MessagesState):
	"""
	LangGraph graph state for receipt_tracker_assistant (reactive_conversational, memory=true)
	"""
	latest: str # Text of the single outbound AI reply produced this run: consent preview + one consent question, post-consent confirmation, discard notice ('nothing was saved'), receipts-based answer, general reply, or error reply (with nothing written). Written by the chat node in postprocess (and best-effort by its error handler) so exactly one reply is sent per run before the run ends.
	pending_receipt: Optional[List[Dict[str, Any]]] # Consent gate: prepared receipts.xlsx row(s) awaiting explicit user consent — one plain dict per receipt photo (single-receipt messages are a one-element list; multi-filepath messages store all prepared rows behind one combined preview and one consent question). Each dict holds the six receipts.xlsx columns — cost (numeric EUR), date (ISO 'YYYY-MM-DD HH:MM'), items ('item1 (quantity x price currency), ...' preserving the ORIGINAL currency inside the string, e.g. 'Latte (1 x 3.50 USD)'), location (merchant), category (free-form; groceries/dining/transport are examples only, any fitting label allowed), comments (optional) — plus optional traceability metadata (original_currency, source_filepath) never written to Excel; append_receipt_row consumes each dict directly. Set when the chat node sends the consent preview and ends the run; on the next run a clear affirmative newest Human message appends each dict via append_receipt_row, while a refusal, an unclear answer, or any non-ingestion turn clears it to None. Nothing is written to receipts.xlsx while rows only live here.
	last_seen_event_id: Optional[str] # Dedupe key for retried inbound events, persisted per the workflow's memory contract ('checkpointer persists messages, last_seen_event_id, pending_receipt'). If an inbound event's id equals this value, the run terminates silently — no reply and no repeated side effects; otherwise it is updated to the current event's id after processing. The chat node itself implements no dedupe (per its detailed annotation); this key keeps the persisted-state contract complete for the duplicate-event guard.




''' Tools '''
@tool
def extract_receipt(filepath: str) -> dict:
	"""
	Overview: Extracts structured receipt data from a single receipt-photo file using an OpenRouter vision model (e.g., a gpt-4o-class vision model; model choice is flexible as long as it reliably performs receipt OCR/extraction). It reads the image from disk, base64-encodes it, sends it to the OpenRouter vision model with an extraction instruction, and returns the total cost, receipt date, line items, and merchant/location. This is the first step of receipt ingestion; the calling LLM afterwards picks a fitting (free-form) category, calls convert_to_eur for the total, and composes the consent preview. The OpenRouter API key is read from the environment INSIDE this tool (OPENROUTER_API_KEY) and is never echoed in any return value, message, or reply.
	
	Caller LLM: chat_llm (the 'chat' node's LLM, invoked with bind_tools([extract_receipt, convert_to_eur, append_receipt_row, query_receipts])).
	
	Outside-the-Tool Work (Tool Handler Function Responsibilities): The ToolNode executes this tool and appends its returned dict as a ToolMessage to state['messages']. The chat node (LLM) then: picks a free-form category (groceries/dining/transport are examples only), calls convert_to_eur(amount, currency, date), enforces the ingestion hard guard (parseable total + date AND an available EUR rate), builds the consent preview, stores the prepared row in state['pending_receipt'], writes the single outbound reply to state['latest'] and state['messages'], and ends the run. On an error result, the LLM replies with an error message and writes nothing.
	
	Inside-the-Tool Work (Tool Responsibilities): Validates that filepath is a non-empty string pointing to an existing, readable image file (jpg/jpeg/png/webp/heic); reads OPENROUTER_API_KEY from the environment and fails gracefully with an error dict if missing; base64-encodes the image and calls the OpenRouter vision model via HTTP; parses the model's JSON/text response into a structured dict; normalizes the date to ISO 'YYYY-MM-DD HH:MM' when possible (preserving whatever the receipt shows, filling unknown time with 00:00); returns line items as a list of {name, quantity, price, currency} plus the total's currency. Never writes to receipts.xlsx or conversions.csv. Never returns or logs the API key.
	
	Instructions:
	1. Validate filepath: must be a non-empty string; resolve it with Path(); check the file exists and is readable; check the extension is a supported image type. On failure return {'status': 'error', 'error': '<reason>', 'filepath': filepath}.
	2. Read OPENROUTER_API_KEY from os.environ. If absent/empty, return {'status': 'error', 'error': 'OpenRouter API key not configured', 'filepath': filepath}.
	3. Read the image bytes and base64-encode them; determine the MIME type from the extension.
	4. Send a request to the OpenRouter chat-completions endpoint with a vision-capable model, attaching the image as a data URL and an instruction to extract, as strict JSON: total (number), currency (ISO code, e.g. 'USD', 'EUR'), date (ISO 'YYYY-MM-DD HH:MM', using 00:00 if the receipt shows no time), items (list of {name, quantity, price, currency}), location/merchant (string), and a confidence/readability flag.
	5. Parse the model response into a dict. If the response is unparseable, the image is unreadable, or the total/date are missing or ambiguous, return {'status': 'error', 'error': '<reason: unreadable or ambiguous>', 'filepath': filepath} — do NOT invent values.
	6. On success return {'status': 'ok', 'filepath': filepath, 'total': <float>, 'currency': <str>, 'date': 'YYYY-MM-DD HH:MM', 'items': [{'name': str, 'quantity': number, 'price': float, 'currency': str}, ...], 'location': <merchant str>}.
	7. Keep the return compact and JSON-serializable; never include the API key, raw HTTP headers, or full raw model output.
	
	State Updates (on the caller function): None — the tool cannot see or change the graph state. The ToolNode appends the returned dict as a ToolMessage; the chat node afterwards updates state['messages'], state['latest'], and (on a successful preview) state['pending_receipt'].
	
	Args:
	- filepath (str): Absolute or relative path to a single receipt-photo image file (jpg/jpeg/png/webp/heic) in the project folder or filesystem.
	
	Returns:
	- dict: On success {'status': 'ok', 'filepath': str, 'total': float, 'currency': str, 'date': 'YYYY-MM-DD HH:MM', 'items': list[dict], 'location': str}. On failure {'status': 'error', 'error': str, 'filepath': str}.
	"""
	supported_extensions = {'jpg', 'jpeg', 'png', 'webp', 'heic'}
	mime_by_extension = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp', 'heic': 'image/heic'}
	try:
		if not isinstance(filepath, str) or not filepath.strip():
			return {'status': 'error', 'error': 'filepath must be a non-empty string', 'filepath': filepath}
		path = Path(filepath).expanduser().resolve()
		if not path.is_file():
			return {'status': 'error', 'error': 'File not found', 'filepath': filepath}
		if not os.access(str(path), os.R_OK):
			return {'status': 'error', 'error': 'File is not readable', 'filepath': filepath}
		extension = path.suffix.lower().lstrip('.')
		if extension not in supported_extensions:
			return {'status': 'error', 'error': f'Unsupported image type ".{extension}" (supported: jpg, jpeg, png, webp, heic)', 'filepath': filepath}
		if path.stat().st_size > 20 * 1024 * 1024:
			return {'status': 'error', 'error': 'Image file too large (max 20 MB)', 'filepath': filepath}

		api_key = (os.environ.get('OPENROUTER_API_KEY') or '').strip()
		if not api_key:
			return {'status': 'error', 'error': 'OpenRouter API key not configured', 'filepath': filepath}

		mime_type = mime_by_extension[extension]
		image_base64 = base64.b64encode(path.read_bytes()).decode('ascii')

		instruction = (
			"Extract the data from this receipt photo. Respond with STRICT JSON only (no markdown fences, no commentary) "
			"exactly in this schema: "
			'{"total": <final amount paid as a number>, '
			'"currency": "<3-letter ISO 4217 code, e.g. USD, EUR>", '
			'"date": "<receipt date and time as YYYY-MM-DD HH:MM; use 00:00 for the time if the receipt shows none>", '
			'"items": [{"name": "<item name>", "quantity": <number>, "price": <number>, "currency": "<ISO code>"}], '
			'"location": "<merchant / store name>", '
			'"confidence": "<high|medium|low>"} '
			"Rules: 'total' is the grand total actually paid (including tax/service). Use the receipt's original currency "
			"for the total and for every item price. If the image is unreadable, ambiguous, or not a receipt, set "
			"'confidence' to 'low'. Never invent values."
		)
		model = (os.environ.get('OPENROUTER_VISION_MODEL') or 'openai/gpt-4o-mini').strip()
		payload = {
			'model': model,
			'messages': [{
				'role': 'user',
				'content': [
					{'type': 'text', 'text': instruction},
					{'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{image_base64}'}},
				],
			}],
			'temperature': 0.0,
			'max_tokens': 1200,
		}
		headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
		response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json=payload, timeout=60)
		if response.status_code != 200:
			return {'status': 'error', 'error': f'Vision model request failed (HTTP {response.status_code})', 'filepath': filepath}

		body = response.json()
		choices = body.get('choices') or [{}]
		raw_content = (choices[0].get('message') or {}).get('content', '')
		if isinstance(raw_content, list):
			raw_content = ' '.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in raw_content)
		if not isinstance(raw_content, str) or not raw_content.strip():
			return {'status': 'error', 'error': 'Empty response from vision model', 'filepath': filepath}

		extracted = _parse_json_object(raw_content)
		if extracted is None:
			return {'status': 'error', 'error': 'Could not parse the vision model response as JSON', 'filepath': filepath}
		if 'total' not in extracted and isinstance(extracted.get('receipt'), dict):
			extracted = extracted['receipt']

		confidence = str(extracted.get('confidence', 'high') or 'high').strip().lower()
		if confidence in ('low', 'none', 'unreadable') or extracted.get('readable') is False:
			return {'status': 'error', 'error': 'Receipt image unreadable or extraction ambiguous', 'filepath': filepath}

		raw_total = extracted.get('total')
		if isinstance(raw_total, str):
			raw_total = raw_total.strip().replace('€', '').replace('$', '').replace('£', '').replace(' ', '')
			if ',' in raw_total and '.' in raw_total:
				raw_total = raw_total.replace(',', '')
			elif ',' in raw_total:
				raw_total = raw_total.replace(',', '.')
		try:
			total = float(raw_total)
		except (TypeError, ValueError):
			return {'status': 'error', 'error': 'Receipt total is missing or not parseable', 'filepath': filepath}
		if not math.isfinite(total) or total < 0:
			return {'status': 'error', 'error': 'Receipt total is missing or not parseable', 'filepath': filepath}

		currency = str(extracted.get('currency', '') or '').strip()
		currency = _CURRENCY_SYMBOLS.get(currency, currency).upper()
		if len(currency) != 3 or not currency.isalpha():
			return {'status': 'error', 'error': 'Receipt currency is missing or invalid', 'filepath': filepath}

		parsed_date = _parse_flexible_date(str(extracted.get('date', '') or ''))
		if parsed_date is None:
			return {'status': 'error', 'error': 'Receipt date is missing or not parseable', 'filepath': filepath}
		date_str = parsed_date.strftime('%Y-%m-%d %H:%M')

		items: List[Dict[str, Any]] = []
		raw_items = extracted.get('items')
		if isinstance(raw_items, list):
			for entry in raw_items:
				if not isinstance(entry, dict):
					continue
				name = str(entry.get('name', '') or '').strip()
				if not name:
					continue
				try:
					quantity = float(entry.get('quantity', 1) or 1)
				except (TypeError, ValueError):
					quantity = 1.0
				try:
					price = float(entry.get('price', 0) or 0)
				except (TypeError, ValueError):
					price = 0.0
				if quantity.is_integer():
					quantity = int(quantity)
				item_currency = str(entry.get('currency', '') or currency).strip()
				item_currency = _CURRENCY_SYMBOLS.get(item_currency, item_currency).upper()
				if not (len(item_currency) == 3 and item_currency.isalpha()):
					item_currency = currency
				items.append({'name': name, 'quantity': quantity, 'price': round(price, 2), 'currency': item_currency})

		location = str(extracted.get('location', '') or extracted.get('merchant', '') or '').strip()

		return {
			'status': 'ok',
			'filepath': filepath,
			'total': round(total, 2),
			'currency': currency,
			'date': date_str,
			'items': items,
			'location': location,
		}
	except Exception as exc:
		return {'status': 'error', 'error': f'Receipt extraction failed: {exc}', 'filepath': filepath if isinstance(filepath, str) else ''}

@tool
def convert_to_eur(amount: float, currency: str, date: str) -> dict:
	"""
	Overview: The SINGLE conversion tool that fully encapsulates conversions.csv. Converts a monetary amount in a given currency to EUR using the free, key-less Frankfurter API (api.frankfurter.dev, ECB-based). For past-dated receipts it requests the historical rate for that date so past receipts convert at that date's rate. Every freshly fetched rate is appended to conversions.csv (columns: currency, rate, date fetched) for persistence. If the Frankfurter API is unreachable or fails, the tool falls back to the most recent cached rate for that currency found in conversions.csv rather than failing. The calling agent/LLM NEVER touches conversions.csv directly — all CSV reading/writing happens inside this tool.
	
	Caller LLM: chat_llm (the 'chat' node's LLM, invoked with bind_tools([extract_receipt, convert_to_eur, append_receipt_row, query_receipts])).
	
	Outside-the-Tool Work (Tool Handler Function Responsibilities): The ToolNode executes this tool and appends its returned dict as a ToolMessage to state['messages']. The chat node (LLM) then uses the returned EUR amount to build the consent preview (EUR cost, date, merchant, items, category, original currency), enforces the ingestion hard guard (a EUR rate must be available fresh or cached — an 'error' result with no rate means ingestion aborts with an error reply and nothing written), stores the prepared row in state['pending_receipt'], writes the single outbound reply to state['latest'] and state['messages'], and ends the run.
	
	Inside-the-Tool Work (Tool Responsibilities): Validates inputs (positive finite amount, 3-letter currency code, date parseable as 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'); if currency is already 'EUR', returns the amount unchanged with source 'same_currency' (no API call, no CSV write); otherwise queries the Frankfurter API (historical endpoint when date is in the past, latest otherwise) for the currency->EUR rate; on success appends {'currency', 'rate', 'date_fetched'} to conversions.csv (creating the file with a header if missing) and returns the converted amount with source 'fresh'; on API failure reads conversions.csv and returns the most recent cached rate for that currency with source 'cached' (and a note that the rate may be stale); if neither the API nor a cached rate is available, returns an error dict. Never echoes secrets; never writes to receipts.xlsx.
	
	Instructions:
	1. Validate amount: must be a finite number > 0; currency: uppercase 3-letter ISO code; date: parse 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' into a date object. On invalid input return {'status': 'error', 'error': '<reason>'}.
	2. If currency == 'EUR', return {'status': 'ok', 'amount_eur': amount, 'rate': 1.0, 'source': 'same_currency', 'currency': 'EUR', 'date': date} without any API call or CSV write.
	3. Build the Frankfurter request: for a past date use the historical endpoint (e.g., GET https://api.frankfurter.dev/v1/{date}?base={currency}&symbols=EUR); for today use the latest endpoint. Use a short timeout (a few seconds).
	4. On a successful response, extract the rate (units of EUR per 1 unit of currency), compute amount_eur = round(amount * rate, 2), append a row {'currency': currency, 'rate': rate, 'date_fetched': <today's 'YYYY-MM-DD HH:MM'>} to conversions.csv (create the file with header 'currency,rate,date_fetched' if it does not exist), and return {'status': 'ok', 'amount_eur': float, 'rate': float, 'source': 'fresh', 'currency': currency, 'date': date}.
	5. On API failure (network error, non-200, timeout, unsupported currency/date), open conversions.csv (if it exists), find the most recent row (by date_fetched) whose currency matches, compute amount_eur from that rate, and return {'status': 'ok', 'amount_eur': float, 'rate': float, 'source': 'cached', 'currency': currency, 'date': date, 'note': 'API unavailable; used cached rate from <date_fetched>'}.
	6. If the API failed AND no cached rate exists for the currency, return {'status': 'error', 'error': 'No EUR rate available (API failed and no cached rate)'} — the caller must then abort ingestion with an error reply and write nothing.
	7. Keep the return compact and JSON-serializable; never return raw HTTP bodies or file contents.
	
	State Updates (on the caller function): None — the tool cannot see or change the graph state. The ToolNode appends the returned dict as a ToolMessage; the chat node afterwards updates state['messages'], state['latest'], and (on a successful preview) state['pending_receipt'].
	
	Args:
	- amount (float): The monetary amount to convert (e.g., the receipt total in its original currency). Must be a positive finite number.
	- currency (str): ISO 4217 currency code of the amount (e.g., 'USD', 'GBP', 'SEK'). 'EUR' short-circuits with no conversion.
	- date (str): The receipt date in ISO format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'. Past dates use Frankfurter's historical rate for that date.
	
	Returns:
	- dict: On success {'status': 'ok', 'amount_eur': float, 'rate': float, 'source': 'fresh'|'cached'|'same_currency', 'currency': str, 'date': str, 'note': str (optional)}. On failure {'status': 'error', 'error': str}.
	"""
	try:
		try:
			amount = float(amount)
		except (TypeError, ValueError):
			return {'status': 'error', 'error': 'amount must be a positive finite number'}
		if isinstance(amount, bool) or not math.isfinite(amount) or amount <= 0:
			return {'status': 'error', 'error': 'amount must be a positive finite number'}

		if not isinstance(currency, str):
			return {'status': 'error', 'error': 'currency must be a 3-letter ISO code'}
		currency = currency.strip().upper()
		if len(currency) != 3 or not currency.isalpha():
			return {'status': 'error', 'error': 'currency must be a 3-letter ISO code'}

		if not isinstance(date, str) or not date.strip():
			return {'status': 'error', 'error': "date must be 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'"}
		parsed_date = _parse_flexible_date(date.strip())
		if parsed_date is None:
			return {'status': 'error', 'error': "date must be 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'"}
		date_str = parsed_date.strftime('%Y-%m-%d %H:%M')

		if currency == 'EUR':
			return {'status': 'ok', 'amount_eur': round(amount, 2), 'rate': 1.0, 'source': 'same_currency', 'currency': 'EUR', 'date': date_str}

		rate: Optional[float] = None
		try:
			if parsed_date.date() < datetime.now().date():
				url = f'https://api.frankfurter.dev/v1/{parsed_date.date().isoformat()}?base={currency}&symbols=EUR'
			else:
				url = f'https://api.frankfurter.dev/v1/latest?base={currency}&symbols=EUR'
			response = requests.get(url, timeout=10)
			if response.status_code == 200:
				data = response.json()
				fetched = (data.get('rates') or {}).get('EUR') if isinstance(data, dict) else None
				if isinstance(fetched, (int, float)) and fetched > 0:
					rate = float(fetched)
		except (requests.RequestException, ValueError, AttributeError):
			rate = None

		if rate is not None:
			_append_conversion_rate(currency, rate)
			return {'status': 'ok', 'amount_eur': round(amount * rate, 2), 'rate': rate, 'source': 'fresh', 'currency': currency, 'date': date_str}

		cached = _latest_cached_rate(currency)
		if cached is not None:
			cached_rate, cached_date = cached
			return {'status': 'ok', 'amount_eur': round(amount * cached_rate, 2), 'rate': cached_rate, 'source': 'cached', 'currency': currency, 'date': date_str, 'note': f'API unavailable; used cached rate from {cached_date}'}

		return {'status': 'error', 'error': 'No EUR rate available (API failed and no cached rate)'}
	except Exception as exc:
		return {'status': 'error', 'error': f'Conversion to EUR failed: {exc}'}

@tool
def append_receipt_row(row: dict, filepath: str = '') -> dict:
	"""
	Overview: Creates (if missing) and appends one consented row to receipts.xlsx in the project folder. Called ONLY after the user has explicitly consented in the conversation (the newest Human message clearly affirms saving the prepared row(s) from state['pending_receipt']). The Excel file has the fixed columns: cost (numeric EUR), date (ISO 'YYYY-MM-DD HH:MM'), items ('item1 (quantity x price currency), ...' preserving the ORIGINAL currency inside the string), location (merchant), category (free-form), comments (optional free-text, e.g., original currency or conversion notes). This is the finalizing side effect of the consent turn; the calling LLM wraps the returned status into the single outbound confirmation reply and ends the run.
	
	Caller LLM: chat_llm (the 'chat' node's LLM, invoked with bind_tools([extract_receipt, convert_to_eur, append_receipt_row, query_receipts])).
	
	Outside-the-Tool Work (Tool Handler Function Responsibilities): The ToolNode executes this tool and appends its returned dict as a ToolMessage to state['messages']. The chat node (LLM) then: on 'ok', writes the short confirmation to state['latest'] and state['messages'] and clears state['pending_receipt'] to None (the row is now persisted); on 'error', writes an error reply (nothing was saved) and keeps or clears pending_receipt per the conversation; then ends the run. The chat node is also responsible for having obtained explicit consent BEFORE calling this tool and for calling it once per prepared row when the user consented to a multi-receipt batch.
	
	Inside-the-Tool Work (Tool Responsibilities): Validates the row dict (required keys: cost as a finite number in EUR, date matching 'YYYY-MM-DD HH:MM', items as a non-empty string, location as a string, category as a non-empty string; optional comments as a string; extra traceability keys like original_currency/source_filepath are ignored and never written); creates receipts.xlsx with the header row if it does not exist; appends exactly one data row with the six columns in order; returns a compact status dict. Performs no conversion, no API calls, and never modifies existing rows.
	
	Instructions:
	1. Validate row: it must be a dict containing 'cost' (int/float, finite, >= 0, already in EUR), 'date' (string matching 'YYYY-MM-DD HH:MM'), 'items' (non-empty string in the format 'item1 (quantity x price currency), ...' with original currency preserved), 'location' (string; empty string allowed if merchant unknown), 'category' (non-empty, free-form string). 'comments' is optional (string, may be empty). Unknown extra keys (e.g., 'original_currency', 'source_filepath') are ignored and NOT written to Excel. On invalid input return {'status': 'error', 'error': '<reason>', 'saved': False} without touching the file.
	2. Resolve the Excel path to receipts.xlsx in the project folder. If the file does not exist, create it with the header: cost, date, items, location, category, comments.
	3. Append one row with the six values in column order: cost as a numeric EUR amount, date as the ISO 'YYYY-MM-DD HH:MM' string, items/location/category/comments as strings.
	4. Save the workbook and verify the write succeeded (e.g., row count increased). On any I/O error return {'status': 'error', 'error': '<reason>', 'saved': False} — never write partial or corrupted data.
	5. On success return {'status': 'ok', 'saved': True, 'filepath': <resolved path>, 'row': {'cost': float, 'date': str, 'items': str, 'location': str, 'category': str, 'comments': str}}.
	6. Keep the return compact and JSON-serializable.
	
	State Updates (on the caller function): None — the tool cannot see or change the graph state. The ToolNode appends the returned dict as a ToolMessage; the chat node afterwards writes the confirmation to state['latest'] and state['messages'] and clears state['pending_receipt'] to None after a successful append (or handles the error reply with nothing saved).
	
	Args:
	- row (dict): The prepared receipts.xlsx row: {'cost': float (EUR), 'date': 'YYYY-MM-DD HH:MM', 'items': 'item1 (quantity x price currency), ...', 'location': str, 'category': str, 'comments': str (optional)}. Extra traceability keys are ignored.
	- filepath (str): Optional explicit path for the Excel file; defaults to receipts.xlsx in the project folder when empty or omitted.
	
	Returns:
	- dict: On success {'status': 'ok', 'saved': True, 'filepath': str, 'row': dict}. On failure {'status': 'error', 'error': str, 'saved': False}.
	"""
	try:
		if isinstance(row, str):
			row = _parse_json_object(row)
		if not isinstance(row, dict):
			return {'status': 'error', 'error': 'row must be a dict', 'saved': False}

		cost = row.get('cost')
		if isinstance(cost, bool) or not isinstance(cost, (int, float)):
			return {'status': 'error', 'error': "row['cost'] must be a numeric EUR amount", 'saved': False}
		cost = float(cost)
		if not math.isfinite(cost) or cost < 0:
			return {'status': 'error', 'error': "row['cost'] must be a finite number >= 0", 'saved': False}

		date_value = row.get('date')
		if not isinstance(date_value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', date_value.strip()):
			return {'status': 'error', 'error': "row['date'] must be an ISO 'YYYY-MM-DD HH:MM' string", 'saved': False}
		date_value = date_value.strip()
		try:
			datetime.strptime(date_value, '%Y-%m-%d %H:%M')
		except ValueError:
			return {'status': 'error', 'error': "row['date'] is not a valid date/time", 'saved': False}

		items = row.get('items')
		if not isinstance(items, str) or not items.strip():
			return {'status': 'error', 'error': "row['items'] must be a non-empty string like 'item1 (quantity x price currency), ...'", 'saved': False}

		location = row.get('location')
		if location is None:
			location = ''
		if not isinstance(location, str):
			return {'status': 'error', 'error': "row['location'] must be a string", 'saved': False}

		category = row.get('category')
		if not isinstance(category, str) or not category.strip():
			return {'status': 'error', 'error': "row['category'] must be a non-empty string", 'saved': False}

		comments = row.get('comments')
		if comments is None:
			comments = ''
		if not isinstance(comments, str):
			return {'status': 'error', 'error': "row['comments'] must be a string", 'saved': False}

		xlsx_path = _receipts_xlsx_path(filepath)
		header = ['cost', 'date', 'items', 'location', 'category', 'comments']
		try:
			xlsx_path.parent.mkdir(parents=True, exist_ok=True)
			if not xlsx_path.exists():
				workbook = Workbook()
				workbook.active.append(header)
				workbook.save(str(xlsx_path))
			workbook = load_workbook(str(xlsx_path))
			sheet = workbook.active
			rows_before = sheet.max_row or 0
			sheet.append([cost, date_value, items, location, category, comments])
			workbook.save(str(xlsx_path))
			workbook.close()
			verification = load_workbook(str(xlsx_path), read_only=True)
			rows_after = verification.active.max_row or 0
			verification.close()
			if rows_after <= rows_before:
				return {'status': 'error', 'error': 'Write verification failed: row count did not increase', 'saved': False}
		except OSError as io_error:
			return {'status': 'error', 'error': f'Excel I/O error: {io_error}', 'saved': False}

		return {
			'status': 'ok',
			'saved': True,
			'filepath': str(xlsx_path),
			'row': {'cost': cost, 'date': date_value, 'items': items, 'location': location, 'category': category, 'comments': comments},
		}
	except Exception as exc:
		return {'status': 'error', 'error': f'Failed to append the receipt row: {exc}', 'saved': False}

@tool
def query_receipts(query: dict) -> dict:
	"""
	Overview: Reads receipts.xlsx (the agent's own persisted data) and answers structured data needs for spending questions: filtering by category and/or date range, summing costs, counting rows, listing matching rows, and ranking items or rows by spend (e.g., top-N items). This tool reads ONLY receipts.xlsx — no web search, no external data. It returns compact, structured numbers that the calling LLM cites in a concise English answer. Non-terminal: after receiving results, the LLM composes the final user-facing reply and ends the run.
	
	Caller LLM: chat_llm (the 'chat' node's LLM, invoked with bind_tools([extract_receipt, convert_to_eur, append_receipt_row, query_receipts])).
	
	Outside-the-Tool Work (Tool Handler Function Responsibilities): The ToolNode executes this tool and appends its returned dict as a ToolMessage to state['messages']. The chat node (LLM) then composes the single outbound English answer citing the returned numbers (including data appended earlier in the same run or in earlier runs), writes it to state['latest'] and state['messages'], and ends the run. If the tool returns an error (e.g., no receipts file yet), the LLM replies that no data is available rather than guessing.
	
	Inside-the-Tool Work (Tool Responsibilities): Validates the query dict; loads receipts.xlsx from the project folder (returning an empty-result dict with a note if the file does not exist yet — this is not an error, it just means no data); parses the cost column as numeric EUR and the date column as ISO 'YYYY-MM-DD HH:MM'; applies optional filters (category substring/case-insensitive match, date_from/date_to inclusive bounds); computes the requested operation: 'sum' (total EUR), 'count' (number of matching rows), 'list' (the matching rows, capped), 'rank_items' (aggregate spend per item name parsed from the items strings, sorted descending, top-N), or 'rank_rows' (matching rows sorted by cost descending, top-N); returns a compact dict with the matching row count, the computed numbers, and (for list/rank) the requested entries. Performs no writes and no network calls.
	
	Instructions:
	1. Validate query: it must be a dict. Recognized keys: 'operation' (one of 'sum', 'count', 'list', 'rank_items', 'rank_rows'; default 'sum'), 'category' (optional str, case-insensitive substring match against the category column), 'date_from' and 'date_to' (optional 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM', inclusive), 'top_n' (optional int, default 3, used by rank operations), 'item' (optional str, substring match against the items column, for targeted questions like 'how much did I spend on coffee?'). Unknown keys are ignored. On a fundamentally invalid query return {'status': 'error', 'error': '<reason>'}.
	2. Load receipts.xlsx from the project folder. If it does not exist, return {'status': 'ok', 'rows_matched': 0, 'result': None, 'note': 'No receipts recorded yet'}.
	3. Parse each row: cost as float (EUR), date as datetime from 'YYYY-MM-DD HH:MM', items/location/category/comments as strings. Skip malformed rows and count them.
	4. Apply filters: category substring (case-insensitive), date range inclusive on both ends, optional item substring in the items string.
	5. Compute the operation over the filtered rows: 'sum' -> total cost rounded to 2 decimals; 'count' -> number of rows; 'list' -> up to 50 matching rows as compact dicts; 'rank_items' -> parse 'item (quantity x price currency)' entries, aggregate total spend per item name across matching rows, return the top 'top_n' as [{'item': str, 'total_eur': float}]; 'rank_rows' -> the top 'top_n' rows by cost as compact dicts.
	6. Return {'status': 'ok', 'rows_matched': int, 'operation': str, 'result': <number | list>, 'skipped_rows': int (optional)}.
	7. Keep the return compact and JSON-serializable; never return the raw file or full unfiltered dumps.
	
	State Updates (on the caller function): None — the tool cannot see or change the graph state. The ToolNode appends the returned dict as a ToolMessage; the chat node afterwards writes the final English answer citing the numbers to state['latest'] and state['messages'] and ends the run.
	
	Args:
	- query (dict): Structured query: {'operation': 'sum'|'count'|'list'|'rank_items'|'rank_rows' (optional, default 'sum'), 'category': str (optional filter), 'item': str (optional items-string filter), 'date_from': str (optional inclusive 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'), 'date_to': str (optional inclusive), 'top_n': int (optional, default 3, for rank operations)}.
	
	Returns:
	- dict: On success {'status': 'ok', 'rows_matched': int, 'operation': str, 'result': float | list[dict] | None, 'note': str (optional), 'skipped_rows': int (optional)}. On invalid input {'status': 'error', 'error': str}.
	"""
	try:
		if isinstance(query, str):
			query = _parse_json_object(query)
		if not isinstance(query, dict):
			return {'status': 'error', 'error': 'query must be a dict'}

		operation = str(query.get('operation') or 'sum').strip().lower()
		if operation not in ('sum', 'count', 'list', 'rank_items', 'rank_rows'):
			return {'status': 'error', 'error': "operation must be one of 'sum', 'count', 'list', 'rank_items', 'rank_rows'"}

		try:
			top_n = int(query.get('top_n', 3))
		except (TypeError, ValueError):
			top_n = 3
		if top_n <= 0:
			top_n = 3

		category_filter = query.get('category')
		category_filter = category_filter.strip().lower() if isinstance(category_filter, str) and category_filter.strip() else None
		item_filter = query.get('item')
		item_filter = item_filter.strip().lower() if isinstance(item_filter, str) and item_filter.strip() else None

		raw_date_from = str(query.get('date_from') or '').strip()
		date_from = None
		if raw_date_from:
			date_from = _parse_flexible_date(raw_date_from)
			if date_from is None:
				return {'status': 'error', 'error': "date_from must be 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'"}
		raw_date_to = str(query.get('date_to') or '').strip()
		date_to = None
		if raw_date_to:
			date_to = _parse_flexible_date(raw_date_to)
			if date_to is None:
				return {'status': 'error', 'error': "date_to must be 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'"}
			if ':' not in raw_date_to:
				date_to = date_to.replace(hour=23, minute=59, second=59)

		xlsx_path = _receipts_xlsx_path('')
		if not xlsx_path.exists():
			return {'status': 'ok', 'rows_matched': 0, 'result': None, 'note': 'No receipts recorded yet'}

		rows, skipped_rows = _load_receipt_rows(xlsx_path)

		filtered: List[Dict[str, Any]] = []
		for row in rows:
			if category_filter and category_filter not in row['category'].lower():
				continue
			if item_filter and item_filter not in row['items'].lower():
				continue
			if date_from is not None and row['date'] < date_from:
				continue
			if date_to is not None and row['date'] > date_to:
				continue
			filtered.append(row)

		result: Any = None
		if operation == 'sum':
			result = round(sum(row['cost'] for row in filtered), 2)
		elif operation == 'count':
			result = len(filtered)
		elif operation == 'list':
			result = [_compact_row(row) for row in filtered[:50]]
		elif operation == 'rank_items':
			result = _rank_items(filtered, top_n)
		elif operation == 'rank_rows':
			result = [_compact_row(row) for row in sorted(filtered, key=lambda entry: entry['cost'], reverse=True)[:top_n]]

		response: Dict[str, Any] = {'status': 'ok', 'rows_matched': len(filtered), 'operation': operation, 'result': result}
		if skipped_rows:
			response['skipped_rows'] = skipped_rows
		return response
	except Exception as exc:
		return {'status': 'error', 'error': f'Receipts query failed: {exc}'}




''' LLM '''
chat_llm = myChatOpenAI(
	temperature= 0.0
).bind_tools([extract_receipt, convert_to_eur, append_receipt_row, query_receipts])

# No-tools fallback LLM: used by the chat node only when the bounded tool-loop guard triggers,
# to force a final text reply (it cannot request further tool calls).
final_answer_llm = myChatOpenAI(
	temperature= 0.0
)




''' Helpful Functions '''

_DATE_FORMATS = (
	'%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M', '%Y-%m-%dT%H:%M:%S',
	'%Y-%m-%d', '%Y/%m/%d %H:%M', '%Y/%m/%d', '%Y.%m.%d',
	'%d.%m.%Y %H:%M', '%d.%m.%Y', '%d/%m/%Y %H:%M', '%d/%m/%Y',
	'%m/%d/%Y %H:%M', '%m/%d/%Y', '%d-%m-%Y %H:%M', '%d-%m-%Y',
	'%B %d, %Y', '%b %d, %Y', '%d %B %Y', '%d %b %Y',
)

_ITEM_ENTRY_PATTERN = re.compile(r'(?P<name>[^,()]+?)\s*\(\s*(?P<quantity>\d+(?:[.,]\d+)?)\s*[x×*]\s*(?P<price>\d+(?:[.,]\d+)?)\s*(?P<currency>[A-Za-z]{3})?\s*\)')


def _parse_flexible_date(value: str) -> Optional[datetime]:
	"""Parse a date/datetime string in the formats receipts commonly use ('YYYY-MM-DD', 'YYYY-MM-DD HH:MM', and a few regional variants); returns a datetime (missing time = 00:00) or None."""
	if not isinstance(value, str):
		return None
	text = value.strip()
	if not text:
		return None
	for date_format in _DATE_FORMATS:
		try:
			return datetime.strptime(text, date_format)
		except ValueError:
			continue
	return None


def _parse_json_object(text: Any) -> Optional[Dict[str, Any]]:
	"""Best-effort parse of the first JSON object found inside an LLM/HTTP response (tolerates surrounding prose and markdown fences); returns a dict or None."""
	if not isinstance(text, str):
		return None
	cleaned = text.strip()
	if not cleaned:
		return None
	start = cleaned.find('{')
	end = cleaned.rfind('}')
	if start == -1 or end <= start:
		return None
	try:
		parsed = json.loads(cleaned[start:end + 1])
	except json.JSONDecodeError:
		return None
	return parsed if isinstance(parsed, dict) else None


def _receipts_xlsx_path(filepath: str = '') -> Path:
	"""Resolve the receipts.xlsx path: an explicit filepath when given, otherwise receipts.xlsx in the project folder."""
	if isinstance(filepath, str) and filepath.strip():
		return Path(filepath.strip()).expanduser()
	return PROJECT_DIR / 'receipts.xlsx'


def _conversions_csv_path() -> Path:
	"""Resolve the conversions.csv path in the project folder."""
	return PROJECT_DIR / 'conversions.csv'


def _append_conversion_rate(currency: str, rate: float) -> None:
	"""Append a freshly fetched exchange rate to conversions.csv (columns: currency, rate, date_fetched), creating the file with its header if missing. Best-effort: persistence failures never break the conversion itself."""
	csv_path = _conversions_csv_path()
	date_fetched = datetime.now().strftime('%Y-%m-%d %H:%M')
	try:
		file_exists = csv_path.exists()
		with open(csv_path, 'a', encoding='utf-8', newline='') as csv_file:
			writer = csv.writer(csv_file)
			if not file_exists:
				writer.writerow(['currency', 'rate', 'date_fetched'])
			writer.writerow([currency, rate, date_fetched])
	except OSError:
		pass


def _latest_cached_rate(currency: str) -> Optional[Tuple[float, str]]:
	"""Return the most recent cached (rate, date_fetched) for a currency from conversions.csv, or None when no cached rate exists."""
	csv_path = _conversions_csv_path()
	if not csv_path.exists():
		return None
	best: Optional[Tuple[datetime, float, str]] = None
	try:
		with open(csv_path, 'r', encoding='utf-8', newline='') as csv_file:
			for row in csv.DictReader(csv_file):
				if not isinstance(row, dict) or str(row.get('currency', '')).strip().upper() != currency:
					continue
				try:
					row_rate = float(row.get('rate'))
				except (TypeError, ValueError):
					continue
				row_date_raw = str(row.get('date_fetched', '') or '').strip()
				row_date = _parse_flexible_date(row_date_raw) or datetime.min
				if best is None or row_date >= best[0]:
					best = (row_date, row_rate, row_date_raw)
	except OSError:
		return None
	return (best[1], best[2]) if best is not None else None


def _load_receipt_rows(xlsx_path: Path) -> Tuple[List[Dict[str, Any]], int]:
	"""Load all data rows from receipts.xlsx as normalized dicts (cost float EUR, date datetime, items/location/category/comments strings); returns (rows, skipped_count). Malformed rows are skipped and counted."""
	rows: List[Dict[str, Any]] = []
	skipped = 0
	workbook = load_workbook(str(xlsx_path), read_only=True, data_only=True)
	try:
		sheet = workbook.active
		header: Optional[List[str]] = None
		for values in sheet.iter_rows(values_only=True):
			if values is None or all(cell is None for cell in values):
				continue
			if header is None:
				header = [str(cell).strip().lower() if cell is not None else '' for cell in values]
				continue
			record = dict(zip(header, values))
			try:
				cost = float(record.get('cost'))
				if not math.isfinite(cost):
					raise ValueError('non-finite cost')
			except (TypeError, ValueError):
				skipped += 1
				continue
			raw_date = record.get('date')
			parsed_date = _parse_flexible_date(str(raw_date)) if raw_date is not None else None
			if parsed_date is None:
				skipped += 1
				continue
			rows.append({
				'cost': cost,
				'date': parsed_date,
				'items': str(record.get('items') or ''),
				'location': str(record.get('location') or ''),
				'category': str(record.get('category') or ''),
				'comments': str(record.get('comments') or ''),
			})
	finally:
		workbook.close()
	return rows, skipped


def _parse_items_string(items: str) -> List[Tuple[str, float, float, str]]:
	"""Parse an items string like 'Latte (1 x 3.50 USD), Croissant (2 x 2.25 EUR)' into (name, quantity, price, currency) tuples."""
	parsed: List[Tuple[str, float, float, str]] = []
	if not isinstance(items, str):
		return parsed
	for match in _ITEM_ENTRY_PATTERN.finditer(items):
		name = match.group('name').strip(' -–—:')
		if not name:
			continue
		try:
			quantity = float(match.group('quantity').replace(',', '.'))
			price = float(match.group('price').replace(',', '.'))
		except ValueError:
			continue
		parsed.append((name, quantity, price, (match.group('currency') or 'EUR').upper()))
	return parsed


def _rank_items(rows: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
	"""Aggregate spend per item name across rows and return the top_n as [{'item': str, 'total_eur': float}]. Each row's EUR cost is split across its parsed items in proportion to each item's quantity x price share of the row, so mixed original currencies still yield EUR totals."""
	totals: Dict[str, float] = {}
	display_names: Dict[str, str] = {}
	for row in rows:
		entries = _parse_items_string(row['items'])
		if not entries:
			continue
		entry_values = [(name, quantity * price) for name, quantity, price, _currency in entries]
		row_total = sum(value for _name, value in entry_values)
		if row_total <= 0:
			continue
		for name, value in entry_values:
			if value <= 0:
				continue
			key = name.strip().lower()
			totals[key] = totals.get(key, 0.0) + row['cost'] * (value / row_total)
			display_names.setdefault(key, name.strip())
	ranked = sorted(totals.items(), key=lambda entry: entry[1], reverse=True)[:top_n]
	return [{'item': display_names.get(key, key), 'total_eur': round(total, 2)} for key, total in ranked]


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
	"""Compact JSON-serializable view of a receipts.xlsx row for list/rank results."""
	return {
		'cost': row['cost'],
		'date': row['date'].strftime('%Y-%m-%d %H:%M'),
		'items': row['items'],
		'location': row['location'],
		'category': row['category'],
		'comments': row['comments'],
	}




''' Nodes '''
def chat(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. The single conversation-driven agent step; the system rules are soft guidelines and the conversation drives what happens each run. First branches on persisted state, not on any mode field. (1) If state.pending_receipt exists, interpret the newest Human message as the consent decision for that prepared row: on clear affirmation call append_receipt_row(pending_receipt) to create/append receipts.xlsx (ISO YYYY-MM-DD HH:MM date, numeric EUR cost, original currency preserved inside the items string and optionally comments) and reply with a short confirmation; on refusal or an unclear answer, discard the pending receipt, reply that nothing was saved, and — conversation-driven — treat any new request contained in the same message as a fresh turn. (2) If no pending_receipt exists, classify intent from the newest Human message plus the full persisted conversation: (a) receipt ingestion — call extract_receipt (OpenRouter vision model, user-provided key from env) for cost, date, items, location; pick a fitting category with a flexible taxonomy (groceries/dining/transport are examples only); call convert_to_eur(amount, currency, date), which internally queries the Frankfurter API (historical rate for past-dated receipts), appends every fetched rate to conversions.csv, and on API failure reads the most recent cached rate from conversions.csv — the agent itself never touches conversions.csv. Per the user's requirement the agent must NOT append automatically: it sends one preview message (EUR cost, date, merchant, items, category, original currency) asking for consent to save, stores the prepared row in state.pending_receipt (for multi-filepath messages it prepares all rows, shows one combined preview, and asks one consent question), and ends the run. Hard guards on ingestion: proceed only if extraction yields a parseable total and date AND a EUR rate is available (fresh or cached); if extraction is unreadable/ambiguous or the key is missing, reply with an error and write nothing. (b) Spending questions — call query_receipts to read/filter/sum/rank receipts.xlsx only (no web search) and cite the numbers, including data appended in earlier runs or earlier in the same run. (c) General chat — reply conversationally. The agent may combine actions adaptively (e.g., answer a question in the same reply as a consent preview, or answer a question right after a consented append). Sends exactly one user-visible AI message per run; never blocks or waits for user input — needing a reply is always send + end.

	Execution: LLM+TOOLS. The single conversation-driven agent node of receipt_tracker_assistant (reactive_conversational). One invocation == one inbound user message (natural-language text and/or one or more receipt-photo filepaths). This is intentionally a plain LLM node with tools: format the prompt with the few pieces of information it needs, bind the receipt tools, and let the LLM decide everything from the conversation. There is NO mode/next_action, NO pending_receipt, NO event-id/dedupe logic, and NO pre-collected run context in state — the persisted conversation (state["messages"], persisted by the MemorySaver checkpointer) is the only memory. The consent gate for writing to receipts.xlsx lives entirely in the conversation: when a prior run sent a preview asking "save this?", the user's yes/no is simply the newest HumanMessage, and the LLM handles it (append_receipt_row + confirmation, or a reply that nothing was saved) like any other turn. The OpenRouter API key is NEVER placed in the prompt, state, messages, or any reply — it is read from the environment only inside the extract_receipt tool.

	Step-by-step:
	1) Preprocess (minimal): obtain the current date/time directly from the datetime module — e.g., datetime.now().strftime('%Y-%m-%d %H:%M') — for the prompt's format argument. Nothing else is gathered: no extra state reads, no key checks, no helpers.
	2) Format prompts.CHAT_PROMPT inline with that value (CHAT_PROMPT.format(current_datetime=...)). The prompt template carries only the soft rules and invariants: reply in English; receipt extraction via extract_receipt (OpenRouter vision; the user-provided key is read from env inside the tool and never echoed); category is free-form (groceries/dining/transport are examples only, any fitting category allowed); conversion via convert_to_eur (Frankfurter fetch with cached fallback; conversions.csv is handled entirely inside the tool); append to receipts.xlsx only after the user explicitly consents in the conversation (show a preview first, append on the next affirmative message); spending Q&A via query_receipts reading receipts.xlsx only (no web search), concise answers citing the numbers; dates ISO 'YYYY-MM-DD HH:MM'; costs numeric EUR; original currency preserved inside the items string/comments; exactly one user-visible reply per run.
	3) Invoke: chat_llm.bind_tools([extract_receipt, convert_to_eur, append_receipt_row, query_receipts]) and safe_invoke(chat_llm, messages=[SystemMessage(content=prompt), *state["messages"]]).
	4) Tool execution via ToolNode (NOT an inline loop): when the model returns tool calls, hand them to a ToolNode constructed with the same tools — ToolNode([extract_receipt, convert_to_eur, append_receipt_row, query_receipts]). The ToolNode executes the requested tools and appends their outputs as ToolMessages (canonical wiring: a dedicated 'tools' ToolNode node with a conditional edge chat -> tools -> chat; never a manual inline for-loop over tool calls). The LLM is then re-invoked with the updated messages until it produces the final AI message (bounded iterations). Typical flows the LLM chooses on its own: (a) receipt ingestion — extract_receipt per filepath, pick a category, convert_to_eur(amount, currency, date), then reply with one preview (EUR cost, date, merchant, items, category, original currency) asking for consent, without appending; (b) consent turn — on clear affirmation call append_receipt_row and confirm, otherwise reply that nothing was saved (handling any new request in the same message); (c) spending question — query_receipts (filter/sum/rank) and cite the numbers; (d) general chat — reply conversationally. Mixed receipt+question messages are handled in one run and one reply, receipt first by default.
	5) Postprocess: take the final AI message text, write it to state["latest"], append the AIMessage to state["messages"], and return the updated state so the checkpointer persists the conversation for the next run.
	6) Error handling: on an unexpected exception, log it (DEBUG) and best-effort write an error reply to state["latest"] and state["messages"] so the user still gets exactly one reply; tool failures (unreadable/ambiguous receipt, missing key inside the tool, no EUR rate available fresh or cached) surface as ToolMessages and the LLM replies with an error, writing nothing; never write corrupted data to receipts.xlsx.

	Inputs (state):
	- state["messages"]: List[BaseMessage] — the full persisted conversation; the newest HumanMessage is this run's inbound event (text and/or receipt filepath(s)) and, together with the history, the sole basis for every decision (intent, consent, category, guards).
	- Environment (read only inside the tools, never in this node's prompt or state): OPENROUTER_API_KEY for extract_receipt; project-folder paths receipts.xlsx and conversions.csv are created/managed by the tools.

	Outputs (state):
	- state["messages"]: updated with this run's ToolMessages (if any) and the single outbound AIMessage reply.
	- state["latest"]: str — text of the single outbound AI reply for this run (preview + consent question, post-consent confirmation, discard notice, receipts-based answer, general reply, or error reply).

	Tools (requested by the LLM via chat_llm.bind_tools([...]); executed by a ToolNode, which calls the tools and appends their output as ToolMessages — never an inline loop):
	- extract_receipt(filepath: str) -> dict — OpenRouter vision extraction of total cost, date, line items, and merchant/location from one receipt photo (reads the user-provided key from env internally; the key is never echoed).
	- convert_to_eur(amount: float, currency: str, date: str) -> dict — single conversion tool fully encapsulating conversions.csv: fresh Frankfurter fetch (historical rates supported), appends fetched rates to conversions.csv, falls back to the most recent cached rate on API failure; the agent never touches the CSV.
	- append_receipt_row(row: dict) -> dict — creates/appends receipts.xlsx in the project folder (auto-created if missing) with a consented row (numeric EUR cost, ISO date, items string preserving original currency, location, category, optional comments).
	- query_receipts(query: dict) -> dict — reads/filters/sums/ranks receipts.xlsx only (no web search).

	Helpful functions: none required — the node is a plain format-prompt (date from the datetime module) -> bind-tools -> invoke -> ToolNode tool execution -> postprocess step.
	"""

    print_function_name()
    try:
        # Preprocess (minimal): only the current date/time is gathered; the persisted conversation is the sole memory.
        current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M')
        prompt = prompts.CHAT_PROMPT.format(current_datetime=current_datetime)

        messages = list(state.get('messages') or [])

        # Bounded tool-loop guard: count tool-call rounds since the newest Human message;
        # past the bound, force a final text reply with the no-tools LLM.
        tool_rounds = 0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, AIMessage) and will_tool_call([message]): # TODO: Changed to [message]
                tool_rounds += 1
        llm = chat_llm if tool_rounds < MAX_TOOL_ROUNDS else final_answer_llm

        result = safe_invoke(llm, messages=[SystemMessage(content=prompt), *messages])
        if result is None:
            raise RuntimeError('LLM invocation returned no result')
        if isinstance(result, AIMessage):
            ai_message = result
        else:
            content = getattr(result, 'content', None)
            ai_message = AIMessage(content=content if isinstance(content, str) else str(result))

        # Tool calls are executed by the dedicated 'tools' ToolNode (chat -> tools -> chat), never inline.
        if will_tool_call([ai_message]): # TODO: Changed to [ai_message]
            return {'messages': [ai_message]}

        # Postprocess: final AI reply -> state['latest'] + state['messages'].
        content = ai_message.content
        if isinstance(content, list):
            text = ' '.join(block.get('text', '') if isinstance(block, dict) else str(block) for block in content)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        try:
            cleaned = clean_llm_output(text)
            text = cleaned if isinstance(cleaned, str) else text
        except Exception:
            pass
        if not text.strip():
            text = 'Sorry — I could not generate a reply. Please try again.'
        return {'messages': [AIMessage(content=text)], 'latest': text}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Best-effort error reply so the user still gets exactly one reply; nothing was written.
        error_text = 'Sorry — something went wrong while processing your message. Nothing was saved. Please try again.'
        try:
            return {'messages': [AIMessage(content=error_text)], 'latest': error_text}
        except Exception:
            return state


def route_chat(state: AgentSchema) -> Literal['tools', 'end']:
    """ Conditional routing after the 'chat' node: route to the 'tools' ToolNode when the newest AI message requests tool calls, otherwise end the run. Deterministic fallback: anything that is not an AIMessage with tool calls routes to END. """
    messages = state.get('messages') or []
    if messages and isinstance(messages[-1], AIMessage) and will_tool_call([messages[-1]]): # TODO: Changed to [messages[-1]]
        return 'tools'
    return 'end'





''' Graph '''
receipt_tracker_assistant_graph = StateGraph(AgentSchema)

receipt_tracker_assistant_graph.add_node("chat", chat)
receipt_tracker_assistant_graph.add_node("tools", ToolNode([extract_receipt, convert_to_eur, append_receipt_row, query_receipts]))

receipt_tracker_assistant_graph.add_edge(START, "chat")
receipt_tracker_assistant_graph.add_conditional_edges("chat", route_chat, {"tools": "tools", "end": END})
receipt_tracker_assistant_graph.add_edge("tools", "chat")


receipt_tracker_assistant_app = receipt_tracker_assistant_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    import uuid

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'no_implementation_test',
            'run_name': 'no_implementation_test',
            'thread_id': f'no_implementation_test:{uuid.uuid4()}',
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

        response = receipt_tracker_assistant_app.invoke(
            {
                'messages': [HumanMessage(content=message)]
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

        user_in = input(f'\n{GREEN}[USER INPUT]{RESET} > ')