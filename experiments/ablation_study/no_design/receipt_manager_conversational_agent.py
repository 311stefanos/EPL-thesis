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
from experiments.ablation_study.no_design import receipt_manager_conversational_agent_prompts as prompts

import base64
import requests
import pandas as pd
from datetime import datetime



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} receipt_manager_conversational_agent') if DEBUG else None



""" Schemas """

class ReceiptExtraction(BaseModel):
	"""
	Pydantic schema for the structured output of the vision model inside the ocr_receipt tool. One instance per receipt photo; fields may be None when not visible on the receipt (the agent handles missing time/date/location gracefully).
	"""
	usable: bool # Whether usable data could be extracted from the photo; False triggers the single polite re-request.
	total_cost: Optional[float] # Receipt total in the original currency.
	currency: Optional[str] # Original currency code/symbol of the amounts on the receipt (e.g. 'USD', 'GBP').
	date: Optional[str] # Receipt date as a raw string exactly as printed (e.g. '2024-05-03', '03/05/24').
	time: Optional[str] # Receipt time as a raw string exactly as printed (e.g. '14:32'); None if not visible.
	items: Optional[List[Dict[str, Any]]] # Line items, each a dict with keys 'name' (str), 'quantity' (float/int), 'price' (float, original currency).
	location: Optional[str] # Store/vendor name or address as printed on the receipt; None if not visible.




class AgentSchema(MessagesState):
	"""
	Main LangGraph state schema for the receipt_manager_conversational_agent. Extends MessagesState for persistent multi-turn message history via the MemorySaver checkpointer.
	"""
	messages: Annotated[List[BaseMessage], add] # Persistent conversation history (HumanMessage inputs, AIMessage replies, ToolMessages); required for multi-turn context and very-recent-information Q&A.
	intent: Optional[Literal['process_receipt', 'answer_question', 'process_receipt_with_notes', 'duplicate', 'none']] # Intent classified from the inbound payload: photo only, text only, photo+text (text becomes comments/context), duplicate no-op, or none/unset.
	mode: str # High-level run mode, e.g. 'idle' or 'processing'; persisted so the next run resumes correctly.
	next_action: Optional[Literal['awaiting_clear_photo', 'none']] # Pending follow-up action; 'awaiting_clear_photo' after a failed OCR so the next inbound photo resumes OCR without an internal retry loop.
	pending_question: Optional[str] # The polite re-request text awaiting a clearer photo, persisted across runs.
	last_seen_event_id: Optional[str] # Identifier of the last processed inbound event, used for duplicate detection (no-op on duplicates).
	event_id: Optional[str] # Identifier of the current inbound event, supplied by the caller in the input payload; compared against last_seen_event_id for duplicate detection.
	photo_paths: Optional[List[str]] # Receipt photo file path(s) extracted from the current inbound payload; multiple paths yield one row per readable receipt.
	user_text: Optional[str] # Accompanying text message from the current payload; used as the user's question or as supplementary notes for the comments column.
	session_receipts: Annotated[List[Dict[str, Any]], add] # Receipt rows successfully stored during the current session, so Q&A can combine receipts.xlsx data with just-processed receipts without re-reading photos.
	reply: Optional[str] # The single user-visible reply for this run (confirmation, answer, or re-request); None for duplicate no-op runs.




''' Tools '''
@tool
def ocr_receipt(image_path: str) -> str:
	"""
	Overview: OCR a receipt photo via an OpenRouter vision model (Qwen2.5-VL or Gemini Flash) and return the extracted structured data as a JSON string conforming to the ReceiptExtraction schema.
	    Caller LLM: chat_llm (the chat node's agent LLM, via tool calling).
	    Outside-the-Tool Work (Tool Handler Function Responsibilities): The tool handler loads the image file, base64-encodes it, instantiates/calls the OpenRouter vision-capable LLM with structured output (ReceiptExtraction), and returns the tool's JSON string to the LLM as a ToolMessage. The handler does NOT convert currencies, assign categories, or write to Excel — that happens in the chat node.
	    Inside-the-Tool Work (Tool Responsibilities): Send the image to the vision model with an extraction prompt; receive structured output (usable, total_cost, currency, date, time, items with name/quantity/price, location); serialize it to a JSON string. Set 'usable': false and null fields when nothing legible can be extracted.
	    Instructions: Call once per photo path, only after the path has been validated. Never call for text-only questions. Do not retry within the tool on failure — return an unusable result and let the agent send its single polite re-request.
	    State Updates (on the caller function): None directly; the chat node reads the returned JSON to drive conversion, formatting, and row appending, and sets next_action/pending_question when 'usable' is false.
	    Args: image_path (str): validated local file path of the receipt photo.
	    Returns: str: JSON string of the ReceiptExtraction result (usable, total_cost, currency, date, time, items, location).
	"""
	print_function_name() if DEBUG else None
	try:
		# Validate the path exists and is a file
		if not os.path.isfile(image_path):
			return json.dumps({'usable': False, 'error': f'Image file not found: {image_path}'})

		# Read the image bytes and base64-encode them
		with open(image_path, 'rb') as f:
			image_bytes: bytes = f.read()
		if not image_bytes:
			return json.dumps({'usable': False, 'error': f'Image file is empty: {image_path}'})
		b64: str = base64.b64encode(image_bytes).decode('utf-8')

		# Guess the MIME type from the file extension
		ext: str = os.path.splitext(image_path)[1].lower().lstrip('.')
		mime: str = {
			'jpg': 'image/jpeg',
			'jpeg': 'image/jpeg',
			'png': 'image/png',
			'webp': 'image/webp',
			'heic': 'image/heic',
		}.get(ext, 'image/jpeg')

		# Strict-JSON extraction prompt for the vision model
		extraction_prompt: str = (
			"You are a receipt OCR assistant. Extract the following information from the receipt photo and return STRICT JSON only — no markdown, no code fences, no explanations, no invented values.\n"
			"JSON keys (exactly these):\n"
			'- "usable": bool — true only if legible receipt data could be extracted, otherwise false.\n'
			'- "total_cost": float or null — the receipt total in the original currency.\n'
			'- "currency": string or null — the original currency code or symbol of the amounts (e.g. "USD", "GBP", "€").\n'
			'- "date": string or null — the receipt date exactly as printed (e.g. "2024-05-03", "03/05/24").\n'
			'- "time": string or null — the receipt time exactly as printed (e.g. "14:32"); null if not visible.\n'
			'- "items": list or null — line items, each an object with keys "name" (string), "quantity" (number), "price" (number, original currency); null if not visible.\n'
			'- "location": string or null — the store/vendor name or address as printed on the receipt; null if not visible.\n'
			"If nothing legible can be extracted, set \"usable\" to false and all other keys to null. Do not guess or fabricate values."
		)

		# Build the multimodal message (text prompt + data-URI image) and call the vision LLM once
		vision_llm = myChatOpenAI(temperature= 0)
		message = HumanMessage(content= [
			{'type': 'text', 'text': extraction_prompt},
			{'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}},
		])
		response: BaseMessage = safe_invoke(vision_llm, messages= [message])

		# Normalize the model output to plain text
		content = response.content
		if isinstance(content, list):
			content = ' '.join(
				part.get('text', '') for part in content
				if isinstance(part, dict) and part.get('type') == 'text'
			)
		text: str = str(content).strip()

		# Strip markdown code fences if present
		if text.startswith('```'):
			text = text.split('\n', 1)[1] if '\n' in text else text.lstrip('`')
		if text.rstrip().endswith('```'):
			text = text.rstrip()[:-3].rstrip()

		# Fallback: slice out the outermost JSON object if prose surrounds it
		if not text.startswith('{'):
			start: int = text.find('{')
			end: int = text.rfind('}')
			if start != -1 and end > start:
				text = text[start: end + 1]

		# Parse and normalize to the ReceiptExtraction key set
		data: Dict[str, Any] = json.loads(text)
		extraction: Dict[str, Any] = {
			'usable': bool(data.get('usable', False)),
			'total_cost': data.get('total_cost'),
			'currency': data.get('currency'),
			'date': data.get('date'),
			'time': data.get('time'),
			'items': data.get('items'),
			'location': data.get('location'),
		}
		return json.dumps(extraction)

	except Exception as e:
		print(f'{RED}[TOOL] [ERR] ocr_receipt{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		# No retry inside the tool — return an unusable result and let the agent send its single polite re-request
		return json.dumps({'usable': False, 'error': f'OCR failed: {e}'})

@tool
def fetch_fx_rate(currency: str) -> str:
	"""
	Overview: Obtain the newest available exchange rate for a currency against EUR, trying a free FX API first (e.g. frankfurter.app or exchangerate.host) and falling back to the local `convertions.csv` cache; on online success, update the cache.
	    Caller LLM: chat_llm (the chat node's agent LLM, via tool calling).
	    Outside-the-Tool Work (Tool Handler Function Responsibilities): The tool handler passes the returned rate string back to the LLM as a ToolMessage. The chat node then uses the rate with convert_amount_to_eur; the tool itself never touches graph state.
	    Inside-the-Tool Work (Tool Responsibilities): (1) Skip the network and return the cached rate if currency is EUR. (2) Query the FX API for currency->EUR; on success, write/update the row in convertions.csv in the format 'eur,<currency>,<rate>' (1 EUR = rate units of currency) and return the fresh rate. (3) On any online failure (timeout, HTTP error, unknown currency), read the rate from convertions.csv (pre-seeded with standard conversions to EUR) and return it, indicating it came from cache. (4) If neither source yields a rate, return an error string.
	    Instructions: Call once per distinct original currency per receipt before converting amounts. Rate semantics are 'eur,dollar,0.9' = 1 EUR = 0.9 USD, so foreign amounts are divided by the returned rate.
	    State Updates (on the caller function): None directly; the chat node consumes the rate string for EUR conversion and may note 'rate from cache' in comments if relevant.
	    Args: currency (str): original currency code/symbol from the receipt (e.g. 'USD', 'GBP', 'EUR').
	    Returns: str: JSON string like '{"currency": "USD", "rate": 0.9, "source": "online"|"cache"}' or an error message if no rate could be obtained.
	"""
	print_function_name() if DEBUG else None

	# ---- Inner helpers (cache read/write/lookup logic kept in one place) ----
	def _read_rows(path: str) -> List[str]:
		try:
			with open(path, 'r', encoding= 'utf-8') as f:
				return [line.strip() for line in f if line.strip()]
		except OSError:
			return []

	def _write_rows(path: str, rows: List[str]) -> None:
		with open(path, 'w', encoding= 'utf-8') as f:
			f.write('\n'.join(rows) + '\n')

	def _find_eur_rate(rows: List[str], cur: str) -> Optional[float]:
		for row in rows:
			parts: List[str] = row.split(',')
			if len(parts) >= 3 and parts[0].strip().lower() == 'eur' and parts[1].strip().lower() == cur:
				try:
					return float(parts[2])
				except (ValueError, TypeError):
					continue
		return None

	try:
		# ---- 1) Normalize the currency input (aliases first, then strip symbols, then aliases again) ----
		ALIASES: Dict[str, str] = {
			'$': 'usd', '£': 'gbp', '€': 'eur', '¥': 'jpy',
			'dollar': 'usd', 'dollars': 'usd', 'pound': 'gbp', 'pounds': 'gbp', 'euro': 'eur',
		}
		raw: str = (currency or '').strip().lower()
		code: str = ALIASES.get(raw, ''.join(ch for ch in raw if ch.isalnum()))
		code = ALIASES.get(code, code)

		if not code:
			return json.dumps({'error': f'No rate available for {currency}'})

		# ---- 2) EUR shortcut: no network, no file I/O ----
		if code in ('eur', 'euro'):
			return json.dumps({'currency': 'EUR', 'rate': 1.0, 'source': 'online'})

		# ---- 3) Cache file: seed with standard conversions if missing ----
		cache_path: str = os.path.join(os.getcwd(), 'convertions.csv')
		SEED: List[str] = [
			# Units-per-EUR convention ('eur,<currency>,<rate>' = 1 EUR = rate units of currency),
			# matching the online path (rate = 1 / EUR-per-unit) and the divide-by-rate rule in convert_amount_to_eur
			'eur,usd,1.087', 'eur,gbp,0.855', 'eur,chf,0.962', 'eur,jpy,161.29', 'eur,cad,1.471',
			'eur,aud,1.639', 'eur,cny,7.692', 'eur,inr,90.909', 'eur,sek,11.364', 'eur,nok,11.765',
			'eur,dkk,7.463', 'eur,pln,4.348', 'eur,try,35.714', 'eur,brl,5.882', 'eur,mxn,18.519',
			'eur,zar,19.608', 'eur,huf,384.615', 'eur,czk,24.390', 'eur,ron,5.000', 'eur,bgn,1.961',
		]
		if not os.path.exists(cache_path):
			try:
				_write_rows(cache_path, SEED)
			except OSError:
				pass  # best-effort seeding; fallback read will simply miss

		# ---- 4) Online attempt: frankfurter.app, short timeout, graceful failure ----
		rate: Optional[float] = None
		try:
			response = requests.get(f'https://api.frankfurter.app/latest?from={code.upper()}&to=EUR', timeout= 5)
			if response.status_code == 200:
				data: Dict[str, Any] = response.json()
				per_unit: Optional[float] = data.get('rates', {}).get('EUR')  # EUR per 1 unit of `code`
				if per_unit:
					rate = 1.0 / per_unit  # 1 EUR = `rate` units of `code` (matches 'eur,<currency>,<rate>' semantics)
		except Exception:
			rate = None

		# ---- 5) Online success: update/append the cache row (best-effort) and return the fresh rate ----
		if rate is not None:
			try:
				rows: List[str] = _read_rows(cache_path)
				new_row: str = f'eur,{code},{rate}'
				for i, row in enumerate(rows):
					parts: List[str] = row.split(',')
					if len(parts) >= 3 and parts[0].strip().lower() == 'eur' and parts[1].strip().lower() == code:
						rows[i] = new_row
						break
				else:
					rows.append(new_row)
				_write_rows(cache_path, rows)
			except OSError:
				pass  # cache update is best-effort; still return the fresh rate
			return json.dumps({'currency': code.upper(), 'rate': rate, 'source': 'online'})

		# ---- 6) Online failure: fall back to the cached rate ----
		cached_rate: Optional[float] = _find_eur_rate(_read_rows(cache_path), code)
		if cached_rate is not None:
			return json.dumps({'currency': code.upper(), 'rate': cached_rate, 'source': 'cache'})

		# ---- 7) Neither source yielded a rate ----
		return json.dumps({'error': f'No rate available for {code.upper()}'})

	except Exception as e:
		print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		return json.dumps({'error': f'Failed to fetch FX rate for {currency}: {str(e)}'})

@tool
def append_receipt_row(cost: float, date: str, items: str, location: str, category: str, comments: str) -> str:
	"""
	Overview: Append exactly one row to the accumulating Excel file `receipts.xlsx` in the working directory, auto-creating the file with the fixed schema if it does not exist.
	    Caller LLM: chat_llm (the chat node's agent LLM, via tool calling).
	    Outside-the-Tool Work (Tool Handler Function Responsibilities): The tool handler returns the confirmation/error string to the LLM as a ToolMessage. The chat node decides what the user-visible confirmation says and records the stored row in state.session_receipts for same-session Q&A.
	    Inside-the-Tool Work (Tool Responsibilities): (1) If receipts.xlsx does not exist, create it with columns exactly ['cost', 'date', 'items', 'location', 'category', 'comments']. (2) Append a single row with the given values (cost in EUR, date as 'YYYY-MM-DD HH:MM', items as 'item1 (quantity x price EUR), ...'). (3) Save the file and return a success confirmation. (4) On any I/O error, return an error string without corrupting existing data.
	    Instructions: Call once per successfully processed receipt (multiple photos -> multiple calls). Never call for text-only questions or failed OCR. All monetary values must already be in EUR.
	    State Updates (on the caller function): None directly; the chat node appends the stored row dict to state.session_receipts so later questions in the same session can reference it without re-reading photos.
	    Args: cost (float): receipt total in EUR. date (str): normalized datetime 'YYYY-MM-DD HH:MM'. items (str): formatted items string 'item1 (quantity x price EUR), item2 (...)'. location (str): store/vendor name or address ('' if not visible). category (str): assigned category (standard set preferred, new category allowed). comments (str): original currency note plus user/agent notes.
	    Returns: str: confirmation message on success (e.g. row appended to receipts.xlsx) or an error description.
	"""
	print_function_name() if DEBUG else None

	file_path: str = os.path.join(os.getcwd(), 'receipts.xlsx')
	tmp_path: str = file_path + '.tmp.xlsx'

	try:
		# Build the typed row dict (cost as float, all others as str)
		row: Dict[str, Any] = {
			'cost': float(cost),
			'date': str(date),
			'items': str(items),
			'location': str(location),
			'category': str(category),
			'comments': str(comments),
		}

		# Read the existing file fully into memory first (original on disk untouched until the final write)
		if os.path.exists(file_path):
			df: pd.DataFrame = pd.read_excel(file_path, engine= 'openpyxl')
		else:
			df = pd.DataFrame(columns= ['cost', 'date', 'items', 'location', 'category', 'comments'])

		# Normalize the columns to the exact fixed schema (guards against schema drift in a hand-edited file)
		df = df.reindex(columns= ['cost', 'date', 'items', 'location', 'category', 'comments'])

		# Append the single new row in memory
		df = pd.concat([df, pd.DataFrame([row])], ignore_index= True)

		# Write to a temporary file first, then atomically swap it in so a failed/partial
		# write can never leave the original receipts.xlsx corrupted
		df.to_excel(tmp_path, index= False, engine= 'openpyxl')
		os.replace(tmp_path, file_path)

		print(f'{GREEN}[TOOL] [REST]{RESET} Row appended to receipts.xlsx: cost={row["cost"]}, date={row["date"]}, category={row["category"]}') if DEBUG else None
		return f"Row appended to receipts.xlsx: cost={row['cost']}, date={row['date']}, category={row['category']}"

	except Exception as e:
		print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None

		# Best-effort cleanup of the temporary file; cleanup errors must not mask the real error
		if os.path.exists(tmp_path):
			try:
				os.remove(tmp_path)
			except OSError:
				pass

		return f"Error appending row to receipts.xlsx: {e}"

@tool
def query_receipts(category: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
	"""
	Overview: Read rows from `receipts.xlsx` via pandas and return the matching records as a JSON string so the agent can compute answers to spending questions.
	    Caller LLM: chat_llm (the chat node's agent LLM, via tool calling).
	    Outside-the-Tool Work (Tool Handler Function Responsibilities): The tool handler returns the JSON string to the LLM as a ToolMessage. The chat node performs the actual aggregation/summary (totals, top items, comparisons) and composes the natural-language answer in the user's language, combining the result with state.session_receipts for very recent receipts.
	    Inside-the-Tool Work (Tool Responsibilities): (1) Load receipts.xlsx with pandas; return an empty-rows indicator if the file does not exist yet. (2) Apply optional filters: category (case-insensitive match), date range [start_date, end_date] inclusive on the 'date' column (YYYY-MM-DD HH:MM format). (3) Serialize the matching rows (all six columns) to a JSON string; cap the row count in the output to keep responses small (e.g. most recent 200 rows) and note if truncation occurred.
	    Instructions: Call for text-only spending questions; never re-read photos. Call once per question with the narrowest filters that satisfy it (e.g. category='Groceries' plus the current month's date range). Do not call when the answer is fully contained in the current conversation context.
	    State Updates (on the caller function): None directly; the chat node uses the returned rows to compute supporting numbers and draft the concise answer.
	    Args: category (Optional[str]): filter by category name (e.g. 'Groceries'); None for all categories. start_date (Optional[str]): inclusive range start 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'; None for no lower bound. end_date (Optional[str]): inclusive range end; None for no upper bound.
	    Returns: str: JSON string of matching receipt rows (cost, date, items, location, category, comments), or an empty/truncated indicator.
	"""
	print_function_name()
	try:
		# (1) Locate the accumulating Excel file; early-exit if it has not been created yet.
		path: str = os.path.join(os.getcwd(), 'receipts.xlsx')
		if not os.path.exists(path):
			return '{"rows": [], "note": "receipts.xlsx does not exist yet"}'

		# (2) Read the workbook.
		df: pd.DataFrame = pd.read_excel(path)
		if df.empty:
			return json.dumps({'rows': [], 'count': 0, 'truncated': False})

		# Defensive: guarantee all six schema columns exist so filtering/serialization never KeyErrors.
		for col in ['cost', 'date', 'items', 'location', 'category', 'comments']:
			if col not in df.columns:
				df[col] = None

		# Parse the 'date' column once; used for both filtering and sorting.
		dates: pd.Series = pd.to_datetime(df['date'], errors='coerce')

		# (3) Build the combined filter mask (all filters optional, date range inclusive).
		mask: pd.Series = pd.Series(True, index=df.index)

		if category:  # case-insensitive exact match on 'category'
			mask &= df['category'].astype(str).str.strip().str.lower() == str(category).strip().lower()

		if start_date and str(start_date).strip():  # inclusive lower bound ('YYYY-MM-DD' parses to 00:00, covering the whole day)
			start: pd.Timestamp = pd.to_datetime(str(start_date).strip())
			mask &= dates >= start

		if end_date and str(end_date).strip():  # inclusive upper bound
			end_str: str = str(end_date).strip()
			end: pd.Timestamp = pd.to_datetime(end_str)
			if ':' not in end_str:  # date-only 'YYYY-MM-DD' -> extend to end-of-day so the whole day is included
				end = end + pd.Timedelta(hours=23, minutes=59, seconds=59)
			mask &= dates <= end

		# Apply the filters.
		df = df.loc[mask]
		dates = dates.loc[mask]

		# (4) Sort by date ascending (NaT first so unparseable dates are dropped first on truncation)
		# and cap the output at the most recent 200 rows.
		sort_order: pd.Index = dates.sort_values(na_position='first').index
		df = df.loc[sort_order]
		dates = dates.loc[sort_order]

		total_matches: int = len(df)
		truncated: bool = total_matches > 200
		if truncated:
			df = df.tail(200)
			dates = dates.tail(200)

		# (5) Serialize: dates back to 'YYYY-MM-DD HH:MM', six columns in schema order, NaN -> null.
		df = df[['cost', 'date', 'items', 'location', 'category', 'comments']].copy()
		df['date'] = dates.dt.strftime('%Y-%m-%d %H:%M')

		rows: List[Dict[str, Any]] = []
		for record in df.to_dict(orient='records'):
			cleaned: Dict[str, Any] = {}
			for key, value in record.items():
				try:
					if pd.isna(value):
						cleaned[key] = None
						continue
				except (TypeError, ValueError):
					pass
				if hasattr(value, 'item'):  # numpy scalar -> native Python type for clean JSON
					try:
						value = value.item()
					except Exception:
						pass
				cleaned[key] = value
			rows.append(cleaned)

		payload: Dict[str, Any] = {'rows': rows, 'count': len(rows), 'truncated': truncated}
		if truncated:
			payload['note'] = f'output truncated to the most recent 200 of {total_matches} matching rows'
		return json.dumps(payload, default=str)

	except Exception as e:
		print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		return json.dumps({'rows': [], 'error': f'failed to query receipts: {e}'})
# TODO: Add Tools (if needed)



''' LLM '''
chat_llm = myChatOpenAI(
	temperature= 0.2
).bind_tools([ocr_receipt, fetch_fx_rate, append_receipt_row, query_receipts])




''' Helpful Functions '''
def classify_intent(photo_paths: Optional[List[str]], user_text: Optional[str]) -> str:
	"""
	Overview: Deterministically classify the inbound payload into one of the workflow intents.
	    Caller Node: chat (folded 'start' routing logic).
	    Instructions: Return 'process_receipt' if photo_paths is non-empty and user_text is empty/None; 'process_receipt_with_notes' if both are present; 'answer_question' if only user_text is present; 'none' if neither.
	    Args: photo_paths (Optional[List[str]]): receipt photo file path(s) from the payload. user_text (Optional[str]): accompanying text message.
	    Returns: str: one of 'process_receipt', 'answer_question', 'process_receipt_with_notes', 'none'.
	"""
	print_function_name() if DEBUG else None

	try:
		# Normalize: None, '' and [] are all treated as empty via plain truthiness
		has_photos: bool = bool(photo_paths)
		has_text: bool = bool(user_text)

		if has_photos and has_text:
			return 'process_receipt_with_notes'
		if has_photos:
			return 'process_receipt'
		if has_text:
			return 'answer_question'
		return 'none'

	except Exception as e:
		print(f'{RED}[FUNC] [ERR] classify_intent{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		return 'none'

def is_duplicate_event(event_id: Optional[str], last_seen_event_id: Optional[str]) -> bool:
	"""
	Overview: Determine whether the current inbound event is a duplicate of the last processed event.
	    Caller Node: chat (folded 'start' dedupe logic).
	    Instructions: Return True only if both event_id and last_seen_event_id are non-None and equal; a None event_id is never treated as a duplicate.
	    Args: event_id (Optional[str]): identifier of the current inbound event. last_seen_event_id (Optional[str]): identifier persisted from the previous run.
	    Returns: bool: True if the event is a duplicate no-op, False otherwise.
	"""
	return event_id is not None and last_seen_event_id is not None and event_id == last_seen_event_id

def validate_photo_path(path: str) -> bool:
	"""
	Overview: Verify that a receipt photo path exists on disk and points to a readable image file before OCR is attempted.
	    Caller Node: chat.
	    Instructions: Check that the path exists, is a file, and has an image-like extension (e.g. .jpg, .jpeg, .png, .webp, .heic); return False on any failure instead of raising.
	    Args: path (str): file path of the receipt photo.
	    Returns: bool: True if the path is an existing, readable-looking image file, False otherwise.
	"""
	print_function_name() if DEBUG else None
	try:
		# Defensive guard: reject None / non-str / empty inputs explicitly (redundant with the except, but self-documenting)
		if not isinstance(path, str) or not path:
			return False

		# Check that the path exists on disk
		if not os.path.exists(path):
			return False

		# Check that it is a regular file (not a directory or special file)
		if not os.path.isfile(path):
			return False

		# Check the extension (case-insensitive) against the allowed image set
		ext: str = os.path.splitext(path)[1].lower().lstrip('.')
		if ext not in {'jpg', 'jpeg', 'png', 'webp', 'heic'}:
			return False

		return True

	except Exception as e:
		print(f'{RED}[HELP] [ERR] validate_photo_path{RESET}', e) if DEBUG else None
		return False

def convert_amount_to_eur(amount: float, currency: str, rate: float) -> float:
	"""
	Overview: Convert a monetary amount from its original currency to EUR using a cached/fetched EUR-basis rate.
	    Caller Node: chat.
	    Instructions: If currency is EUR (case-insensitive), return the amount unchanged. Otherwise divide the amount by `rate`, where rate follows the convertions.csv semantics 'eur,<currency>,<rate>' meaning 1 EUR = rate units of <currency> (e.g. eur,dollar,0.9 -> 1 EUR = 0.9 USD, so USD amount / 0.9 = EUR). Round the result to 2 decimal places.
	    Args: amount (float): monetary amount in the original currency. currency (str): original currency code/symbol. rate (float): EUR-basis rate from fetch_fx_rate (1 EUR = rate units of currency).
	    Returns: float: the amount converted to EUR, rounded to 2 decimals.
	"""
	print_function_name() if DEBUG else None

	# Normalize the currency (case-insensitive, stripped; same idiom as fetch_fx_rate)
	cur: str = (currency or '').strip().lower()

	# EUR shortcut: return the amount unchanged — no rate applied, no rounding
	if cur in ('eur', 'euro'):
		return float(amount)

	# Safe fallback: a falsy rate (None / 0 / 0.0) would divide by zero — treat the amount as EUR-equivalent
	if not rate:
		return round(float(amount), 2)

	# 1 EUR = rate units of the original currency -> foreign amount / rate = EUR
	return round(float(amount) / float(rate), 2)

def format_receipt_datetime(date_str: Optional[str], time_str: Optional[str]) -> tuple:
	"""
	Overview: Normalize a raw receipt date/time into the storage format 'YYYY-MM-DD HH:MM'.
	    Caller Node: chat.
	    Instructions: Parse date_str tolerantly (common printed formats such as YYYY-MM-DD, DD/MM/YYYY, MM/DD/YY, '3 May 2024'). Append time_str parsed as HH:MM (or HH:MM:SS) when present; default to '00:00' when time is missing. If date_str is None or unparseable, use the current processing date and set date_was_missing to True so the caller can note it in comments. Return a (datetime_str, date_was_missing) tuple.
	    Args: date_str (Optional[str]): raw date as printed on the receipt; None if not visible. time_str (Optional[str]): raw time as printed on the receipt; None if not visible.
	    Returns: tuple: (datetime_str, date_was_missing) — datetime_str is the normalized 'YYYY-MM-DD HH:MM' string (00:00 default when time is missing; processing date when date is missing); date_was_missing is True when the receipt date was missing or unparseable.
	"""
	print_function_name() if DEBUG else None
	try:
		# ---- 1) Normalize inputs: non-string or empty/whitespace-only values are treated as missing ----
		date_raw: str = date_str.strip() if isinstance(date_str, str) else ''
		time_raw: str = time_str.strip() if isinstance(time_str, str) else ''

		# ---- 2) Tolerant date parsing: fromisoformat first, then the fixed printed-format list ----
		DATE_FORMATS: tuple = (
			'%Y-%m-%d', '%d/%m/%Y', '%d/%m/%y', '%m/%d/%Y', '%m/%d/%y',
			'%d-%m-%Y', '%Y/%m/%d', '%d.%m.%Y', '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y',
		)
		parsed_date: Optional[datetime] = None
		if date_raw:
			try:
				parsed_date = datetime.fromisoformat(date_raw)
			except (ValueError, TypeError):
				for fmt in DATE_FORMATS:
					try:
						parsed_date = datetime.strptime(date_raw, fmt)
						break
					except ValueError:
						continue

		# ---- 3) Missing/unparseable date: fall back to the current processing date and flag it ----
		date_was_missing: bool = parsed_date is None
		if date_was_missing:
			parsed_date = datetime.now()

		# ---- 4) Tolerant time parsing: combine hour/minute onto the parsed date, or default to 00:00 ----
		TIME_FORMATS: tuple = ('%H:%M', '%H:%M:%S', '%H.%M', '%I:%M %p')
		parsed_time: Optional[datetime] = None
		if time_raw:
			for fmt in TIME_FORMATS:
				try:
					parsed_time = datetime.strptime(time_raw, fmt)
					break
				except ValueError:
					continue

		if parsed_time is not None:
			parsed_date = parsed_date.replace(hour= parsed_time.hour, minute= parsed_time.minute, second= 0, microsecond= 0)
		else:
			parsed_date = parsed_date.replace(hour= 0, minute= 0, second= 0, microsecond= 0)

		# ---- 5) Serialize to the storage format ----
		datetime_str: str = parsed_date.strftime('%Y-%m-%d %H:%M')
		return (datetime_str, date_was_missing)

	except Exception as e:
		print(f'{RED}[HELPFUL FUNCTION] [ERR] format_receipt_datetime{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		# Never raise: degrade to the current processing date at 00:00, flagged as missing
		fallback: datetime = datetime.now().replace(hour= 0, minute= 0, second= 0, microsecond= 0)
		return (fallback.strftime('%Y-%m-%d %H:%M'), True)

def format_items_string(items: List[Dict[str, Any]], rate: float, currency: str) -> str:
	"""
	Overview: Render a receipt's extracted line items into the exact storage format 'item1 (quantity x price EUR), item2 (quantity x price EUR), ...'.
	    Caller Node: chat.
	    Instructions: For each item dict (keys 'name', 'quantity', 'price'), convert price to EUR via the divide-by-rate rule (use convert_amount_to_eur), round to 2 decimals, and emit '<name> (<quantity> x <price> EUR)'; join entries with ', ' in original order. Return an empty string for an empty list.
	    Args: items (List[Dict[str, Any]]): extracted line items with 'name', 'quantity', 'price'. rate (float): EUR-basis rate (1 EUR = rate units of currency). currency (str): original currency of the prices.
	    Returns: str: the formatted items string with all prices in EUR.
	"""
	print_function_name() if DEBUG else None

	# Empty/None items list -> empty string
	if not items:
		return ''

	parts: List[str] = []
	for item in items:
		# Skip non-dict entries (malformed OCR output)
		if not isinstance(item, dict):
			continue

		# Name: must be a non-empty string after stripping; otherwise skip the item
		raw_name = item.get('name')
		if not isinstance(raw_name, str):
			continue
		name: str = raw_name.strip()
		if not name:
			continue

		# Price: required; skip the item if missing/None or not numeric
		# (checked with `is None`, NOT falsiness, so a legitimate price of 0 is kept)
		raw_price = item.get('price')
		if raw_price is None:
			continue
		try:
			price: float = float(raw_price)
		except (TypeError, ValueError):
			continue

		# Quantity: default 1 when missing/None; keep as int if integral else float
		# (e.g. 2.0 -> 2 so it prints as '2', not '2.0'; 1.5 stays 1.5)
		quantity = item.get('quantity', 1)
		if quantity is None:
			quantity = 1
		try:
			q: float = float(quantity)
			quantity = int(q) if q.is_integer() else q
		except (TypeError, ValueError):
			pass  # unparseable quantity is printed as-is

		# Convert price to EUR (divide-by-rate rule, EUR shortcut, and rounding live in convert_amount_to_eur)
		price_eur: float = convert_amount_to_eur(price, currency, rate)

		# Emit '<name> (<quantity> x <price> EUR)' with the price at exactly 2 decimals (e.g. '3.50')
		parts.append(f'{name} ({quantity} x {price_eur:.2f} EUR)')

	return ', '.join(parts)

def build_receipt_row(extraction: Dict[str, Any], rate: float, user_notes: Optional[str]) -> Dict[str, Any]:
	"""
	Overview: Assemble the complete receipts.xlsx row (cost, date, items, location, category, comments) from an OCR extraction plus FX rate and user notes.
	    Caller Node: chat.
	    Instructions: Convert total_cost to EUR with convert_amount_to_eur, normalize the datetime with format_receipt_datetime, format items with format_items_string, and build the comments column as 'Original currency: <currency>' plus any user_notes and any agent flags (e.g. date missing, used processing date). Leave category to the LLM (standard set preferred, new category allowed); accept it as part of `extraction` or as a separate argument if simpler.
	    Args: extraction (Dict[str, Any]): structured OCR result (total_cost, currency, date, time, items, location). rate (float): EUR-basis rate for the original currency. user_notes (Optional[str]): accompanying user text to record in comments.
	    Returns: Dict[str, Any]: dict with keys 'cost', 'date', 'items', 'location', 'category', 'comments' matching the receipts.xlsx schema.
	"""
	print_function_name() if DEBUG else None

	# (1) Resolve the original currency; default to EUR when missing/None/empty
	currency: str = extraction.get('currency') or 'EUR'

	# (2) Convert the receipt total to EUR (EUR passes through unchanged; foreign amounts divided by the EUR-basis rate)
	cost_eur: float = convert_amount_to_eur(float(extraction.get('total_cost') or 0.0), currency, rate)

	# (3) Normalize the receipt date/time to 'YYYY-MM-DD HH:MM'; flag when the receipt date was missing/unparseable
	datetime_str: str
	date_was_missing: bool
	datetime_str, date_was_missing = format_receipt_datetime(extraction.get('date'), extraction.get('time'))

	# (4) Render line items as 'item1 (quantity x price EUR), ...' with prices converted to EUR
	items_str: str = format_items_string(extraction.get('items') or [], rate, currency)

	# (5) Location string ('' when not visible on the receipt)
	location: str = str(extraction.get('location') or '')

	# (6) Category: LLM-assigned value from the extraction when present, otherwise 'Other'
	category: str = str(extraction.get('category') or 'Other')

	# (7) Comments: mandatory original-currency note, then agent flags (missing date -> processing date), then user notes;
	# parts are joined with '; ' so the first part carries no leading separator
	comment_parts: List[str] = [f'Original currency: {currency}']
	if date_was_missing:
		comment_parts.append(f'date missing on receipt, used processing date {datetime.now().strftime("%Y-%m-%d")}')
	if user_notes and user_notes.strip():
		comment_parts.append(f'user notes: {user_notes}')
	comments: str = '; '.join(comment_parts)

	# Assemble the row dict matching the receipts.xlsx schema exactly
	row: Dict[str, Any] = {
		'cost': cost_eur,
		'date': datetime_str,
		'items': items_str,
		'location': location,
		'category': category,
		'comments': comments,
	}
	return row
# TODO: Add Helpful Functions (if needed)



''' Nodes '''
def route_after_chat(state: AgentSchema) -> Literal['ocr_receipt_handler', 'fetch_fx_rate_handler', 'append_receipt_row_handler', 'query_receipts_handler', 'done']:
    """
    Overview: Conditional-edge router for the 'chat' node. Inspects the latest message in the conversation and routes each tool call to its dedicated handler node (or ends the run).
        Caller Node: chat (via add_conditional_edges).
        Instructions: Read the last message from state['messages']. If it is not an AIMessage or carries no tool_calls, return 'done'. Otherwise take the FIRST tool call (last.tool_calls[0]) and map its 'name' to the corresponding handler node: 'ocr_receipt' -> 'ocr_receipt_handler', 'fetch_fx_rate' -> 'fetch_fx_rate_handler', 'append_receipt_row' -> 'append_receipt_row_handler', 'query_receipts' -> 'query_receipts_handler'. An unknown tool name also yields 'done'. Never raises: any unexpected error degrades to 'done'.
        Args: state (AgentSchema): the current graph state (MessagesState TypedDict; accessed via state['key'] / state.get('key', default)).
        Returns: Literal: the name of the next node — one of the four tool handler nodes, or 'done' (END).
    """
    try:
        # Read the conversation history (AgentSchema is a MessagesState TypedDict -> dict-style access)
        messages: List[BaseMessage] = state.get('messages', [])
        if not messages:
            return 'done'

        last: BaseMessage = messages[-1]

        # Only an AIMessage can request tool calls; anything else ends the run
        if not isinstance(last, AIMessage):
            return 'done'

        tool_calls = getattr(last, 'tool_calls', None)
        if not tool_calls:
            return 'done'

        # Route the FIRST tool call to its dedicated handler node
        tool_to_handler: Dict[str, str] = {
            'ocr_receipt': 'ocr_receipt_handler',
            'fetch_fx_rate': 'fetch_fx_rate_handler',
            'append_receipt_row': 'append_receipt_row_handler',
            'query_receipts': 'query_receipts_handler',
        }
        name: Optional[str] = tool_calls[0].get('name')
        return tool_to_handler.get(name, 'done')

    except Exception as e:
        print(f'{RED}[NODE] [ERR] route_after_chat{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return 'done'


def ocr_receipt_handler(state: AgentSchema) -> Dict[str, Any]:
	"""
	Overview: Tool-handler node for the `ocr_receipt` tool. Extracts the ocr_receipt tool call from the last message, invokes the tool on the validated photo path, and appends the resulting ToolMessage so the chat node's LLM loop can continue.
	    Caller Node: chat (routed here by route_after_chat when the latest AIMessage carries an ocr_receipt tool call).
	    Outside-the-Tool Work (Tool Handler Function Responsibilities): This node IS the tool handler: it reads the last message, locates the ocr_receipt tool call (checking last.tool_calls first, then last.additional_kwargs['tool_calls'] defensively), invokes ocr_receipt via .invoke(args), and appends the returned JSON string as a ToolMessage for the calling LLM. It does NOT convert currencies, assign categories, write to Excel, or set next_action/pending_question — the chat node reads this ToolMessage's JSON ('usable', 'total_cost', 'currency', 'date', 'time', 'items', 'location') to drive conversion, formatting, storage, and the single polite re-request when 'usable' is false.
	    Inside-the-Tool Work (Tool Responsibilities): None here beyond dispatch and relay — image loading, base64 encoding, the vision-model call, and JSON serialization all live inside the ocr_receipt tool.
	    Instructions: Find the tool call whose 'name' == 'ocr_receipt' by iterating the last message's tool calls; if none is found, return {} (no state change). Invoke the tool exactly once with its args dict and return {'messages': [ToolMessage]} — no other state keys are updated. Tool invocation is permitted only inside this handler node.
	    Error handling: On any exception, print the standard DEBUG error line (print(f'{RED}[NODE] [ERR] ocr_receipt_handler{RESET}', e) if DEBUG else None; traceback.print_exc() if DEBUG else None) and return a ToolMessage with the JSON error string {'usable': False, 'error': str(e)} for the same tool_call_id so the LLM loop can continue gracefully (the chat node treats it as an unusable OCR and sends its single polite re-request). If no tool_call_id was ever identified, return {}.
	    State keys read: messages.
	    State keys returned/updated: messages (one appended ToolMessage carrying the OCR JSON result or the error JSON).
	    Args: state (AgentSchema): LangGraph state (MessagesState-style TypedDict; accessed via state['key'] / state.get(key, default)).
	    Returns: Dict[str, Any]: {'messages': [ToolMessage]} on success or handled error; {} when no ocr_receipt tool call is present.
	"""
	print_function_name() if DEBUG else None

	tool_call_id: Optional[str] = None

	try:
		# ---- 1) Read the last message (defensive: empty history -> nothing to do) ----
		messages: List[BaseMessage] = state.get('messages', []) or []
		if not messages:
			return {}
		last: BaseMessage = messages[-1]

		# ---- 2) Extract the tool calls (standard .tool_calls first, additional_kwargs fallback) ----
		tool_calls: List[Any] = []
		from_kwargs: bool = False
		if getattr(last, 'tool_calls', None):
			tool_calls = list(last.tool_calls)
		else:
			from_kwargs = True
			tool_calls = list((getattr(last, 'additional_kwargs', {}) or {}).get('tool_calls', []) or [])

		# ---- 3) Find the ocr_receipt tool call (normalize both possible call formats) ----
		selected: Optional[Dict[str, Any]] = None
		for tool_call in tool_calls:
			call: Dict[str, Any] = tool_call if isinstance(tool_call, dict) else {}
			if from_kwargs:
				# OpenAI raw kwargs format: {'id': ..., 'function': {'name': ..., 'arguments': '<json str>'}}
				fn: Dict[str, Any] = call.get('function', {}) or {}
				if fn.get('name') == 'ocr_receipt':
					selected = {'id': call.get('id'), 'name': fn.get('name'), 'args': fn.get('arguments', {})}
					break
			else:
				# Standard LangChain format: {'name': ..., 'args': {...}, 'id': ...}
				if call.get('name') == 'ocr_receipt':
					selected = call
					break

		if selected is None:
			return {}  # No ocr_receipt tool call in the last message; no state change

		tool_call_id = selected.get('id')

		# ---- 4) Extract and normalize the args dict ----
		args: Any = selected.get('args', {}) or selected.get('arguments', {})
		if isinstance(args, str):
			args = parse_tool_arguments(args)
		if not isinstance(args, dict):
			args = {}

		# ---- 5) Invoke the tool (tool invocation is allowed ONLY in a tool handler node) ----
		result: str = ocr_receipt.invoke(args)

		# ---- 6) Append the ToolMessage for the calling LLM; no other state keys are touched ----
		tool_message: ToolMessage = ToolMessage(content= result, name= 'ocr_receipt', tool_call_id= tool_call_id)
		return {'messages': [tool_message]}

	except Exception as e:
		print(f'{RED}[NODE] [ERR] ocr_receipt_handler{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		# Graceful error ToolMessage for the same tool_call_id so the LLM loop can continue
		if tool_call_id:
			return {'messages': [ToolMessage(content= json.dumps({'usable': False, 'error': str(e)}), name= 'ocr_receipt', tool_call_id= tool_call_id)]}
		return {}


def fetch_fx_rate_handler(state: AgentSchema) -> Dict[str, Any]:
    """
    Overview: Tool-handler node for the `fetch_fx_rate` tool. Reads the latest AIMessage from state, locates the `fetch_fx_rate` tool call, invokes the tool with its arguments, and appends the resulting ToolMessage (the FX-rate JSON string) back to the message history so the calling LLM can complete the EUR conversion on the next chat turn.
        Caller Node: route_after_chat (routes here whenever the last AIMessage contains a fetch_fx_rate tool call).
        Inside-the-Node Work (Node Responsibilities): (1) Read the last message from state.get('messages', []); it should be an AIMessage with tool_calls. (2) Iterate last.tool_calls and find the call whose 'name' == 'fetch_fx_rate'; return {} (no state change) when none is found. (3) Extract the call's args dict and invoke the tool via fetch_fx_rate.invoke(args). (4) Wrap the tool result (a JSON string with 'currency', 'rate', 'source') in a ToolMessage bound to the same tool_call_id and return {'messages': [tool_message]}. The handler never modifies any other graph state — the chat node consumes the returned JSON for EUR conversion and may note 'rate from cache' in comments.
        Error handling: On any exception, print the standard DEBUG error line and traceback, then return a ToolMessage containing a JSON error string for the same tool_call_id so the LLM loop can continue gracefully rather than crashing on an unanswered tool call.
        Instructions: Invoke the tool once per tool call; do not convert currencies, assign categories, or write to Excel here (those occur in the chat node).
        Args: state (AgentSchema): the graph state whose 'messages' list ends with an AIMessage containing the fetch_fx_rate tool call.
        Returns: Dict[str, Any]: {'messages': [ToolMessage]} — the FX-rate result (or a JSON error string) for the calling LLM; {} when no fetch_fx_rate tool call is present.
    """
    print_function_name() if DEBUG else None

    # Pre-initialize so the except branch can safely reference it for the error ToolMessage
    tool_call_id: Optional[str] = None
    try:
        # 1) Read the last message; nothing to answer when the history is empty
        messages: List[BaseMessage] = state.get('messages', [])
        if not messages:
            return {}

        # 2) Find the fetch_fx_rate tool call by name
        last_message: BaseMessage = messages[-1]
        tool_calls: List[Dict[str, Any]] = getattr(last_message, 'tool_calls', None) or []
        tool_call: Optional[Dict[str, Any]] = next(
            (tc for tc in tool_calls if tc.get('name') == 'fetch_fx_rate'), None
        )
        if tool_call is None:
            return {}
        tool_call_id = tool_call.get('id')

        # 3) Extract the arguments and invoke the tool (the tool itself never touches graph state)
        args: Dict[str, Any] = tool_call.get('args', {}) or {}
        result: str = fetch_fx_rate.invoke(args)

        # 4) Append the ToolMessage for the calling LLM
        tool_message: ToolMessage = ToolMessage(
            content=result,
            name=tool_call.get('name', 'fetch_fx_rate'),
            tool_call_id=tool_call_id,
        )
        return {'messages': [tool_message]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR] fetch_fx_rate_handler{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Return a JSON error ToolMessage for the same tool_call_id so the LLM loop can continue gracefully
        return {'messages': [ToolMessage(
            content=json.dumps({'error': f'fetch_fx_rate failed: {str(e)}'}),
            name='fetch_fx_rate',
            tool_call_id=tool_call_id or 'fetch_fx_rate_unknown',
        )]}


def append_receipt_row_handler(state: AgentSchema) -> Dict[str, Any]:
    """
    Overview: Tool-handler node for the `append_receipt_row` tool. Extracts the append_receipt_row tool call from the last message — supporting both AIMessage instances and dictionary-form messages, and both the normalized 'tool_calls' attribute/key and the OpenAI-style 'additional_kwargs'['tool_calls'] representation — invokes the tool, appends its ToolMessage for the calling LLM, and — on success — records the stored row in state.session_receipts for same-session Q&A.
        Caller Node: chat (via route_after_chat conditional edges).
        Outside-the-Tool Work (Handler Responsibilities): (1) Read the last message and locate the tool call whose name is 'append_receipt_row'; if none is found, return {} (no state change). (2) Invoke append_receipt_row with the tool call's args. (3) Append a ToolMessage(content=result, tool_call_id=tool_call['id']) so the LLM sees the confirmation/error. (4) If the result indicates SUCCESS (does NOT start with 'Error'), build the stored row dict (keys 'cost', 'date', 'items', 'location', 'category', 'comments'; cost coerced to float, others to str) and return it under 'session_receipts' — the Annotated[List[Dict[str, Any]], add] reducer APPENDS it, so only the new row(s) are returned, never the full accumulated list. (5) On failure, return only the ToolMessage and do NOT touch session_receipts.
        Message Representations Supported: (a) AIMessage with .tool_calls as [{'name', 'args', 'id'}]; (b) AIMessage with .additional_kwargs['tool_calls'] in OpenAI raw format (name/arguments nested under 'function', arguments possibly a JSON string parsed via parse_tool_arguments); (c) dictionary-form messages with a top-level 'tool_calls' key or an 'additional_kwargs'['tool_calls'] key — tool-call fields ('name', 'args'/'arguments', 'id', nested 'function') are read safely in every shape.
        State Updates: messages (appended ToolMessage), session_receipts (appended row dict on success only).
        Args: state (AgentSchema): the LangGraph state (MessagesState-style TypedDict; access via state['key'] / state.get('key')).
        Returns: Dict[str, Any]: {'messages': [tool_message], 'session_receipts': [row]} on success, {'messages': [tool_message]} on failure/no tool call, or an error ToolMessage for the same tool_call_id on exception.
    """
    print_function_name() if DEBUG else None

    # Sentinel declared before the try so the except branch can always address the error ToolMessage
    tool_call: Optional[Dict[str, Any]] = None

    try:
        # ---- 1) Read the last message from state (AIMessage instance OR dictionary-form message) ----
        messages: List[Any] = state.get('messages', [])
        if not messages:
            return {}

        last: Any = messages[-1]

        # Extract the raw tool_calls list from whichever representation is present:
        #   - AIMessage.tool_calls (LangChain-normalized [{'name', 'args', 'id'}])
        #   - AIMessage.additional_kwargs['tool_calls'] (OpenAI raw format)
        #   - dict-form message with a 'tool_calls' key or 'additional_kwargs'['tool_calls'] key
        raw_tool_calls: List[Any] = []
        if isinstance(last, dict):
            # Dictionary-form message (e.g. serialized AIMessage at the graph/test boundary)
            if last.get('tool_calls'):
                raw_tool_calls = list(last['tool_calls'])
            else:
                kwargs_obj: Any = last.get('additional_kwargs') or {}
                if isinstance(kwargs_obj, dict):
                    raw_tool_calls = list(kwargs_obj.get('tool_calls') or [])
        else:
            # Message object (AIMessage or any BaseMessage subclass)
            if getattr(last, 'tool_calls', None):
                raw_tool_calls = list(last.tool_calls)
            else:
                kwargs_obj = getattr(last, 'additional_kwargs', None) or {}
                if isinstance(kwargs_obj, dict):
                    raw_tool_calls = list(kwargs_obj.get('tool_calls') or [])

        if not raw_tool_calls:
            return {}

        # ---- 2) Normalize each tool call to {'name', 'args', 'id'} and find the append_receipt_row call ----
        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue

            # OpenAI raw format nests name/arguments under 'function'; the normalized
            # LangChain format carries them at the top level of the tool-call dict
            fn: Any = tc.get('function')
            if isinstance(fn, dict):
                name: str = str(fn.get('name') or '')
                raw_args: Any = fn.get('arguments')
            else:
                name = str(tc.get('name') or '')
                raw_args = tc.get('args')
                if raw_args is None:
                    raw_args = tc.get('arguments')

            if name != 'append_receipt_row':
                continue

            # Args may be a dict (normalized) or a JSON string (OpenAI raw) — parse defensively
            if isinstance(raw_args, str):
                raw_args = parse_tool_arguments(raw_args)
            if not isinstance(raw_args, dict):
                raw_args = {}

            tool_call = {'name': name, 'args': raw_args, 'id': tc.get('id')}
            break

        if tool_call is None:
            return {}

        # ---- 3) Extract args and invoke the tool ----
        args: Dict[str, Any] = tool_call['args']
        result: str = append_receipt_row.invoke(args)

        # ---- 4) Append the ToolMessage for the calling LLM ----
        tool_message: ToolMessage = ToolMessage(
            content= result,
            name= 'append_receipt_row',
            tool_call_id= tool_call['id'],
        )

        # ---- 5) On success, record the stored row in session_receipts (add reducer appends it) ----
        if not result.startswith('Error'):
            try:
                cost: float = float(args.get('cost') or 0.0)
            except (TypeError, ValueError):
                cost = 0.0
            row: Dict[str, Any] = {
                'cost': cost,
                'date': str(args.get('date', '')),
                'items': str(args.get('items', '')),
                'location': str(args.get('location', '')),
                'category': str(args.get('category', '')),
                'comments': str(args.get('comments', '')),
            }
            return {'messages': [tool_message], 'session_receipts': [row]}

        # ---- 6) On failure, return only the ToolMessage (do NOT touch session_receipts) ----
        return {'messages': [tool_message]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR] append_receipt_row_handler{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Always answer the tool call so the next LLM turn is well-formed
        return {'messages': [
            ToolMessage(
                content= f'Error in append_receipt_row_handler: {e}',
                name= 'append_receipt_row',
                tool_call_id= tool_call['id'] if tool_call else 'append_receipt_row',
            )
        ]}


def query_receipts_handler(state: AgentSchema) -> Dict[str, Any]:
    """
    Overview: Tool-handler node for the `query_receipts` tool. Extracts the `query_receipts` tool call from the last AIMessage, invokes the tool with its arguments, and appends the result as a ToolMessage for the calling LLM.
        Caller Node: chat (routed here via route_after_chat when the latest AIMessage contains a `query_receipts` tool call).
        Outside-the-Tool Work (Tool Handler Function Responsibilities): (1) Read the last message from state['messages'] and locate the tool call whose 'name' == 'query_receipts' (checking both the `tool_calls` attribute format — id/name/args flat on the call dict — and the `additional_kwargs['tool_calls']` fallback format, where the id lives on the OUTER dict while name/arguments are nested under 'function'). (2) Invoke query_receipts once with the extracted args dict (string arguments are parsed via parse_tool_arguments). (3) Append a ToolMessage(content=result, tool_call_id=...) to messages so the chat node can aggregate the rows and compose the natural-language answer (combined with state.session_receipts for very recent receipts). (4) On any exception, print the standard DEBUG error line and return a ToolMessage with a JSON error string for the same tool_call_id so the LLM loop can continue gracefully; the tool_call_id is best-effort recovered from the last message if the failure happened before it was captured. If no matching tool call is found, return {} (no state change).
        Inside-the-Tool Work (Tool Responsibilities): None here — the pandas read, filtering, and JSON serialization all live inside the `query_receipts` tool itself.
        Instructions: Never call the tool outside this handler. Do not touch any state keys other than 'messages' — the chat node performs the aggregation/summary and composes the user-visible reply.
        State Updates (on the caller function): messages (appended ToolMessage with the query result or error payload).
        Args: state (AgentSchema): the LangGraph state; reads state['messages'].
        Returns: Dict[str, Any]: {'messages': [ToolMessage]} on success or on handled error; {} when no matching tool call is present.
    """
    print_function_name() if DEBUG else None

    tool_call_id: str = ''

    try:
        # ---- 1) Read the last message and extract its tool calls ----
        messages: List[BaseMessage] = state.get('messages', [])
        last: Optional[BaseMessage] = messages[-1] if messages else None
        if last is None:
            return {}

        # Primary format: the `tool_calls` attribute; fallback: `additional_kwargs['tool_calls']`
        from_kwargs: bool = False
        tool_calls: List[Any] = getattr(last, 'tool_calls', None) or []
        if not tool_calls:
            from_kwargs = True
            tool_calls = (getattr(last, 'additional_kwargs', None) or {}).get('tool_calls', []) or []

        # ---- 2) Locate the tool call for `query_receipts` ----
        tool_call: Optional[Dict[str, Any]] = None
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            if from_kwargs:
                # additional_kwargs/OpenAI-wire format: id on the OUTER dict, name/arguments nested under 'function'
                fn: Any = tc.get('function')
                if isinstance(fn, dict) and fn.get('name') == 'query_receipts':
                    tool_call = fn
                    tool_call_id = str(tc.get('id', '') or '')  # id lives on the OUTER dict in this format
                    break
            elif tc.get('name') == 'query_receipts':
                # standard tool_calls attribute format: id/name/args flat on the call dict
                tool_call = tc
                tool_call_id = str(tc.get('id', '') or '')
                break

        # No matching tool call -> no state change (routing misfire guard)
        if tool_call is None:
            return {}

        # ---- 3) Extract args and invoke the tool (allowed only in tool-handler nodes) ----
        args: Any = tool_call.get('args')
        if args is None:
            args = tool_call.get('arguments')  # additional_kwargs wire format stores JSON-string arguments
        if isinstance(args, str):
            args = parse_tool_arguments(args)
        if not isinstance(args, dict):
            args = {}

        result: str = query_receipts.invoke(args)

        # ---- 4) Append the ToolMessage for the calling LLM ----
        return {'messages': [ToolMessage(
            content= result,
            name= 'query_receipts',
            tool_call_id= tool_call_id,
        )]}

    except Exception as e:
        print(f'{RED}[NODE] [ERR] query_receipts_handler{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Best-effort recovery of the tool_call_id so the error ToolMessage always carries the
        # correct id and the LLM loop can continue gracefully even after an early failure
        if not tool_call_id:
            try:
                recovery_msgs: List[BaseMessage] = state.get('messages', [])
                recovery_last: Optional[BaseMessage] = recovery_msgs[-1] if recovery_msgs else None
                recovery_calls: List[Any] = list(getattr(recovery_last, 'tool_calls', None) or [])
                recovery_calls += (getattr(recovery_last, 'additional_kwargs', None) or {}).get('tool_calls', []) or []
                for tc in recovery_calls:
                    if isinstance(tc, dict) and (
                        tc.get('name') == 'query_receipts'
                        or (isinstance(tc.get('function'), dict) and tc['function'].get('name') == 'query_receipts')
                    ):
                        tool_call_id = str(tc.get('id', '') or '')
                        break
            except Exception:
                pass  # keep ''; the error ToolMessage is still returned so the loop can continue
        # Return a JSON error ToolMessage for the same tool_call_id so the LLM loop continues gracefully
        return {'messages': [ToolMessage(
            content= json.dumps({'error': f'query_receipts failed: {e}'}),
            name= 'query_receipts',
            tool_call_id= tool_call_id,
        )]}


def chat(state: AgentSchema) -> Dict[str, Any]:
    """ Execution: LLM+TOOLS. The single conversational brain. Reads the intent set by start plus state.mode/next_action and pending_question. If state.next_action equals awaiting_clear_photo and the new message contains a photo, retry OCR for that receipt; otherwise start fresh. Tools available: ocr_receipt (calls OpenRouter vision with Qwen2.5-VL or Gemini Flash to extract total, date/time, items with quantity/price/original currency, and location), fetch_fx_rate (query a free FX API first, e.g., frankfurter.app or exchangerate.host; on success write the rate to convertions.csv, on failure read the cached rate from convertions.csv), append_receipt_row (create/append receipts.xlsx rows with columns cost, date, items, location, category, comments), and query_receipts (pandas read of receipts.xlsx). For a photo, verify the path exists and is readable before OCR; if OCR yields usable data, convert every amount to EUR (foreign amount divided by the stored EUR-basis rate), auto-assign a category from the standard set or create one if nothing fits, store date as YYYY-MM-DD HH:MM (00:00 if no time visible; processing date flagged in comments if no date at all), represent items as item1 (quantity x price EUR), item2 (...), write exactly one row per successfully processed receipt (multiple photos -> multiple rows), and reply with a short confirmation. If the message also contains text, incorporate it as comments/context. For text-only messages, do not re-read photos; answer the spending question concisely with supporting numbers from query_receipts plus just-processed receipts in current conversation context, in the user's message language (default English). If OCR produces no usable data, send exactly one polite request for a clearer photo, set next_action to awaiting_clear_photo and pending_question if appropriate, then produce that as the single reply and end the run; never retry internally on the same run. On a duplicate event, produce no reply. 
	Execution: LLM+TOOLS. The single conversational brain of the expense-tracking agent (also absorbs the folded-in responsibilities of the workflow's 'start' routing/dedupe step, since the compiled graph contains only this node between START and END).
	
	Purpose:
	    Receives the inbound payload (photo file path(s), a text message, or both), classifies it into an intent, and either processes receipt(s) into `receipts.xlsx` or answers a spending question — producing exactly one user-visible reply per run (or none for duplicate events).
	
	Processing steps:
	    1. Routing/dedupe (folded 'start' logic): read persisted state (mode, next_action, pending_question, last_seen_event_id, messages) from the checkpointer; classify the payload deterministically via `classify_intent` into process_receipt / answer_question / process_receipt_with_notes; if `is_duplicate_event` flags the event as a duplicate, mark the run as a no-op and return without producing a reply.
	    2. Awaiting-clear-photo resume: if state.next_action == 'awaiting_clear_photo' and the new message contains a photo path, validate it with `validate_photo_path` and proceed to OCR; otherwise start fresh.
	    3. Receipt processing (photo present): for each photo path, call the `ocr_receipt` tool (OpenRouter vision model, Qwen2.5-VL or Gemini Flash) to extract total, date/time, items (quantity, price, original currency), and location. If OCR yields no usable data, emit exactly ONE polite re-request for a clearer photo, set next_action='awaiting_clear_photo' and pending_question, and end the run — never retry internally on the same run.
	    4. EUR conversion: for each extracted receipt, obtain the rate via the `fetch_fx_rate` tool (online FX API first, `convertions.csv` fallback) and convert every amount with `convert_amount_to_eur` (foreign amount divided by the stored EUR-basis rate).
	    5. Row assembly: normalize the date with `format_receipt_datetime` (YYYY-MM-DD HH:MM; 00:00 if no time visible; processing date flagged in comments if no date at all), format items with `format_items_string` ('item1 (quantity x price EUR), item2 (...)'), assign a category (standard set preferred, new category allowed), merge any accompanying user text into comments, and persist via the `append_receipt_row` tool — exactly one row per successfully processed receipt. Record the stored row in state.session_receipts for same-session Q&A. Reply with a short confirmation.
	    6. Question answering (text only): never re-read photos; call the `query_receipts` tool (pandas over receipts.xlsx) and combine the result with state.session_receipts (just-processed receipts in the current conversation) to answer concisely with supporting numbers, in the language of the user's message (default English).
	
	State keys read:
	    messages, intent, mode, next_action, pending_question, last_seen_event_id, photo_paths, user_text, session_receipts.
	
	State keys returned/updated:
	    messages (appended AIMessage reply and any ToolMessages), intent, mode, next_action, pending_question, last_seen_event_id, session_receipts (appended stored receipt rows), reply.
	
	Helper functions required:
	    classify_intent, is_duplicate_event, validate_photo_path, convert_amount_to_eur, format_receipt_datetime, format_items_string, build_receipt_row.
	
	Tools required (bound to chat_llm):
	    ocr_receipt, fetch_fx_rate, append_receipt_row, query_receipts.
	"""

    print_function_name()
    try:
        # ---- 1) Read state (AgentSchema is a MessagesState TypedDict -> .get access) ----
        messages: List[BaseMessage] = state.get('messages', [])
        photo_paths: List[str] = state.get('photo_paths') or []
        user_text: Optional[str] = state.get('user_text')
        mode: str = state.get('mode', 'idle')
        next_action: Optional[str] = state.get('next_action')
        pending_question: Optional[str] = state.get('pending_question')
        session_receipts: List[Dict[str, Any]] = state.get('session_receipts') or []
        event_id: Optional[str] = state.get('event_id')  # may be absent; .get returns None
        last_seen_event_id: Optional[str] = state.get('last_seen_event_id')

        # ---- 2) Duplicate no-op: no reply, no LLM call ----
        if is_duplicate_event(event_id, last_seen_event_id):
            return {'mode': 'idle', 'reply': None, 'next_action': 'none', 'pending_question': None, 'last_seen_event_id': event_id}

        # ---- 3) Classify the inbound payload (never early-return on 'none': the LLM may be resuming
        # mid-run after a ToolMessage, so this is the only early exit before the LLM call) ----
        intent: str = classify_intent(photo_paths, user_text)

        # ---- 4) Compact context block for the system prompt (metadata only; the conversation itself
        # goes in as messages and is NEVER formatted into the prompt) ----
        annotated_photos: List[str] = []
        for photo_path in photo_paths:
            validity: str = 'valid' if validate_photo_path(photo_path) else 'INVALID'
            annotated_photos.append(f'{photo_path} ({validity})')
        session_receipts_json: str = json.dumps(session_receipts[-5:], default= str)
        current_time: str = datetime.now().strftime('%Y-%m-%d %H:%M')
        context: str = (
            '=== RUN CONTEXT ===\n'
            f'INTENT: {intent}\n'
            f'PHOTOS: {"; ".join(annotated_photos) if annotated_photos else "(none)"}\n'
            f'USER_TEXT: {user_text if user_text else "(none)"}\n'
            f'NEXT_ACTION: {next_action if next_action else "(none)"}\n'
            f'PENDING_QUESTION: {pending_question if pending_question else "(none)"}\n'
            f'SESSION_RECEIPTS: {session_receipts_json}\n'
            f'CURRENT_TIME: {current_time}'
        )

        # ---- 5) Format the prompt: pass a superset of the likely placeholder names (str.format silently
        # ignores extra kwargs). str.format can also raise ValueError on unescaped/unbalanced literal
        # braces in the template (e.g. JSON examples) and AttributeError on bad attribute lookups, so
        # EVERY formatting failure falls back to the raw prompt with the context block appended — a
        # placeholder or brace mismatch can never crash the node nor skip the LLM call. ----
        try:
            prompt: str = prompts.CHAT_PROMPT.format(
                context= context,
                intent= intent,
                mode= mode,
                next_action= next_action if next_action else 'none',
                pending_question= pending_question if pending_question else 'none',
                photo_paths= '; '.join(annotated_photos) if annotated_photos else 'none',
                user_text= user_text if user_text else '',
                session_receipts= session_receipts_json,
                current_time= current_time,
            )
            # Guarantee the full context metadata reaches the LLM even if CHAT_PROMPT declares only a
            # subset of the placeholders above (or none at all): the unique header only appears in the
            # prompt once the context block itself has been injected.
            if '=== RUN CONTEXT ===' not in prompt:
                prompt = prompt + '\n\n' + context
        except (TypeError, KeyError, IndexError, ValueError, AttributeError):
            base_prompt: str = prompts.CHAT_PROMPT if isinstance(prompts.CHAT_PROMPT, str) else str(prompts.CHAT_PROMPT)
            prompt = base_prompt + '\n\n' + context

        # ---- 6) Single LLM call: system prompt + full conversation history (messages passed exactly once) ----
        result: BaseMessage = safe_invoke(chat_llm, messages= [SystemMessage(content= prompt)] + list(messages))

        # ---- 7) Tool-call branch: hand off to the matching handler node via the conditional edge.
        # Tools are NEVER invoked inside this LLM node. ----
        result_tool_calls: Any = getattr(result, 'tool_calls', None)
        if not result_tool_calls:
            # Defensive: some providers surface tool calls in additional_kwargs instead
            result_tool_calls = (getattr(result, 'additional_kwargs', {}) or {}).get('tool_calls', [])
        if result_tool_calls:
            # Enforce exactly ONE tool call per turn: the handler nodes service a single tool
            # call per AIMessage, so parallel tool calls would leave unanswered tool_calls in
            # the history (rejected by providers such as Anthropic). Extra calls are dropped;
            # the LLM re-emits them on the next chat turn after the handler's ToolMessage.
            if len(result_tool_calls) > 1:
                first_call: Any = result_tool_calls[0]
                # Normalize a raw OpenAI-wire call (name/arguments nested under 'function') to
                # the LangChain format so the rebuilt single-call AIMessage stays well-formed.
                if isinstance(first_call, dict) and isinstance(first_call.get('function'), dict):
                    raw_fn: Dict[str, Any] = first_call['function']
                    raw_args: Any = raw_fn.get('arguments')
                    if isinstance(raw_args, str):
                        raw_args = parse_tool_arguments(raw_args)
                    first_call = {'name': raw_fn.get('name'), 'args': raw_args if isinstance(raw_args, dict) else {}, 'id': first_call.get('id')}
                result = AIMessage(content= result.content, tool_calls= [first_call])
            return {'messages': [result], 'intent': intent, 'mode': 'processing', 'photo_paths': photo_paths, 'user_text': user_text}

        # ---- 8) Final-reply branch: normalize the content and detect the single polite re-request ----
        content: Any = result.content
        if isinstance(content, list):
            content = ' '.join(
                part.get('text', '') for part in content
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        reply: str = str(content).strip()

        # Last ToolMessage in the conversation (if any), content normalized to plain text
        last_tool_message: Optional[ToolMessage] = next(
            (m for m in reversed(messages) if isinstance(m, ToolMessage)), None
        )
        tool_content: Any = last_tool_message.content if last_tool_message is not None else ''
        if isinstance(tool_content, list):
            tool_content = ' '.join(
                part.get('text', '') for part in tool_content
                if isinstance(part, dict) and part.get('type') == 'text'
            )
        tool_content_str: str = str(tool_content)

        # Single polite re-request: the last ToolMessage reported unusable OCR AND this is the first
        # re-request (next_action != 'awaiting_clear_photo'), so a retry loop can never form.
        first_re_request: bool = (
            last_tool_message is not None
            and next_action != 'awaiting_clear_photo'
            and ('"usable": false' in tool_content_str or '"usable":false' in tool_content_str)
        )

        if first_re_request:
            return {
                'messages': [result],
                'intent': intent,
                'mode': 'idle',
                'next_action': 'awaiting_clear_photo',
                'pending_question': reply,
                'last_seen_event_id': event_id,
                'reply': reply,
            }
        return {
            'messages': [result],
            'intent': intent,
            'mode': 'idle',
            'next_action': 'none',
            'pending_question': None,
            'last_seen_event_id': event_id,
            'reply': reply,
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR] chat{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return {
            'messages': [AIMessage(content= 'Sorry, something went wrong while processing your request. Please try again.')],
            'mode': 'idle',
            'reply': 'Sorry, something went wrong while processing your request. Please try again.',
            'next_action': 'none',
            'pending_question': None,
        }








''' Graph '''
receipt_manager_conversational_agent_graph = StateGraph(AgentSchema)

receipt_manager_conversational_agent_graph.add_node("chat", chat)
receipt_manager_conversational_agent_graph.add_node("ocr_receipt_handler", ocr_receipt_handler)
receipt_manager_conversational_agent_graph.add_node("fetch_fx_rate_handler", fetch_fx_rate_handler)
receipt_manager_conversational_agent_graph.add_node("append_receipt_row_handler", append_receipt_row_handler)
receipt_manager_conversational_agent_graph.add_node("query_receipts_handler", query_receipts_handler)

receipt_manager_conversational_agent_graph.add_edge(START, "chat")
receipt_manager_conversational_agent_graph.add_conditional_edges(
    "chat",
    route_after_chat,
    {
        "ocr_receipt_handler": "ocr_receipt_handler",
        "fetch_fx_rate_handler": "fetch_fx_rate_handler",
        "append_receipt_row_handler": "append_receipt_row_handler",
        "query_receipts_handler": "query_receipts_handler",
        "done": END,
    },
)
receipt_manager_conversational_agent_graph.add_edge("ocr_receipt_handler", "chat")
receipt_manager_conversational_agent_graph.add_edge("fetch_fx_rate_handler", "chat")
receipt_manager_conversational_agent_graph.add_edge("append_receipt_row_handler", "chat")
receipt_manager_conversational_agent_graph.add_edge("query_receipts_handler", "chat")


receipt_manager_conversational_agent_app = receipt_manager_conversational_agent_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    import uuid

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'ablate_design_test',
            'run_name': 'ablate_design_test',
            'thread_id': f'ablate_design_test:{uuid.uuid4()}',
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

        event_id = str(uuid.uuid4())

        if user_in.startswith('re:'):
            receipt_input = user_in[3:].strip()

            if '|' in receipt_input:
                image_path, user_text = receipt_input.split('|', 1)
                image_path = image_path.strip()
                user_text = user_text.strip()

                human_message = (
                    f'Receipt image path: {image_path}\n'
                    f'Accompanying text: {user_text}'
                )

            else:
                image_path = receipt_input.strip()
                user_text = None
                human_message = f'Receipt image path: {image_path}'

            state_input = {
                'messages': [HumanMessage(content=human_message)],
                'photo_paths': [image_path],
                'user_text': user_text,
                'event_id': event_id,
            }

        else:
            state_input = {
                'messages': [HumanMessage(content=user_in)],
                'photo_paths': [],
                'user_text': user_in,
                'event_id': event_id,
            }

        response = receipt_manager_conversational_agent_app.invoke(
            state_input,
            config=config,
        )

        messages = response.get('messages', [])

        # Find the latest HumanMessage from this turn.
        last_human_index = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_index = i
                break

        # Print everything produced after the latest user message.
        for message in messages[last_human_index + 1:]:
            if message.content:
                print(f'\n{BLUE}[ANSWER]{RESET} {message.content}')
            else:
                print(f'\n{BLUE}[ANSWER]{RESET} {message}')

        # Also expose important state for behavioral testing.
        if response.get('reply'):
            print(f'\n{GREEN}[REPLY]{RESET} {response["reply"]}')

        if response.get('next_action') not in (None, 'none'):
            print(f'{GREEN}[NEXT ACTION]{RESET} {response.get("next_action")}')

        user_in = input(f'\n{GREEN}[USER INPUT]{RESET} > ')