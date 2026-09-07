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
import os
import traceback
import json

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS, read_state_file, clean_llm_output
from creations.receipt_spending_assistant import receipt_spending_assistant_prompts as prompts

import base64
import io
import mimetypes
import re
import requests
import pandas as pd
from datetime import date, datetime



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} receipt_spending_assistant') if DEBUG else None



""" Schemas """

class PendingQuestion(TypedDict):
	"""
	TypedDict describing a question or follow-up that the assistant is waiting on. Stored in AgentSchema.pending_question when next_action is 'pending_question'. `question` is the user-facing question that was asked, `context` holds any aggregation or filtering context needed to answer it, and `asked_at` is an optional ISO timestamp for when it was stored.
	"""
	question: str # The question text or follow-up prompt the assistant is waiting for an answer to.
	context: Optional[str] # Supplementary context such as the requested category, time period, or merchant filter needed to complete the query.
	asked_at: Optional[str] # Optional ISO timestamp recording when the question was stored.




class ReceiptLocation(BaseModel):
	"""
	BaseModel representing the location extracted from a receipt. `merchant` is the merchant name used both for display and for dedupe on (date, total, merchant); `address` and `city` are optional and used to enrich the location column.
	"""
	merchant: str # Merchant name, required for dedupe and for the location column.
	address: Optional[str] = None # Street address when present on the receipt.
	city: Optional[str] = None # City when present on the receipt.
	raw_location: Optional[str] = None # Optional raw location string as read by the vision model, useful for debugging without logging OCR text.

	def to_str(self) -> str:
			"""
		Returns a compact location string: 'merchant', 'merchant, address', or 'merchant, address, city', omitting absent optional parts.
			"""
			parts: List[str] = [self.merchant]
			for part in (self.address, self.city):
				if isinstance(part, str) and part.strip():
					parts.append(part.strip())
			return ', '.join(parts)


class LineItem(BaseModel):
	"""
	BaseModel for one line item extracted from a receipt. `description` is the item name, `quantity` is the purchased quantity and defaults to 1 when absent, `unit_price` is the price per unit in the receipt currency, and `total_price` is an optional computed total used by the validation sanity check.
	"""
	description: str # Item description as read from the receipt.
	quantity: Optional[float] = None # Quantity of the item; defaults to 1 when unknown or absent.
	unit_price: float # Unit price in the receipt currency.
	total_price: Optional[float] = None # Optional line total (quantity * unit_price), used to check the sum of items against the grand total.

	def line_total(self) -> float:
			"""
		Returns quantity * unit_price, treating a missing quantity as 1.
			"""
			return (self.quantity if self.quantity is not None else 1.0) * self.unit_price

	def format(self, currency: str) -> str:
			"""
		Returns the required display format 'description (qty x price currency)'.
			"""
			qty: float = self.quantity if self.quantity is not None else 1
			return f'{self.description} ({qty} x {self.unit_price} {currency})'


class ReceiptData(BaseModel):
	"""
	Pydantic-style BaseModel for the validated receipt extracted by the vision LLM. `cost` is the grand total including tax, `date` is the purchase date normalized to ISO YYYY-MM-DD, `items` is the list of line items, `currency` is the receipt currency, `location` is the merchant/location object, `confidence` is the optional parser confidence used to decide whether to retry when below 0.7, and `category` is the auto-assigned category filled in by append_receipt.
	"""
	cost: float # Grand total including tax, as read from the receipt.
	date: str # Purchase date normalized to ISO YYYY-MM-DD; falls back to processing date if unreadable.
	items: List[LineItem] # Line items parsed from the receipt, each with description, quantity, and unit price.
	currency: str # Currency read from the receipt, stored per row and used when formatting items.
	location: ReceiptLocation # Merchant name plus optional address/city.
	confidence: Optional[float] = None # Optional extraction confidence; values below 0.7 trigger one bounded retry in validate_receipt.
	category: Optional[str] = None # Auto-assigned category from the configurable taxonomy, defaulting to 'Other' when unrecognizable.

	def items_formatted(self) -> str:
			"""
		Returns all line items formatted as 'description (qty x price currency), ...' using self.currency.
			"""
			return ', '.join(item.format(self.currency) for item in self.items)

	def location_str(self) -> str:
			"""
	Delegates to self.location.to_str() to produce the location column value.
			"""
			return self.location.to_str()

	def to_row(self, category: str) -> tuple[float, str, str, str, str]:
			"""
			Returns a row tuple (cost, date, items_formatted, location_str, category) matching the Excel header columns [cost, date, items, location, category].
			"""
			return (
				self.cost,
				self.date,
				self.items_formatted(),
				self.location_str(),
				category,
			)


class AgentSchema(MessagesState):
	"""
	State schema for the receipt_spending_assistant graph. It extends MessagesState so `messages` is an append-reduced conversation history persisted by the checkpointer across runs. The start node appends the new inbound HumanMessage, the chat node routes the run based on mode/next_action and produces one outbound reply stored in `latest`, and the end node persists the final AI message. `pending_question` stores an open question or follow-up context when the assistant needs more input; `dedupe_state` stores receipt signatures already seen in the conversation; `processed_event_ids` guards against retried inbound events; `receipt_store_path` is the configurable .xlsx location; `categories` is the configurable category taxonomy; `last_error_context` holds minimal safe error context for a resend flow.
	"""
	messages: Annotated[list[BaseMessage], add] # Conversation history. Inherited from MessagesState; the add reducer appends new messages on each run. messages[-1] is always the new inbound HumanMessage for the current run.
	mode: Optional[str] # Persistent flow mode from previous runs. Used together with next_action to decide routing in chat, for example 'receipt' or 'question' mode.
	next_action: Optional[Literal['idle', 'await_resend', 'pending_question']] # Resume marker for the next inbound event. 'idle' means no pending follow-up, 'await_resend' means the user is resending a failed receipt, and 'pending_question' means the user is answering or following up on a stored question.
	pending_question: Optional[Union[PendingQuestion, str]] # Stored question/context awaiting user input. Either a PendingQuestion dict with details or a plain string when only the question text is needed. Cleared once the answer/follow-up is processed.
	latest: Optional[str] # The single final user-visible reply produced by the current run. The end node persists it as the outbound AI message.
	dedupe_state: Optional[set] # Set of receipt signature tuples (date, total, merchant) already seen in this conversation. append_receipt also checks the Excel store, but this set provides in-conversation dedupe context.
	processed_event_ids: Optional[set] # Set of inbound event IDs already processed, used to avoid duplicate side effects when webhook events are retried.
	receipt_store_path: Optional[str] # Path to the local .xlsx receipt store. Defaults to ./spending.xlsx when absent.
	categories: Optional[List[str]] # Configurable category taxonomy used by append_receipt. Defaults to [Groceries, Dining, Transport, Utilities, Housing, Health, Entertainment, Shopping, Travel, Other].
	last_error_context: Optional[str] # Minimal, safe error context carried into an await_resend flow so the retry can use corrective hints. Never stores raw image bytes or raw OCR text.




''' Tools '''
@tool
def parse_receipt_image(image_source: Union[str, bytes]) -> dict:
	"""
	Overview: Converts a receipt image (local file path or raw bytes) into a base64 data URL and sends it to the OpenRouter OpenAI-compatible /api/v1/chat/completions endpoint using the vision model from env var RECEIPT_MODEL (default google/gemini-2.5-flash). It requests structured JSON containing cost, date, items, currency, location, and confidence. response_format JSON schema is used when supported; otherwise a strict return-JSON-only prompt plus a permissive parser that extracts the first balanced JSON object or code fence is used. This is a non-terminal tool: it only provides raw extracted data to the caller LLM for validation.
	Caller LLM: chat_llm in the chat node receipt flow after the inbound message is classified as a receipt payload.
	Outside-the-Tool Work (Tool Handler Function Responsibilities): None. The ToolNode appends the resulting ToolMessage; the caller LLM continues the same run and should call validate_receipt next. No state updates happen from this tool call; final state updates are deferred until the receipt flow ends.
	Inside-the-Tool Work (Tool Responsibilities): Read the image source, base64-encode it as a data URL, call OpenRouter with the configured vision model, request strict JSON, fall back to the permissive parser when needed, and return the extracted structure and confidence. Never log raw image bytes or raw OCR text beyond minimal error details.
	Instructions:
	1. Accept image_source as either an existing local file path, a file:// or data:image/ URI (str), or raw image bytes. Local file paths must reside in the working directory or the directory configured by env var RECEIPT_IMAGE_DIR.
	2. Determine the image MIME type from the file extension or magic bytes and build the data URL.
	3. Build the OpenAI-compatible request to OPENROUTER_BASE_URL or https://openrouter.ai/api/v1 with path /chat/completions using the model from RECEIPT_MODEL or default google/gemini-2.5-flash.
	4. Put the image in the content as an image_url block whose url is the data URL.
	5. Instruct the model to return JSON only: cost (grand total including tax), date (ISO YYYY-MM-DD, processing date fallback), items (description, quantity, unit_price), currency, location (merchant, address, city), confidence.
	6. If the endpoint supports response_format JSON schema, pass it; otherwise rely on the strict prompt.
	7. Parse the model response. If it is not valid JSON, run the permissive parser to extract the first balanced JSON object or code fence.
	8. Return a compact JSON-serializable dict; never include raw OCR text or the base64 payload.
	State Updates (on the caller function): None.
	Args:
	- image_source (Union[str, bytes]): Local path, file:// or data:image/ URI, or raw bytes of the receipt image.
	Returns:
	- dict: Keys: success (bool), receipt_json (dict or None), confidence (float or None), error (str, present only on failure).
	"""
	def failure(message: str) -> dict:
		return {
			'success': False,
			'receipt_json': None,
			'confidence': None,
			'error': message,
		}

	def parse_permissive(content: Any) -> Optional[dict]:
		"""Parse LLM content: direct JSON, code fence, then first balanced object."""
		if not isinstance(content, str):
			return None
		text: str = content.strip()

		try:
			parsed: Any = json.loads(text)
			if isinstance(parsed, dict):
				return parsed
		except Exception:
			pass

		fence_match: Optional[re.Match] = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
		if fence_match:
			try:
				parsed = json.loads(fence_match.group(1).strip())
				if isinstance(parsed, dict):
					return parsed
			except Exception:
				pass

		start: int = text.find('{')
		while start != -1:
			depth: int = 0
			in_string: bool = False
			escaped: bool = False
			for idx in range(start, len(text)):
				char: str = text[idx]
				if in_string:
					if escaped:
						escaped = False
					elif char == '\\':
						escaped = True
					elif char == '"':
						in_string = False
				elif char == '"':
					in_string = True
				elif char == '{':
					depth += 1
				elif char == '}':
					depth -= 1
					if depth == 0:
						try:
							parsed = json.loads(text[start:idx + 1])
							if isinstance(parsed, dict):
								return parsed
						except Exception:
							pass
						break
			start = text.find('{', start + 1)
		return None

	def normalize_confidence(value: Any) -> Optional[float]:
		"""Return confidence as a float in [0, 1] or None when invalid."""
		if value is None:
			return None
		try:
			confidence: float = float(value)
		except (TypeError, ValueError):
			return None
		if confidence != confidence or not (0.0 <= confidence <= 1.0):
			return None
		return confidence

	# API key guard ----------------------------------------------------------
	api_key: str = os.getenv('OPENROUTER_API_KEY', '')
	if not api_key:
		return failure('missing API key')

	# 1) Normalize the image source into a base64 data URL --------------------
	try:
		if isinstance(image_source, str):
			source: str = image_source.strip()
			if source.lower().startswith('data:image'):
				data_url: str = source
			else:
				local_path: str = source[7:] if source.lower().startswith('file://') else source
				# Path safety: only permit local paths inside an approved root.
				approved_roots: List[str] = [os.path.abspath('.')]
				configured_dir: str = os.getenv('RECEIPT_IMAGE_DIR', '').strip()
				if configured_dir:
					approved_roots.append(os.path.abspath(configured_dir))
				resolved_path: str = os.path.abspath(local_path)
				if not any(resolved_path == root or resolved_path.startswith(root + os.sep) for root in approved_roots):
					return failure('image path not permitted')
				with open(local_path, 'rb') as image_file:
					image_bytes: bytes = image_file.read()
				mime_type: str = mimetypes.guess_type(local_path)[0] or 'image/jpeg'
				data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
		elif isinstance(image_source, bytes):
			image_bytes = image_source
			if image_bytes.startswith(b'\xff\xd8\xff'):
				mime_type = 'image/jpeg'
			elif image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
				mime_type = 'image/png'
			elif image_bytes.startswith(b'GIF8'):
				mime_type = 'image/gif'
			elif len(image_bytes) >= 12 and image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
				mime_type = 'image/webp'
			else:
				mime_type = 'image/jpeg'
			data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
		else:
			return failure('invalid image source')
	except FileNotFoundError:
		return failure('image file not found')
	except PermissionError:
		return failure('unable to read image file')
	except OSError:
		return failure('unable to read image file')
	except Exception:
		return failure('unable to read image source')

	# 2) Build the strict JSON-only extraction prompt -------------------------
	extraction_text: str = (
		'Inspect the receipt image and return JSON only. '
		'Do not include markdown code fences, explanations, or any text outside the JSON object. '
		'The JSON must contain exactly these top-level keys: "cost", "date", "items", '
		'"currency", "location", "confidence". '
		'"cost" must be the grand total including tax and must be a number. '
		'"date" must be the purchase date in ISO format YYYY-MM-DD; if the purchase date '
		"is unreadable, use the processing date (today's date). "
		'"items" must be an array of objects, each containing "description" (string), '
		'"quantity" (number), and "unit_price" (number). '
		'"currency" must be a currency code such as "USD". '
		'"location" must be an object with a required "merchant" string and optional '
		'"address" and "city" strings. '
		'"confidence" must be a number from 0 to 1.'
	)

	# 3) Call the OpenRouter OpenAI-compatible chat completions endpoint ------
	base_url: str = os.getenv('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1'
	url: str = base_url.rstrip('/') + '/chat/completions'
	model: str = os.getenv('RECEIPT_MODEL') or 'google/gemini-2.5-flash'
	headers: dict = {
		'Authorization': f'Bearer {api_key}',
		'Content-Type': 'application/json',
	}

	base_payload: dict = {
		'model': model,
		'temperature': 0.0,
		'messages': [
			{
				'role': 'user',
				'content': [
					{'type': 'text', 'text': extraction_text},
					{'type': 'image_url', 'image_url': {'url': data_url}},
				],
			}
		],
	}

	response: Optional[requests.Response] = None

	for use_response_format in (True, False):
		payload: dict = dict(base_payload)
		if use_response_format:
			payload['response_format'] = {'type': 'json_object'}

		fallback_to_plain: bool = False

		for attempt in range(3):
			try:
				response = requests.post(url, headers=headers, json=payload, timeout=120)

				if response.status_code >= 500:
					if attempt < 2:
						sleep(2)
						continue
					break

				if response.status_code >= 400:
					error_text: str = (response.text or '').lower()
					unsupported_format: bool = (
						'response_format' in error_text
						or 'json_object' in error_text
						or 'structured output' in error_text
						or 'structured_output' in error_text
						or 'unsupported' in error_text
					)
					if use_response_format and unsupported_format:
						fallback_to_plain = True
						break
					return failure(f'request failed (HTTP {response.status_code})')

				break

			except requests.RequestException:
				if attempt < 2:
					sleep(2)
				else:
					break

		if fallback_to_plain:
			response = None
			continue

		break

	if response is None or not response.ok:
		return failure('unable to obtain receipt response')

	# 4) Parse the model response (strict, then permissive) -------------------
	try:
		response_body: dict = response.json()
		content: Any = response_body['choices'][0]['message']['content']
	except (KeyError, IndexError, TypeError, ValueError):
		return failure('invalid receipt response')

	parsed: Optional[dict] = parse_permissive(content)
	if not isinstance(parsed, dict):
		return failure('receipt JSON could not be parsed')

	# 5) Normalize confidence and return --------------------------------------
	return {
		'success': True,
		'receipt_json': parsed,
		'confidence': normalize_confidence(parsed.get('confidence')),
	}

@tool
def validate_receipt(receipt_json: dict) -> dict:
	"""
	Overview: Validates the raw receipt JSON produced by parse_receipt_image with Pydantic-style models and totals sanity checks. It builds LineItem, ReceiptLocation, and ReceiptData objects, normalizes the date to ISO YYYY-MM-DD, defaults missing quantity to 1, and checks that the sum of line-item totals is within approximately 2 percent of the grand total. If the structure is invalid or confidence is below 0.7, it performs one bounded retry, for example a JSON-repair or re-parse pass using the same OpenRouter LLM. This tool is non-terminal: it returns normalized receipt data to the caller LLM and never writes to disk.
	Caller LLM: chat_llm in the chat node receipt flow after parse_receipt_image returns a receipt_json.
	Outside-the-Tool Work (Tool Handler Function Responsibilities): None. The ToolNode appends the ToolMessage; the caller LLM continues in the same run and either calls append_receipt on success or handles the failure by producing a resend prompt and setting next_action to await_resend at the end of the run.
	Inside-the-Tool Work (Tool Responsibilities): Apply Pydantic-style validation, normalize fields, run the sum-vs-total sanity check, perform one bounded retry on failure or low confidence, and return either a validated receipt or a clear error. It must never append or modify the Excel store.
	Instructions:
	1. Confirm receipt_json is a dict; if not, return a failure dict with a compact user-safe error.
	2. Build LineItem objects; if quantity is missing or invalid default to 1. Build ReceiptLocation from the location dict and ReceiptData from the top-level fields.
	3. Normalize date to ISO YYYY-MM-DD; if unreadable use the processing date. Convert cost to a positive float.
	4. Compute sum of line_total for all items and compare to cost. Fail the sanity check if the relative difference is greater than 0.02.
	5. If validation failed or confidence is below 0.7, run one bounded retry, for example a strict repair prompt to the same OpenRouter LLM that returns corrected JSON only, then repeat steps 2-4.
	6. If the retry also fails, do NOT record the receipt; return success False with an error message asking the user to resend.
	7. On success, return a normalized validated receipt dict with cost, date, items, currency, location, confidence, and category (None for now; append_receipt assigns it).
	State Updates (on the caller function): None.
	Args:
	- receipt_json (dict): Raw parsed receipt JSON from parse_receipt_image, expected to contain cost, date, items, currency, location, and optionally confidence.
	Returns:
	- dict: Keys: success (bool), validated_receipt (dict or None), error (str, present only on failure).
	"""

	def normalize_date(value: Any) -> str:
		"""Normalize a receipt date to ISO YYYY-MM-DD, falling back to the processing date."""
		if isinstance(value, datetime):
			return value.strftime('%Y-%m-%d')
		if isinstance(value, date):
			return value.isoformat()
		if isinstance(value, str) and value.strip():
			candidate: str = value.strip()
			try:
				return datetime.strptime(candidate, '%Y-%m-%d').strftime('%Y-%m-%d')
			except ValueError:
				pass
			for fmt in ('%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%b %d, %Y', '%d %b %Y', '%Y-%m-%dT%H:%M:%S'):
				try:
					return datetime.strptime(candidate, fmt).strftime('%Y-%m-%d')
				except ValueError:
					continue
		return date.today().isoformat()

	def is_finite(value: float) -> bool:
		"""Return True when value is a finite number (rejects NaN and infinities)."""
		return value == value and abs(value) != float('inf')

	def build_and_validate(raw: dict):
		"""Build the Pydantic models and run all sanity checks. Returns (ReceiptData, None) or (None, compact_error)."""
		try:
			if not isinstance(raw, dict):
				return None, 'The receipt data could not be read.'

			# Cost: must be a positive finite float; NaN/inf must not pass. ------
			try:
				cost: float = float(raw.get('cost'))
			except (TypeError, ValueError):
				cost = 0.0
			if not (cost > 0 and is_finite(cost)):
				return None, 'The receipt total could not be read.'

			# Date -----------------------------------------------------------
			date_str: str = normalize_date(raw.get('date'))

			# Line items ------------------------------------------------------
			raw_items: Any = raw.get('items')
			if not isinstance(raw_items, list) or not raw_items:
				return None, 'The receipt has no readable line items.'

			items: List[LineItem] = []
			for raw_item in raw_items:
				if not isinstance(raw_item, dict):
					return None, 'A line item could not be read.'
				try:
					description: str = str(raw_item.get('description') or '')
					try:
						unit_price: float = float(raw_item.get('unit_price', 0.0))
					except (TypeError, ValueError):
						unit_price = 0.0
					if not is_finite(unit_price):
						unit_price = 0.0
					try:
						quantity: float = 1.0
						raw_quantity: Any = raw_item.get('quantity')
						if raw_quantity is not None:
							quantity = float(raw_quantity)
						if not (is_finite(quantity) and quantity > 0):
							quantity = 1.0
					except (TypeError, ValueError):
						quantity = 1.0
					items.append(LineItem(description=description, quantity=quantity, unit_price=unit_price))
				except Exception:
					return None, 'A line item could not be read.'

			# Location ---------------------------------------------------------
			raw_location: Any = raw.get('location')
			if isinstance(raw_location, dict):
				address: Any = raw_location.get('address')
				city: Any = raw_location.get('city')
				location: ReceiptLocation = ReceiptLocation(
					merchant=str(raw_location.get('merchant') or ''),
					address=str(address) if address is not None else None,
					city=str(city) if city is not None else None,
					raw_location=raw_location.get('raw_location'),
				)
			else:
				location = ReceiptLocation(merchant='', address=None, city=None, raw_location=None)

			# Confidence ---------------------------------------------------------
			confidence: Optional[float] = raw.get('confidence')
			if confidence is not None:
				try:
					confidence = float(confidence)
				except (TypeError, ValueError):
					confidence = None
				if confidence is not None and not is_finite(confidence):
					confidence = None

			currency: str = str(raw.get('currency') or '')

			receipt_data: ReceiptData = ReceiptData(
				cost=cost,
				date=date_str,
				items=items,
				currency=currency,
				location=location,
				confidence=confidence,
				category=None,
			)

			# Sum-of-items vs total sanity check --------------------------------
			sum_total: float = sum(item.line_total() for item in items)
			if abs(sum_total - cost) / cost > 0.02:
				return None, 'Line items do not reconcile with the receipt total.'

			# Low-confidence check ------------------------------------------------
			if confidence is not None and confidence < 0.7:
				return None, 'Extraction confidence is too low.'

			return receipt_data, None
		except Exception:
			return None, 'The receipt could not be parsed.'

	def to_result(receipt_data: ReceiptData) -> dict:
		"""Serialize a validated ReceiptData into the public result dict."""
		return {
			'cost': receipt_data.cost,
			'date': receipt_data.date,
			'items': [
				{
					'description': item.description,
					'quantity': item.quantity,
					'unit_price': item.unit_price,
				}
				for item in receipt_data.items
			],
			'currency': receipt_data.currency,
			'location': {
				'merchant': receipt_data.location.merchant,
				'address': receipt_data.location.address,
				'city': receipt_data.location.city,
			},
			'confidence': receipt_data.confidence,
			'category': None,
		}

	def parse_permissive(content: Any) -> Optional[dict]:
		"""Parse an LLM reply permissively: direct JSON, code fence, then first balanced object."""
		if not isinstance(content, str):
			return None
		text: str = content.strip()

		try:
			parsed: Any = json.loads(text)
			if isinstance(parsed, dict):
				return parsed
		except (TypeError, ValueError):
			pass

		fence_match: Optional[re.Match] = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
		if fence_match:
			try:
				parsed = json.loads(fence_match.group(1).strip())
				if isinstance(parsed, dict):
					return parsed
			except (TypeError, ValueError):
				pass

		start: int = text.find('{')
		if start != -1:
			depth: int = 0
			in_string: bool = False
			escaped: bool = False
			for idx in range(start, len(text)):
				char: str = text[idx]
				if in_string:
					if escaped:
						escaped = False
					elif char == '\\':
						escaped = True
					elif char == '"':
						in_string = False
				elif char == '"':
					in_string = True
				elif char == '{':
					depth += 1
				elif char == '}':
					depth -= 1
					if depth == 0:
						try:
							parsed = json.loads(text[start:idx + 1])
							if isinstance(parsed, dict):
								return parsed
						except (TypeError, ValueError):
							return None
		return None

	def request_repair(failed_json: dict) -> Optional[dict]:
		"""Send exactly one text-only repair request; return corrected JSON or None on failure."""
		model: str = os.getenv('RECEIPT_MODEL') or 'google/gemini-2.5-flash'
		base_url: str = os.getenv('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1'
		api_key: str = os.getenv('OPENROUTER_API_KEY', '')
		if not api_key:
			return None

		prompt: str = (
			'You are a receipt extraction repair assistant. The JSON below was extracted from a '
			'receipt image but failed validation. Return ONLY corrected JSON matching this schema '
			'exactly, with no markdown, no prose, and no explanations:\n'
			'{\n'
			'  "cost": <grand total including tax, finite positive number>,\n'
			'  "date": "<ISO YYYY-MM-DD>",\n'
			'  "items": [{"description": "<string>", "quantity": <number, use 1 if unknown>, '
			'"unit_price": <number>}],\n'
			'  "currency": "<currency code>",\n'
			'  "location": {"merchant": "<string>", "address": "<string>", "city": "<string>"},\n'
			'  "confidence": <number between 0 and 1>\n'
			'}\n'
			'Failed JSON:\n'
			+ json.dumps(failed_json, ensure_ascii=False)
		)

		try:
			sleep(2)
			response = requests.post(
				f"{base_url.rstrip('/')}/chat/completions",
				headers={
					'Authorization': f'Bearer {api_key}',
					'Content-Type': 'application/json',
				},
				json={
					'model': model,
					'temperature': 0.0,
					'messages': [{'role': 'user', 'content': prompt}],
				},
				timeout=120,
			)
			response.raise_for_status()
			content: Any = response.json()['choices'][0]['message']['content']
			return parse_permissive(content)
		except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
			return None

	if not isinstance(receipt_json, dict):
		return {
			'success': False,
			'validated_receipt': None,
			'error': 'The receipt data could not be read. Please resend the receipt.',
		}

	# First validation attempt ------------------------------------------------
	receipt_data, error = build_and_validate(receipt_json)
	if error is None:
		return {'success': True, 'validated_receipt': to_result(receipt_data)}

	# Exactly one bounded retry ------------------------------------------------
	repaired_json: Optional[dict] = request_repair(receipt_json)
	if repaired_json is not None:
		receipt_data, error = build_and_validate(repaired_json)
		if error is None:
			return {'success': True, 'validated_receipt': to_result(receipt_data)}

	return {
		'success': False,
		'validated_receipt': None,
		'error': 'The receipt could not be validated. Please resend a clearer photo of the receipt.',
	}

@tool
def append_receipt(validated_receipt: dict, store_path: Optional[str] = None, categories: Optional[List[str]] = None) -> dict:
	"""
	Overview: Appends one row per validated receipt to the local .xlsx store (default ./spending.xlsx; optional store_path can override). Columns are exactly cost, date, items, location, category. The file is created with a header on first use. Rows are de-duplicated on (date, total, merchant); duplicates skip the write and return an already-recorded message. Dates are written as ISO YYYY-MM-DD, items are formatted as description (qty x price currency), and one category is auto-assigned from the optional categories taxonomy (default [Groceries, Dining, Transport, Utilities, Housing, Health, Entertainment, Shopping, Travel, Other]), defaulting to Other. The store path must reside in the working directory or the directory configured by env var RECEIPT_STORE_DIR. This is a terminal tool: after it returns, the caller LLM must end the run and send the returned message without further tool calls.
	Caller LLM: chat_llm in the chat node receipt flow after validate_receipt succeeds.
	Outside-the-Tool Work (Tool Handler Function Responsibilities): After the ToolNode returns, the caller LLM must set state latest to the returned message, append the final AIMessage to state messages, set next_action to idle, add the returned signature to state dedupe_state, and end the run. If the tool failed, do not set latest to success; use the returned error message and set next_action to await_resend.
	Inside-the-Tool Work (Tool Responsibilities): Load or create the workbook, normalize fields, check the dedupe signature, auto-assign category, append the row, save the workbook, and return a compact final result or message.
	Instructions:
	1. Read validated_receipt; use store_path if provided else ./spending.xlsx, and categories if provided else the default ten-category taxonomy.
	2. Normalize date to ISO YYYY-MM-DD, cost to float, merchant from location.merchant (lowercased and stripped).
	3. Build the row (cost, date, items_formatted, location_str, category).
	4. Compute the dedupe signature (date, round(cost, 2), merchant). If the signature already exists in the store, do not write; return already recorded.
	5. Load the workbook or create an empty one with header cost, date, items, location, category; add missing columns if needed.
	6. Auto-assign category by matching item descriptions and merchant against keywords in each category; default to Other.
	7. Append the row and save the workbook.
	8. Return success True with appended True and a user-facing success message, or appended False with an already-recorded message; include the signature so the caller can update in-conversation dedupe state.
	State Updates (on the caller function): state latest set to returned message; state messages append AIMessage; state next_action set to idle; state dedupe_state updated with returned signature; state mode set to receipt.
	Args:
	- validated_receipt (dict): Normalized receipt from validate_receipt.
	- store_path (Optional[str]): Path to the .xlsx store; defaults to ./spending.xlsx.
	- categories (Optional[List[str]]): Category taxonomy; defaults to [Groceries, Dining, Transport, Utilities, Housing, Health, Entertainment, Shopping, Travel, Other].
	Returns:
	- dict: Keys: success (bool), appended (bool), message (str), signature (list).
	"""
	DEFAULT_CATEGORIES: List[str] = [
		'Groceries', 'Dining', 'Transport', 'Utilities', 'Housing',
		'Health', 'Entertainment', 'Shopping', 'Travel', 'Other'
	]

	# Defensive defaults; refined as normalization succeeds so the error branch
	# still returns a valid, safe signature.
	normalized_date: str = date.today().isoformat()
	total: float = 0.0
	merchant: str = ''

	try:
		# 1. Resolve store path and category taxonomy.
		path: str = store_path if isinstance(store_path, str) and store_path.strip() else './spending.xlsx'

		# Enforce filesystem-path safety: the store must reside in the working
		# directory or the directory configured by env var RECEIPT_STORE_DIR.
		approved_roots: List[str] = [os.path.abspath('.')]
		configured_root: str = os.getenv('RECEIPT_STORE_DIR', '').strip()
		if configured_root:
			approved_roots.append(os.path.abspath(configured_root))

		resolved_path: str = os.path.abspath(path)
		if not any(
			resolved_path == root or resolved_path.startswith(root + os.sep)
			for root in approved_roots
		):
			return {
				"success": False,
				"appended": False,
				"message": "The receipt store path is not permitted.",
				"signature": [normalized_date, round(total, 2), merchant],
			}

		if isinstance(categories, list) and categories and any(isinstance(c, str) and c.strip() for c in categories):
			category_list: List[str] = [c for c in categories if isinstance(c, str) and c.strip()]
		else:
			category_list: List[str] = DEFAULT_CATEGORIES

		# 2. Normalize date to ISO YYYY-MM-DD, falling back to processing date.
		receipt: dict = validated_receipt if isinstance(validated_receipt, dict) else {}
		raw_date = receipt.get('date')
		try:
			if isinstance(raw_date, datetime):
				normalized_date = raw_date.date().isoformat()
			elif isinstance(raw_date, date):
				normalized_date = raw_date.isoformat()
			elif raw_date is not None:
				date_text: str = str(raw_date).strip()
				parsed_datetime: Optional[datetime] = None
				for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S'):
					try:
						parsed_datetime = datetime.strptime(date_text, fmt)
						break
					except ValueError:
						continue
				if parsed_datetime is None:
					try:
						parsed_datetime = datetime.fromisoformat(date_text.replace('Z', '+00:00'))
					except ValueError:
						parsed_datetime = None
				normalized_date = parsed_datetime.date().isoformat() if parsed_datetime is not None else date.today().isoformat()
		except Exception:
			normalized_date = date.today().isoformat()

		# Normalize cost to float.
		try:
			total = float(receipt.get('cost', 0.0))
		except (TypeError, ValueError):
			total = 0.0

		# Normalize merchant plus optional address/city.
		location = receipt.get('location')
		if isinstance(location, dict):
			raw_merchant = location.get('merchant', '')
			raw_address = location.get('address')
			raw_city = location.get('city')
		else:
			raw_merchant = getattr(location, 'merchant', '')
			raw_address = getattr(location, 'address', None)
			raw_city = getattr(location, 'city', None)

		merchant = str(raw_merchant or '').strip().lower()

		# 3. Build location_str (merchant plus optional address/city).
		location_parts: List[str] = [merchant]
		for part in (raw_address, raw_city):
			if part is not None and not pd.isna(part):
				part_text: str = str(part).strip()
				if part_text:
					location_parts.append(part_text)
		location_str: str = ', '.join(location_parts)

		# Build items_formatted using the receipt currency (quantity defaults to 1).
		currency: str = str(receipt.get('currency', '') or '')
		items: List[Any] = receipt.get('items') or []
		formatted_items: List[str] = []
		item_texts: List[str] = []

		for item in items:
			if isinstance(item, dict):
				description = item.get('description', '')
				quantity = item.get('quantity')
				unit_price = item.get('unit_price', 0.0)
			else:
				description = getattr(item, 'description', '')
				quantity = getattr(item, 'quantity', None)
				unit_price = getattr(item, 'unit_price', 0.0)

			if quantity is None:
				quantity = 1

			desc_text: str = str(description or '')
			formatted_items.append(f"{desc_text} ({quantity} x {unit_price} {currency})")
			item_texts.append(desc_text.lower())

		items_formatted: str = ', '.join(formatted_items)

		# 4. Build the canonical dedupe signature.
		total_rounded: float = round(total, 2)
		signature: tuple[str, float, str] = (normalized_date, total_rounded, merchant)

		# 5. Load or create the workbook with exactly the expected header.
		columns: List[str] = ['cost', 'date', 'items', 'location', 'category']
		df: pd.DataFrame = load_or_create_spreadsheet(path, columns)

		# 6. Check existing rows for duplicates, guarding against NaN/missing values.
		# The stored location cell contains 'merchant, address, city'; extract the
		# merchant (first segment) so existing signatures compare merchant-only
		# values consistently with the incoming signature.
		existing_signatures: set = set()
		for _, row in df.iterrows():
			try:
				row_date = row.get('date')
				row_cost = row.get('cost')
				row_location = row.get('location')
				if row_date is None or row_cost is None or row_location is None:
					continue
				if pd.isna(row_date) or pd.isna(row_cost) or pd.isna(row_location):
					continue
				existing_signatures.add((
					str(row_date)[:10],
					round(float(row_cost), 2),
					str(row_location).split(',')[0].strip().lower(),
				))
			except (TypeError, ValueError, KeyError, AttributeError):
				continue

		if signature in existing_signatures:
			return {
				"success": True,
				"appended": False,
				"message": "This receipt has already been recorded.",
				"signature": [normalized_date, total_rounded, merchant],
			}

		# 7. Auto-assign a category by case-insensitive keyword matching.
		keyword_map: Dict[str, List[str]] = {
			'groceries': ['grocery', 'supermarket', 'walmart', 'tesco', 'market', 'produce', 'food'],
			'dining': ['restaurant', 'cafe', 'coffee', 'starbucks', 'mcdonald', 'subway', 'pizza', 'bar'],
			'transport': ['uber', 'lyft', 'taxi', 'gas', 'fuel', 'train', 'bus', 'parking'],
			'utilities': ['electric', 'water', 'internet', 'phone', 'utility', 'power'],
			'housing': ['rent', 'mortgage', 'hotel', 'airbnb', 'realty', 'lease'],
			'health': ['pharmacy', 'doctor', 'hospital', 'clinic', 'dental', 'medical'],
			'entertainment': ['cinema', 'netflix', 'movie', 'game', 'spotify', 'concert'],
			'shopping': ['amazon', 'target', 'mall', 'clothing', 'shop', 'best buy'],
			'travel': ['flight', 'airline', 'airbnb', 'hotel', 'booking', 'travel'],
		}

		searchable_text: str = ' '.join([merchant] + item_texts)
		selected_category: str = 'Other'
		highest_score: int = 0

		for configured_category in category_list:
			category_name: str = str(configured_category).lower()
			keywords: List[str] = keyword_map.get(category_name, [])
			score: int = sum(1 for keyword in keywords if keyword in searchable_text)
			if score > highest_score:
				highest_score = score
				selected_category = configured_category

		# 8. Append the row and save the workbook.
		row = (total, normalized_date, items_formatted, location_str, selected_category)
		new_row: pd.DataFrame = pd.DataFrame([row], columns=columns)
		df = pd.concat([df, new_row], ignore_index=True)
		df.to_excel(path, index=False)

		return {
			"success": True,
			"appended": True,
			"message": "Receipt saved successfully.",
			"signature": [normalized_date, total_rounded, merchant],
		}

	except Exception as e:
		print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		return {
			"success": False,
			"appended": False,
			"message": f"Unable to save receipt: {type(e).__name__}.",
			"signature": [normalized_date, round(total, 2), merchant],
		}

@tool
def query_spending(question: str, store_path: Optional[str] = None) -> dict:
	"""
	Overview: Aggregates the stored Excel receipt data with pandas to answer natural-language spending questions. It loads the .xlsx store, normalizes columns, and computes aggregates such as total spend by category, by month, by merchant, or top items by spend, depending on cues in the question. If the store is missing or the data is insufficient, it returns an insufficient_data marker so the caller LLM can decline gracefully instead of fabricating numbers. No currency conversion is performed; if the user explicitly asks for conversion, the tool returns requires_fx True so the caller can say conversion is unavailable or use an authorized rate source. This is a non-terminal tool: the caller LLM phrases the final natural-language answer. The store path must reside in the working directory or the directory configured by env var RECEIPT_STORE_DIR.
	Caller LLM: chat_llm in the chat node question flow.
	Outside-the-Tool Work (Tool Handler Function Responsibilities): None. The caller LLM turns the returned answer_blocks into one user-visible reply, sets state latest, appends an AIMessage, sets next_action to idle (or pending_question if a follow-up is expected), and ends the run.
	Inside-the-Tool Work (Tool Responsibilities): Load data, parse dates, aggregate by the question, and return compact structured results or an insufficient-data marker.
	Instructions:
	1. Read question; use store_path if provided else ./spending.xlsx.
	2. Load the xlsx with pandas. If the file is missing, empty, or lacks required columns, return insufficient_data True with a compact reason.
	3. Coerce cost to float and date to datetime; parse items, location, and category.
	4. Detect intent cues in the question: category name, month or date range, merchant, top N, group-by dimension.
	5. Compute the relevant aggregates, for example sum by category within a month, top items by spend, or per-merchant totals. Never fabricate values not present in the data.
	6. If the requested aggregation cannot be computed from the available rows, return insufficient_data True.
	7. Return the aggregated answer_blocks plus the set of currencies found in the store; do not convert. If conversion is explicitly requested, set requires_fx True and provide no converted values.
	State Updates (on the caller function): None.
	Args:
	- question (str): The user's natural-language spending question.
	- store_path (Optional[str]): Path to the .xlsx store; defaults to ./spending.xlsx.
	Returns:
	- dict: Keys: success (bool), insufficient_data (bool), reason (str, optional), answer_blocks (list of dicts or None), currencies (list), requires_fx (bool).
	"""
	path: str = store_path.strip() if isinstance(store_path, str) and store_path.strip() else './spending.xlsx'
	required_columns: set = {'cost', 'date', 'items', 'location', 'category'}

	def _insufficient(reason: str) -> dict:
		return {
			"success": True,
			"insufficient_data": True,
			"reason": reason,
			"answer_blocks": None,
			"currencies": [],
			"requires_fx": False,
		}

	# Enforce filesystem-path safety: the store must reside in the working
	# directory or the directory configured by env var RECEIPT_STORE_DIR.
	approved_roots: List[str] = [os.path.abspath('.')]
	configured_root: str = os.getenv('RECEIPT_STORE_DIR', '').strip()
	if configured_root:
		approved_roots.append(os.path.abspath(configured_root))
	resolved_path: str = os.path.abspath(path)
	if not any(
		resolved_path == root or resolved_path.startswith(root + os.sep)
		for root in approved_roots
	):
		return _insufficient("spending store path not permitted")

	try:
		df: pd.DataFrame = pd.read_excel(path)
	except FileNotFoundError:
		return _insufficient("spending store not found")
	except Exception:
		return _insufficient("unable to read the spending store")

	if df is None or df.empty:
		return _insufficient("spending store is empty")
	if not required_columns.issubset(df.columns):
		return _insufficient("required spending columns are missing")

	df = df.copy()
	df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
	df['date'] = pd.to_datetime(df['date'], errors='coerce')
	df = df.dropna(subset=['cost'])
	if df.empty:
		return _insufficient("spending store contains no valid cost rows")

	item_pattern: re.Pattern = re.compile(
		r'^(.*?)\s*\(\s*([\d.]+)\s*x\s*([\d.]+)\s*([A-Za-z$€£¥]+)\s*\)$'
	)

	def _parse_items(frame: pd.DataFrame):
		totals: Dict[str, float] = {}
		currencies: set = set()
		for cell in frame['items'].fillna('').astype(str):
			for raw_entry in cell.split(','):
				match = item_pattern.match(raw_entry.strip())
				if not match:
					continue
				description, quantity_raw, price_raw, currency = match.groups()
				currencies.add(currency)
				try:
					total: float = float(quantity_raw) * float(price_raw)
				except ValueError:
					continue
				key: str = description.strip()
				totals[key] = totals.get(key, 0.0) + total
		return totals, currencies

	_, found_currencies = _parse_items(df)
	currency_list: List[str] = sorted(found_currencies)

	q: str = str(question or '').lower()

	# Convert the question to lower case and detect FX intent.
	fx_pattern: re.Pattern = re.compile(
		r'\b(convert|conversion|exchange|fx)\b|\b(in|to)\s+(usd|eur|gbp|jpy|cad|aud|cny|chf|inr)\b'
	)
	requires_fx: bool = bool(fx_pattern.search(q))
	if requires_fx:
		return {
			"success": True,
			"insufficient_data": True,
			"reason": "currency conversion is unavailable",
			"answer_blocks": None,
			"currencies": currency_list,
			"requires_fx": True,
		}

	# Year / month / date-range cues.
	year_filter: Optional[int] = None
	year_matches: List[str] = re.findall(r'\b20\d{2}\b', q)
	if year_matches:
		year_filter = int(year_matches[0])

	month_filter: Optional[str] = None
	date_from: Optional[str] = None
	date_to: Optional[str] = None
	ym_matches: List[tuple] = re.findall(r'\b(20\d{2})[-/](0?[1-9]|1[0-2])\b', q)
	if ym_matches:
		periods: List[str] = sorted({f"{y}-{int(m):02d}" for y, m in ym_matches})
		if len(periods) >= 2:
			date_from, date_to = periods[0], periods[-1]
		else:
			month_filter = periods[0]

	if month_filter is None and date_from is None:
		month_map: Dict[str, int] = {
			'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
			'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
			'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10,
			'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12,
		}
		for month_name, month_num in month_map.items():
			if re.search(r'\b' + month_name + r'\b', q):
				base_year: int = year_filter if year_filter is not None else date.today().year
				month_filter = f"{base_year:04d}-{month_num:02d}"
				break

	if month_filter is None and date_from is None:
		today: date = date.today()
		if re.search(r'\blast month\b', q):
			if today.month == 1:
				month_filter = f"{today.year - 1}-12"
			else:
				month_filter = f"{today.year:04d}-{today.month - 1:02d}"
		elif re.search(r'\bthis month\b', q):
			month_filter = f"{today.year:04d}-{today.month:02d}"

	# Category cue: match against the default taxonomy and distinct category values.
	default_taxonomy: List[str] = [
		'Groceries', 'Dining', 'Transport', 'Utilities', 'Housing',
		'Health', 'Entertainment', 'Shopping', 'Travel', 'Other',
	]
	category_keyword: Optional[str] = None
	category_candidates: set = set(c.lower() for c in default_taxonomy)
	category_candidates.update(
		str(c).strip().lower() for c in df['category'].dropna().astype(str).unique() if str(c).strip()
	)
	for candidate in category_candidates:
		if re.search(r'\b' + re.escape(candidate) + r'\b', q):
			category_keyword = candidate
			break

	# Merchant cue: match question tokens against words found in location cells.
	stopwords: set = {
		'the', 'and', 'for', 'with', 'from', 'ave', 'st', 'rd', 'street', 'road',
		'city', 'store', 'shop', 'spent', 'spend', 'spending', 'receipt', 'receipts',
		'month', 'year', 'last', 'this', 'total', 'much', 'how', 'what', 'many', 'top',
		'highest', 'by', 'in', 'on', 'at', 'of', 'to', 'per', 'my', 'did', 'do', 'we',
		'me', 'give', 'show', 'see', 'list', 'over', 'during', 'since', 'all', 'sum',
		'amount', 'category', 'categories', 'items', 'item', 'merchant', 'merchants',
		'location', 'locations', 'between', 'cost', 'costs',
	}
	location_token_map: Dict[str, set] = {}
	for location_value in df['location'].dropna().astype(str).unique():
		for token in re.findall(r"[a-zA-Z][a-zA-Z0-9']{2,}", location_value.lower()):
			if token in stopwords:
				continue
			location_token_map.setdefault(token, set()).add(location_value)
	q_tokens: set = set(re.findall(r"[a-zA-Z][a-zA-Z0-9']{2,}", q))
	merchant_tokens: List[str] = [token for token in q_tokens if token in location_token_map]

	# Apply all detected filters.
	working: pd.DataFrame = df.copy()
	if date_from is not None and date_to is not None:
		working['_month'] = working['date'].dt.to_period('M').astype(str)
		working = working[(working['_month'] >= date_from) & (working['_month'] <= date_to)]
		working = working.drop(columns=['_month'])
	elif month_filter is not None:
		working['_month'] = working['date'].dt.to_period('M').astype(str)
		working = working[working['_month'] == month_filter]
		working = working.drop(columns=['_month'])
	elif year_filter is not None:
		working = working[working['date'].dt.year == year_filter]

	if category_keyword is not None:
		working = working[
			working['category'].fillna('').astype(str).str.lower().str.contains(re.escape(category_keyword), regex=True)
		]

	if merchant_tokens:
		working = working[
			working['location'].fillna('').astype(str).str.lower().apply(
				lambda cell: any(token in cell for token in merchant_tokens)
			)
		]

	if working.empty:
		return {
			"success": True,
			"insufficient_data": True,
			"reason": "no spending rows match the requested filters",
			"answer_blocks": None,
			"currencies": currency_list,
			"requires_fx": False,
		}

	# Grouping cue.
	top_n: Optional[int] = None
	top_match = re.search(r'\b(?:top|highest)\s+(\d+)\b', q)
	if top_match:
		top_n = int(top_match.group(1))

	grouping: str = 'total'
	if re.search(r'\btop items\b', q) or (
		top_n is not None and re.search(r'\b(top|highest)\b', q) and re.search(r'\bitems?\b', q)
	):
		grouping = 'items'
	elif re.search(r'\bby category\b', q):
		grouping = 'category'
	elif re.search(r'\bby month\b', q):
		grouping = 'month'
	elif re.search(r'\bby merchant\b', q):
		grouping = 'merchant'

	block_currency: str = currency_list[0] if len(currency_list) == 1 else 'unknown'
	answer_blocks: List[Dict[str, Any]] = []

	if grouping == 'items':
		item_totals, filtered_currencies = _parse_items(working)
		if filtered_currencies:
			block_currency = sorted(filtered_currencies)[0] if len(filtered_currencies) == 1 else 'unknown'
		if item_totals:
			ranked: List[tuple] = sorted(item_totals.items(), key=lambda kv: kv[1], reverse=True)
			if top_n is not None:
				ranked = ranked[:top_n]
			answer_blocks = [
				{"label": str(label), "amount": round(float(amount), 2), "currency": block_currency}
				for label, amount in ranked
			]
		else:
			answer_blocks = []
	elif grouping == 'category':
		category_series = working.groupby('category')['cost'].sum().sort_values(ascending=False)
		if top_n is not None:
			category_series = category_series.head(top_n)
		answer_blocks = [
			{"label": str(label), "amount": round(float(amount), 2), "currency": block_currency}
			for label, amount in category_series.items()
		]
	elif grouping == 'month':
		if working['date'].notna().any():
			month_series = working.groupby(working['date'].dt.to_period('M').astype(str))['cost'].sum()
			if top_n is not None:
				month_series = month_series.sort_values(ascending=False).head(top_n)
			else:
				month_series = month_series.sort_index()
			answer_blocks = [
				{"label": str(label), "amount": round(float(amount), 2), "currency": block_currency}
				for label, amount in month_series.items()
			]
		else:
			answer_blocks = []
	elif grouping == 'merchant':
		merchant_series = working.groupby('location')['cost'].sum().sort_values(ascending=False)
		if top_n is not None:
			merchant_series = merchant_series.head(top_n)
		answer_blocks = [
			{"label": str(label), "amount": round(float(amount), 2), "currency": block_currency}
			for label, amount in merchant_series.items()
		]
	else:
		total_amount: float = float(working['cost'].sum())
		answer_blocks = [
			{"label": "total", "amount": round(total_amount, 2), "currency": block_currency}
		]

	if not answer_blocks:
		return {
			"success": True,
			"insufficient_data": True,
			"reason": "the requested spending breakdown cannot be computed from the available rows",
			"answer_blocks": None,
			"currencies": currency_list,
			"requires_fx": False,
		}

	return {
		"success": True,
		"insufficient_data": False,
		"reason": None,
		"answer_blocks": answer_blocks,
		"currencies": currency_list,
		"requires_fx": False,
	}
# TODO: Add Tools (if needed)



''' LLM '''
chat_llm = myChatOpenAI(
	temperature= 0.0
).bind_tools([parse_receipt_image, validate_receipt, append_receipt, query_spending])




''' Helpful Functions '''
def classify_inbound_message(message: BaseMessage) -> Literal["receipt", "question"]:
	"""
	Overview: Determines whether an inbound user message is a receipt payload (image attachment, existing image file path, raw image bytes, data URL, or receipt-type marker) or a plain-text spending question. The chat node routes by payload shape, not by LLM classification, so this helper isolates that routing logic.
	Caller Node: chat
	Instructions:
	1. Inspect message.content. Multimodal HumanMessages may have a list of content blocks; plain text is a string; bytes are accepted for raw image payloads.
	2. If content is a list, return 'receipt' if any block is of type 'image' or contains 'image_url' or a data URL; otherwise continue with the text parts concatenated.
	3. If content is bytes, check for a common image signature (JPEG, PNG, GIF, WebP) or return 'receipt' if the payload is non-empty and not valid UTF-8 text.
	4. If content is a string: strip whitespace. Return 'receipt' if it is a path to an existing image file, a 'data:image/' URL, a 'file://' URL, or contains receipt-type markers such as 'receipt', 'receipt image', 'scan', 'expense'.
	5. Otherwise return 'question'.
	Args:
	- message (BaseMessage): the inbound HumanMessage from state['messages'][-1].
	Returns:
	- Literal['receipt', 'question']: 'receipt' routes to receipt ingestion; 'question' routes to spending Q&A.
	"""
	content = message.content

	if isinstance(content, list):
		text_parts: List[str] = []
		for block in content:
			if isinstance(block, dict):
				if block.get('type') == 'image' or 'image_url' in block:
					return 'receipt'
				text_parts.append(str(block.get('text', '')))
			elif isinstance(block, str):
				if 'data:image/' in block:
					return 'receipt'
				text_parts.append(block)
			else:
				text_parts.append(str(block))
		content = ''.join(text_parts)
	elif isinstance(content, bytes):
		if (
			content[:3] == b'\xff\xd8\xff'
			or content[:8] == b'\x89PNG\r\n\x1a\n'
			or content[:4] == b'GIF8'
			or (len(content) >= 12 and content[:4] == b'RIFF' and content[8:12] == b'WEBP')
		):
			return 'receipt'
		try:
			content = content.decode('utf-8')
		except UnicodeDecodeError:
			return 'receipt'
	elif not isinstance(content, str):
		return 'question'

	text: str = content.strip()
	if text.startswith(('data:image/', 'file://')):
		return 'receipt'
	if re.search(r'\.(png|jpe?g|gif|webp|bmp)$', text, re.IGNORECASE) and os.path.exists(text):
		return 'receipt'
	if re.search(r'receipt|receipt image|scan|expense', text, re.IGNORECASE):
		return 'receipt'
	return 'question'

def resolve_receipt_store_path(state: AgentSchema) -> str:
	"""
	Overview: Resolves the configured path to the local .xlsx receipt store from state['receipt_store_path'], defaulting to './spending.xlsx' when the key is absent or empty. The append_receipt flow writes one row per receipt to this file, creating it with a header on first use.
	Caller Node: chat (receipt flow, before calling append_receipt)
	Instructions:
	1. Read state.get('receipt_store_path').
	2. If it is a non-empty string, return it.
	3. Otherwise return './spending.xlsx'.
	Args:
	- state (AgentSchema): current graph state; only receipt_store_path is read.
	Returns:
	- str: path to the receipt store.
	"""
	path = state.get('receipt_store_path')
	if isinstance(path, str) and path.strip():
		return path.strip()
	return './spending.xlsx'

def resolve_categories(state: AgentSchema) -> List[str]:
	default_categories: List[str] = [
		'Groceries', 'Dining', 'Transport', 'Utilities', 'Housing',
		'Health', 'Entertainment', 'Shopping', 'Travel', 'Other',
	]
	categories: Any = state.get('categories', default_categories)
	if isinstance(categories, list) and any(isinstance(c, str) and c.strip() for c in categories):
		return [c for c in categories if isinstance(c, str) and c.strip()]
	return default_categories

def receipt_dedupe_signature(receipt: dict) -> tuple[str, float, str]:
	"""
	Overview: Builds the canonical dedupe signature (date, total, merchant) from a validated receipt so append_receipt can detect duplicates already recorded in the conversation or the Excel store. Deduplication is on (date, total, merchant); this helper normalizes each part for stable tuple comparisons.
	Caller Node: chat (receipt flow, via append_receipt tool)
	Instructions:
	1. Accept a validated receipt as a dict or an object with .date, .cost, and .location attributes. For a dict, read 'date', 'cost', and 'location' (location may be a dict with 'merchant' or an object).
	2. Normalize date: if it is a datetime, format as YYYY-MM-DD; if a string, extract the leading YYYY-MM-DD; if unparseable use 'unknown'.
	3. Normalize total: convert cost to float and round to 2 decimals; on failure use 0.0.
	4. Normalize merchant: take location.merchant (or nested dict key), strip whitespace and lowercase; if absent use ''.
	5. Return (date, total, merchant).
	Args:
	- receipt (dict): validated receipt data from validate_receipt.
	Returns:
	- tuple[str, float, str]: canonical signature (date, total, merchant).
	"""
	if isinstance(receipt, dict):
		raw_date: Any = receipt.get('date')
		raw_cost: Any = receipt.get('cost')
		raw_location: Any = receipt.get('location')
	else:
		raw_date = getattr(receipt, 'date', None)
		raw_cost = getattr(receipt, 'cost', None)
		raw_location = getattr(receipt, 'location', None)

	# Normalize date ----------------------------------------------------------
	if isinstance(raw_date, datetime):
		date_str: str = raw_date.strftime('%Y-%m-%d')
	elif isinstance(raw_date, date):
		date_str = raw_date.isoformat()
	elif isinstance(raw_date, str):
		candidate: str = raw_date.strip()[:10]
		date_str = candidate if re.match(r'^\d{4}-\d{2}-\d{2}$', candidate) else 'unknown'
	else:
		date_str = 'unknown'

	# Normalize total ---------------------------------------------------------
	try:
		total: float = round(float(raw_cost), 2)
	except (TypeError, ValueError):
		total = 0.0

	# Normalize merchant ------------------------------------------------------
	if isinstance(raw_location, dict):
		raw_merchant: Any = raw_location.get('merchant')
	else:
		raw_merchant = getattr(raw_location, 'merchant', None)

	if raw_merchant is None:
		merchant: str = ''
	else:
		merchant = str(raw_merchant).strip().lower()

	return (date_str, total, merchant)

def load_or_create_spreadsheet(path: str, columns: List[str]) -> pd.DataFrame:
	"""
	Overview: Loads the existing .xlsx receipt store into a pandas DataFrame, or creates an empty DataFrame with exactly the expected header columns when the file does not exist yet. This guarantees append_receipt always appends rows to a store whose header is [cost, date, items, location, category] on first use.
	Caller Node: chat (receipt flow, via append_receipt tool)
	Instructions:
	1. Check whether path exists and is a non-empty file.
	2. If it exists, load it with pandas.read_excel(path).
	3. If the loaded DataFrame is missing any of the columns in columns, add missing columns filled with None.
	4. If the file does not exist, create an empty pandas DataFrame with columns=columns.
	5. Return the DataFrame. Do not write to disk; append_receipt performs the write.
	Args:
	- path (str): path to the .xlsx receipt store.
	- columns (List[str]): expected column names, exactly ['cost', 'date', 'items', 'location', 'category'].
	Returns:
	- pd.DataFrame: DataFrame with the expected columns, empty or loaded.
	"""
	if os.path.exists(path) and os.path.getsize(path) > 0:
		df: pd.DataFrame = pd.read_excel(path)
		for col in columns:
			if col not in df.columns:
				df[col] = None
		return df
	return pd.DataFrame(columns=columns)

def build_final_state_update(state: AgentSchema, reply: str, next_action: Literal["idle", "await_resend", "pending_question"], mode: Optional[str], pending_question: Optional[Any]) -> AgentSchema:
	"""
	Overview: Builds the final state update that completes one chat run: it records the single user-visible reply in state['latest'], appends the outbound AIMessage to state['messages'], and sets state['next_action'] (and optionally state['mode'] and state['pending_question']) so the next inbound event resumes correctly. This enforces the turn-based contract that waiting for the user is expressed as a reply plus resume markers, never as a pause inside the graph.
	Caller Node: chat (end of receipt flow and question flow)
	Instructions:
	1. Create an AIMessage with content=reply.
	2. Return a dict (state update) with the keys 'latest': reply, 'messages': [new AIMessage], 'next_action': next_action, 'mode': mode, and 'pending_question': pending_question.
	3. Do NOT return the full existing messages list in 'messages'; LangGraph's add reducer appends the list, so pass only the single new AIMessage.
	4. Leave all other state keys unchanged; LangGraph merges the returned update with the persisted state.
	Args:
	- state (AgentSchema): current graph state; read for context but not mutated in place.
	- reply (str): the exact user-visible reply for this run.
	- next_action (Literal['idle', 'await_resend', 'pending_question']): resume marker; default 'idle'.
	- mode (Optional[str]): flow mode to persist; None clears it.
	- pending_question (Optional[Any]): question/context to persist, or None to clear.
	Returns:
	- AgentSchema: state update dict containing latest, messages, next_action, mode, and pending_question.
	"""
	message: AIMessage = AIMessage(content=reply)
	return {
		'latest': reply,
		'messages': [message],
		'next_action': next_action,
		'mode': mode,
		'pending_question': pending_question,
	}
# TODO: Add Helpful Functions (if needed)


def route_chat_tools(state: AgentSchema) -> Literal["standard_tools", "append_receipt_handler", "end"]:
	"""Route the latest chat output to the requested tool handler or finish the run."""
	last_message: BaseMessage = state['messages'][-1]
	if isinstance(last_message, AIMessage) and last_message.tool_calls:
		for tool_call in last_message.tool_calls:
			if tool_call['name'] == 'append_receipt':
				return 'append_receipt_handler'
		return 'standard_tools'
	return 'end'


def append_receipt_handler(state: AgentSchema) -> AgentSchema:
	"""Invoke terminal append_receipt and convert its result into the documented final state update."""
	# 1. Locate the append_receipt tool call in the last AIMessage.
	last_message: BaseMessage = state['messages'][-1]
	tool_call: Optional[dict] = None
	for call in (last_message.tool_calls or []):
		if call.get('name') == 'append_receipt':
			tool_call = call
			break

	# Safe fallback when the expected tool call is missing.
	if tool_call is None:
		return build_final_state_update(
			state,
			'Unable to save the receipt. Please try again.',
			'await_resend',
			'receipt',
			None,
		)

	# 2. Extract the tool arguments, defaulting optional keys to None.
	raw_args: dict = tool_call.get('args') or {}
	args: dict = {
		'validated_receipt': raw_args.get('validated_receipt'),
		'store_path': raw_args.get('store_path'),
		'categories': raw_args.get('categories'),
	}

	# 3. Invoke the terminal append_receipt tool.
	result: dict = append_receipt.invoke(args)

	# 4. ToolMessage carrying the JSON-serialized tool result.
	tool_message: ToolMessage = ToolMessage(
		content=json.dumps(result, ensure_ascii=False),
		tool_call_id=tool_call.get('id', ''),
		name='append_receipt',
	)

	# 5. Build the final state update.
	if result.get('success'):
		reply: str = result.get('message', 'Receipt saved successfully.')
		update: AgentSchema = build_final_state_update(state, reply, 'idle', 'receipt', None)
		dedupe_set: set = set(state.get('dedupe_state') or set())
		if isinstance(result.get('signature'), list):
			dedupe_set.add(tuple(result['signature']))
		update['dedupe_state'] = dedupe_set
	else:
		reply = result.get('message', 'Unable to save the receipt. Please resend a clearer photo.')
		update = build_final_state_update(state, reply, 'await_resend', 'receipt', None)

	# 6. Override messages: ToolMessage + final AI reply (AIMessage with tool call already in state).
	update['messages'] = [tool_message, AIMessage(content=reply)]
	return update



''' Nodes '''
def chat(state: AgentSchema) -> AgentSchema:
	""" Execution: LLM+TOOLS. The single LLM-driven brain. If state.mode/next_action indicates a resumable flow such as await_resend, route directly back to the appropriate handling path with corrective context. Otherwise inspect the inbound message: an image attachment, image path, or receipt-type marker routes to the receipt flow; plain text routes to the spending-question flow. Receipt flow calls tools: parse_receipt_image (image to base64 data URL, then OpenRouter /api/v1/chat/completions with model from env var default google/gemini-2.5-flash, JSON via response_format schema with a permissive parser fallback), validate_receipt (Pydantic models plus sum-of-items versus total sanity check within approximately 2 percent, one bounded retry on low confidence below 0.7), and append_receipt (dedup on date plus total plus merchant; duplicate skips the append and replies with an already-recorded message; creates ./spending.xlsx with header cost, date, items, location, category on first use, ISO dates, item formatted as item (qty x price currency), category auto-assigned defaulting to Other). Question flow calls query_spending (pandas aggregation over the Excel store) and phrases the answer with the same OpenRouter LLM, converting currency only when explicitly asked. The node sends exactly one outbound user-visible reply per run and sets mode/next_action (for example await_resend after a failed parse, or pending_question for follow-up) before ending. Never logs raw image bytes or raw OCR text beyond minimal error details; declines gracefully when data is insufficient; never fabricates answers. 
	Execution: LLM+TOOLS. The single LLM-driven brain for the receipt_spending_assistant graph. Reached after the start node appends the new inbound HumanMessage to state['messages']. It routes every run into one of two flows: receipt ingestion (image attachment, local file path, raw bytes, or receipt-type marker) or spending Q&A (plain text). Resumable flows are handled first by inspecting state['mode'] and state['next_action']; for example 'await_resend' means the user is resending a failed receipt, and 'pending_question' means the user is answering or following up on a stored question. The node must produce exactly one outbound user-visible reply, store it in state['latest'], append an AIMessage to state['messages'], update resume markers, and return the state so the built-in end node persists it. Waiting for the user is never modeled inside this node: any need for more input is expressed as a reply plus state['next_action'] and optional state['pending_question'], then the run ends.
	
	Step-by-step:
	1. Log entry with print_function_name().
	2. Read the new inbound message from state['messages'][-1] and inspect state['mode'], state['next_action'], state['pending_question'], and state['dedupe_state'].
	3. If state['next_action'] == 'await_resend', clear that marker, carry forward the previous error context, and route the new message to the receipt flow as a resend attempt.
	4. If state['next_action'] == 'pending_question', clear it, treat the new message as the answer/follow-up to state['pending_question'], and route to the question flow.
	5. Otherwise classify the payload by shape: image attachment/file path/bytes/receipt marker -> receipt flow; plain text -> spending-question flow.
	6. Receipt flow:
	   a. Ask the LLM to call parse_receipt_image with the image source. The tool encodes the image as a base64 data URL, calls the OpenAI-compatible OpenRouter /api/v1/chat/completions endpoint with the vision model from the configured env var (default google/gemini-2.5-flash), requests JSON via response_format schema when supported, and otherwise falls back to a strict return-JSON-only prompt plus a permissive parser that extracts the first balanced JSON/code fence. It should extract cost (grand total including tax), date (purchase date with processing date as fallback), line items (description, quantity, unit price), currency, and location (merchant plus address/city when present).
	   b. Ask the LLM to call validate_receipt on the parsed JSON. The tool applies Pydantic-style models (cost, date, items with description/quantity/unit_price, currency, location) and a sanity check that the sum of line-item totals is within about 2 percent of the grand total. On invalid JSON or confidence below 0.7 it retries once.
	   c. If validation fails: log only minimal error details (never raw image bytes or raw OCR text), do NOT append anything, set state['latest'] to a clear resend prompt, set state['mode']/state['next_action'] to 'await_resend', append the AI message, and end the run.
	   d. If validation succeeds, ask the LLM to call append_receipt. The tool de-duplicates on (date, total, merchant); on duplicate it skips the append and returns an already-recorded message. Otherwise it creates the Excel store (default ./spending.xlsx, configurable) with header [cost, date, items, location, category] on first use, appends one row per receipt, stores the detected currency, normalizes date to ISO YYYY-MM-DD, formats items as 'item (qty x price currency)', defaults quantity to 1 when absent, and auto-assigns one category from the configurable taxonomy [Groceries, Dining, Transport, Utilities, Housing, Health, Entertainment, Shopping, Travel, Other], defaulting to Other.
	   e. Set state['latest'] to the success or duplicate message returned by append_receipt; set next_action to 'idle' unless a follow-up is required.
	7. Question flow:
	   a. Ask the LLM to call query_spending with the user's natural-language question. The tool loads the Excel data with pandas and computes aggregates such as month-by-category totals, top items by spend, and per-merchant totals. If data is insufficient it returns an empty/insufficient result.
	   b. Use the same OpenRouter LLM to phrase a natural-language answer from the tool result. Convert currency only if the user explicitly asked; decline gracefully when data is insufficient and never fabricate numbers.
	   c. Set state['latest'] to that answer, set next_action to 'idle' (or 'pending_question' if a follow-up is expected), and append the AI message.
	8. Ensure exactly one user-visible reply per run, update state['dedupe_state'] if a receipt signature was seen, and return the updated state for the end node to persist. On unexpected exceptions, log the traceback in DEBUG mode, produce a safe user-facing fallback, do not append a receipt, and do not fabricate data.
	
	State inputs expected by this node:
	- state['messages'] (list[BaseMessage]): conversation history; messages[-1] is the new inbound HumanMessage for this run.
	- state['mode'] (Optional[str]): persistent flow mode from previous runs.
	- state['next_action'] (Optional[str]): resume marker such as 'idle', 'await_resend', or 'pending_question'.
	- state['pending_question'] (Optional[Dict[str, Any]] or str): stored question/context awaiting user input.
	- state['latest'] (Optional[str]): last user-visible reply, for context.
	- state['dedupe_state'] (Optional[set]): receipt signatures (date, total, merchant) already seen in this conversation; the append tool may also check the Excel file.
	- state['processed_event_ids'] (Optional[set]): optional dedupe for retried inbound events, to avoid duplicate side effects.
	- state['receipt_store_path'] (Optional[str]): path to the .xlsx store; if absent the default ./spending.xlsx is used.
	
	State outputs produced by this node:
	- state['messages']: original messages plus the new outbound AIMessage.
	- state['latest']: the single final user-facing reply string.
	- state['mode']: updated flow mode if changed.
	- state['next_action']: updated resume marker for the next run.
	- state['pending_question']: set, updated, or cleared depending on the flow.
	- state['dedupe_state']: updated with new receipt signatures when a receipt is appended or detected as duplicate.
	- state['processed_event_ids']: updated when inbound event dedupe is applied.
	
	Possible tools (bound to chat_llm via chat_llm.bind_tools([...]); a ToolNode executes the requested tools and returns ToolMessages before the final reply is composed):
	- parse_receipt_image(image_source: Union[str, bytes]) -> dict: converts the image to a base64 data URL and runs the OpenRouter vision LLM to extract structured receipt JSON (cost, date, items, currency, location).
	- validate_receipt(receipt_json: dict) -> dict: Pydantic-style validation and sum-of-items vs total sanity check with one bounded retry; returns normalized receipt data or a clear error.
	- append_receipt(validated_receipt: dict) -> dict: de-duplicates on (date, total, merchant), writes one row per receipt to the .xlsx store with columns [cost, date, items, location, category], and returns success/duplicate/failure status.
	- query_spending(question: str) -> dict: aggregates the stored Excel data with pandas and returns structured spending insights or an insufficient-data marker.
	
	Possible helpful functions:
	- print_function_name(): existing debug helper used at node entry.
	- safe_invoke(...): existing guarded LLM invocation wrapper; use it for chat_llm / OpenRouter calls so network or model errors are handled uniformly.
	No new prompt-building, LLM-output-parsing, or formatting helper functions should be added; those concerns stay inside the tools and the LLM prompt.
	"""

	print_function_name()
	try:
		# 1. Read the conversation history and flow context ----------------------
		messages: List[BaseMessage] = state.get('messages') or []
		mode: Optional[str] = state.get('mode')
		next_action: Optional[str] = state.get('next_action')
		pending_question: Optional[Any] = state.get('pending_question')
		dedupe_state: set = state.get('dedupe_state') or set()
		receipt_store_path: Optional[str] = state.get('receipt_store_path')
		categories: Optional[List[str]] = state.get('categories')

		# 2. Build a safe inbound content snippet for the prompt ------------------
		# Never leak raw image bytes, data URLs, or file URIs into the prompt.
		inbound_message: BaseMessage = messages[-1] if messages else HumanMessage(content='')
		inbound_content: Any = inbound_message.content
		if isinstance(inbound_content, str):
			inbound_content_text: str = inbound_content.strip()
			if inbound_content_text.startswith('data:image/') or inbound_content_text.startswith('file://'):
				inbound_content_text = '<image payload>'
		else:
			inbound_content_text = '<non-text payload>'

		# 3. Build the system prompt with the current flow context ----------------
		try:
			prompt: str = prompts.CHAT_PROMPT.format(
				mode=mode or 'None',
				next_action=next_action or 'None',
				pending_question=str(pending_question) if pending_question is not None else 'None',
				dedupe_state=str(dedupe_state) if dedupe_state else 'None',
				receipt_store_path=receipt_store_path if receipt_store_path else './spending.xlsx',
				categories=str(categories) if categories else 'None',
				inbound_message=inbound_content_text,
			)
		except (KeyError, ValueError, IndexError):
			prompt = prompts.CHAT_PROMPT

		# 4. Invoke the LLM once with the full conversation history ---------------
		result: BaseMessage = safe_invoke(chat_llm, messages=[SystemMessage(content=prompt)] + messages)

		# 5. If the LLM requested tools, return only the AIMessage ----------------
		# The conditional edge routes to standard_tools or append_receipt_handler.
		if getattr(result, 'tool_calls', None):
			return {'messages': [result]}

		# 6. Final user-visible reply ---------------------------------------------
		reply: str = str(result.content or '').strip() or 'I did not understand that. Please try again.'

		# 7. Determine the flow outcome from the most recent ToolMessage -----------
		next_action_out: Literal['idle', 'await_resend', 'pending_question'] = 'idle'
		mode_out: Optional[str] = None
		new_dedupe_state: set = set(dedupe_state)
		dedupe_updated: bool = False

		for msg in reversed(messages):
			if isinstance(msg, HumanMessage):
				break
			if isinstance(msg, ToolMessage):
				tool_name: str = getattr(msg, 'name', '') or ''
				data: dict = {}
				content: Any = msg.content
				if isinstance(content, dict):
					data = content
				elif isinstance(content, str):
					try:
						parsed: Any = json.loads(content)
						if isinstance(parsed, dict):
							data = parsed
					except (TypeError, ValueError):
						data = {}
				if tool_name == 'parse_receipt_image' or 'receipt_json' in data:
					if not data.get('success'):
						next_action_out = 'await_resend'
						mode_out = 'receipt'
				elif tool_name == 'validate_receipt' or 'validated_receipt' in data:
					if not data.get('success'):
						next_action_out = 'await_resend'
						mode_out = 'receipt'
				elif tool_name == 'query_spending' or 'answer_blocks' in data:
					next_action_out = 'idle'
					mode_out = 'question'
				elif tool_name == 'append_receipt' or 'appended' in data:
					if data.get('success'):
						next_action_out = 'idle'
						mode_out = 'receipt'
						signature: Any = data.get('signature')
						if isinstance(signature, list):
							new_dedupe_state.add(tuple(signature))
							dedupe_updated = True
					else:
						next_action_out = 'await_resend'
						mode_out = 'receipt'
				break

		# 8. Clear pending_question once it has been answered ----------------------
		pending_question_out: Optional[Any] = None
		if next_action == 'pending_question':
			pending_question_out = None
		else:
			pending_question_out = pending_question

		# 9. Build and return the final state update --------------------------------
		update: AgentSchema = build_final_state_update(state, reply, next_action_out, mode_out, pending_question_out)
		if dedupe_updated:
			update['dedupe_state'] = new_dedupe_state
		return update

	except Exception as e:
		print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
		traceback.print_exc() if DEBUG else None
		return build_final_state_update(state, 'Something went wrong. Please try again.', 'idle', None, None)








''' Graph '''
receipt_spending_assistant_graph = StateGraph(AgentSchema)

receipt_spending_assistant_graph.add_node("chat", chat)
receipt_spending_assistant_graph.add_node(
	"standard_tools",
	ToolNode([parse_receipt_image, validate_receipt, query_spending]),
)
receipt_spending_assistant_graph.add_node("append_receipt_handler", append_receipt_handler)

receipt_spending_assistant_graph.add_edge(START, "chat")
receipt_spending_assistant_graph.add_conditional_edges(
	"chat",
	route_chat_tools,
	{
		"standard_tools": "standard_tools",
		"append_receipt_handler": "append_receipt_handler",
		"end": END,
	},
)
receipt_spending_assistant_graph.add_edge("standard_tools", "chat")
receipt_spending_assistant_graph.add_edge("append_receipt_handler", END)


receipt_spending_assistant_app = receipt_spending_assistant_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(receipt_spending_assistant_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/receipt_spending_assistant_app.png', 'wb') as f:
        f.write(receipt_spending_assistant_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'receipt_spending_assistant'
    os.environ['LANGSMITH_PROJECT'] = 'receipt_spending_assistant'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'receipt_spending_assistant',
            'run_name': 'receipt_spending_assistant',
            'thread_id': 'receipt_spending_assistant', 
        }
    }

    user = '' # TODO: add
    response = receipt_spending_assistant_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
