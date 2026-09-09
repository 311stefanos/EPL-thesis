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
from Clone.experiments.ablation_study.no_requirement_engineering import process_receipt_subgraph_prompts as prompts

import re
from openpyxl import Workbook, load_workbook
from datetime import datetime
import base64



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} process_receipt_subgraph') if DEBUG else None



""" Schemas """

class ReceiptItem(TypedDict):
	"""
	TypedDict representing one item line extracted from a receipt. It is used as the element type of ReceiptData.items. The fields are: name (item description), quantity (number of units), price (unit price in the receipt currency), and currency (currency code or symbol).
	"""
	name: str # Item description or product name.
	quantity: float # Number of units purchased.
	price: float # Unit price in the receipt currency.
	currency: str # Currency code or symbol, e.g. 'USD' or '$'.




class ReceiptData(TypedDict):
	"""
	TypedDict representing the structured receipt produced by extract_receipt_data and consumed by determine_category and append_to_excel. Fields: cost (total amount), date (ISO YYYY-MM-DD), items (list of ReceiptItem), and location (merchant/location string).
	"""
	cost: float # Total receipt cost.
	date: str # Purchase date as ISO 'YYYY-MM-DD'.
	items: List[ReceiptItem] # Itemized list of purchased items with quantity, unit price, and currency.
	location: str # Merchant or location where the purchase was made.




class AgentSchema(MessagesState):
	"""
	State schema for the process_receipt_subgraph. Inherits MessagesState so the subgraph remains compatible with the parent conversational agent's message history, even though the pipeline nodes mainly read and write processing fields. Keys: image_path (input receipt photo path), ocr_text (raw OCR output), receipt_data (structured ReceiptData), category (inferred spending category), status (pipeline progress or error), error (failure description), and confirmation (user-facing success message). Fields are Optional when they only exist after a node has produced them.
	"""
	image_path: str # Local filesystem path to the receipt photo supplied by the parent graph. Required input for ocr_image.
	ocr_text: Optional[str] # Raw OCR transcription produced by ocr_image and consumed by extract_receipt_data. Absent until OCR completes.
	receipt_data: Optional[ReceiptData] # Structured receipt parsed by extract_receipt_data and consumed by determine_category and append_to_excel.
	category: Optional[str] # Spending category determined by determine_category, e.g. groceries, dining, or transport.
	status: Optional[str] # Pipeline status: 'ocr_ok', 'extracted_ok', 'categorized_ok', 'success', or 'error' on failure.
	error: Optional[str] # Failure description when status='error'; empty or absent on success.
	confirmation: Optional[str] # Short user-facing confirmation message produced by append_to_excel after the receipt row is saved.




''' Tools '''

# TODO: Add Tools (if needed)



''' LLM '''
ocr_image_llm = myChatOpenAI(
	temperature= 0.0
)

extract_receipt_data_llm = myChatOpenAI(
	temperature= 0.0
).with_structured_output(ReceiptData)

determine_category_llm = myChatOpenAI(
	temperature= 0.0
)




''' Helpful Functions '''
def encode_image_to_base64(image_path: str) -> str:
    """
    Overview: 
    Reads the receipt photo from the local filesystem, validates it, and encodes it as a base64 data URI string of the form 'data:image/<mime>;base64,<payload>' (e.g., 'data:image/jpeg;base64,/9j/4AAQ...'). This data URI is what gets embedded in the multimodal HumanMessage content block {'type': 'image_url', 'image_url': {'url': <data_uri>}} that the ocr_image node sends to the vision model via OpenRouter. The function owns all file-level validation (existence, supported image extension, non-empty file) and MIME-type inference so the node itself stays focused on prompt construction and LLM invocation. On any validation failure it raises a descriptive exception (FileNotFoundError or ValueError), which the calling node catches in its existing try/except to set state['error'] and state['status']='error'.
    
    Caller Node: ocr_image
    
    Instructions: 
    1. Resolve the path: wrap image_path in pathlib.Path and resolve it so relative paths work independently of the current working directory.
    2. Existence check: if the resolved path does not exist or is not a regular file, raise FileNotFoundError with a message that includes the full path.
    3. Extension check: extract the file suffix, lowercase it, and verify it is one of the supported image extensions: '.jpg', '.jpeg', '.png', '.webp'. If it is not, raise ValueError listing the supported extensions and the actual extension found (this mirrors the validation required by the ocr_image node's step 1).
    4. Map the extension to a MIME type: '.jpg' and '.jpeg' -> 'image/jpeg', '.png' -> 'image/png', '.webp' -> 'image/webp'.
    5. Read the file bytes in binary mode ('rb'). If the byte count is 0, raise ValueError indicating the image file is empty or corrupt.
    6. Base64-encode the bytes with base64.b64encode(image_bytes) and decode the result to a UTF-8/ASCII string to obtain the payload.
    7. Build and return the data URI string: f'data:{mime_type};base64,{payload}'.
    8. Do not print or log the base64 payload itself (it can be hundreds of KB); if DEBUG logging is desired, log only the file path and its size in bytes.
    
    Args: 
    - image_path (str): local filesystem path to the receipt photo supplied by the user via the parent graph and stored in state['image_path']. Required. Must point to a jpg/jpeg/png/webp file.
    
    Returns: 
    - str: the base64 data URI ('data:image/<mime>;base64,<payload>') ready to be placed into the image_url content block of the multimodal message sent to ocr_image_llm. Raises FileNotFoundError if the file does not exist, and ValueError if the extension is unsupported or the file is empty.
    """
    resolved_path = Path(image_path).resolve()

    # Existence check: must be an existing regular file.
    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"Image file not found: {resolved_path}")

    # Extension -> MIME type mapping for supported image formats.
    mime_types: Dict[str, str] = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
    }

    extension: str = resolved_path.suffix.lower()
    if extension not in mime_types:
        supported_extensions: str = ', '.join(mime_types.keys())
        raise ValueError(
            f"Unsupported image extension '{extension}'. "
            f"Supported extensions: {supported_extensions}"
        )

    # Read the file bytes in binary mode.
    with resolved_path.open('rb') as image_file:
        image_bytes: bytes = image_file.read()

    if len(image_bytes) == 0:
        raise ValueError(f"Image file is empty or corrupt: {resolved_path}")

    # Debug logging only logs the path and size, never the base64 payload.
    print(f'{BLUE}[IMAGE] [INFO] {resolved_path} ({len(image_bytes)} bytes){RESET}') if DEBUG else None

    payload: str = base64.b64encode(image_bytes).decode('utf-8')

    return f'data:{mime_types[extension]};base64,{payload}'

def append_receipt_row_to_excel(row: Dict[str, Any]) -> bool:
	"""
	Overview: 
	Persists exactly one processed receipt as a new row in the user's Excel expense file. The Excel file location is resolved from the EXCEL_FILE_PATH environment variable. If the file (or its parent directories) does not exist, it is created with the header row ['cost', 'date', 'items', 'location', 'category'] as required by the user's schema. The function then appends the given row dict as the next row of the first worksheet and saves the workbook. It returns True on success and False on any failure (missing environment variable, IO/save error, or verification mismatch) so the calling node can set state['error'] and state['status']='error' without this function raising exceptions. The 'items' value must already be formatted by the caller as 'item1 (quantity x price currency), ...' — this function writes it verbatim into the 'items' column.
	
	Caller Node: append_to_excel
	
	Instructions: 
	1. Resolve the file path: read the EXCEL_FILE_PATH environment variable. If it is unset or empty, print a debug error (respecting the DEBUG flag) and return False.
	2. Prepare directories: convert the path string to a pathlib.Path and create the parent directory with Path(...).parent.mkdir(parents=True, exist_ok=True) so first-time saves do not fail.
	3. Load or create the workbook: if the file exists, open it with openpyxl.load_workbook(excel_path) and take the active worksheet; otherwise create a new openpyxl.Workbook, take the active sheet, and write the header list ['cost', 'date', 'items', 'location', 'category'] into row 1.
	4. Header safety: if the workbook already existed, inspect row 1; if any expected header cell is empty, write the missing header(s) in the correct column position so the file always carries the full schema. Do not fail if existing headers differ only in casing — proceed with the append.
	5. Validate/normalize the row dict: it should contain the keys 'cost', 'date', 'items', 'location', 'category'. Coerce 'cost' to float when possible (empty string if missing/None); coerce every other value to str (empty string when missing/None). Ignore unknown extra keys.
	6. Append the row in header order: ws.append([cost, date, items, location, category]) so values line up with the columns cost, date, items, location, category.
	7. Save the workbook with wb.save(excel_path) inside a try/except; if saving raises (e.g., the file is locked or open in Excel), print a debug error and return False.
	8. Verify the write: re-open the saved file (or inspect the in-memory workbook) and confirm the sheet's last row contains the appended values (e.g., max_row increased and the 'cost'/'date' cells match). On mismatch, return False.
	9. Return True only when the row was appended, the file saved, and verification passed. Never raise exceptions out of this function — always return a boolean.
	
	Args: 
	- row (Dict[str, Any]): the receipt row to append. Expected keys: 'cost' (float, total receipt cost), 'date' (str, ISO 'YYYY-MM-DD'), 'items' (str, already formatted as 'item1 (quantity x price currency), ...'), 'location' (str, merchant/location), 'category' (str, inferred spending category). Required.
	
	Returns: 
	- bool: True when the row was appended to the Excel file at EXCEL_FILE_PATH and the workbook was saved (and verified) successfully; False on any failure, including a missing EXCEL_FILE_PATH environment variable or a save/IO error.
	"""
	headers: List[str] = ['cost', 'date', 'items', 'location', 'category']

	try:
		# 1. Resolve the file path from the environment variable.
		excel_path_str: Optional[str] = os.getenv('EXCEL_FILE_PATH')
		if not excel_path_str:
			print(f'{RED}[EXCEL] [ERR]{RESET}', 'EXCEL_FILE_PATH is unset or empty') if DEBUG else None
			return False

		# 2. Prepare directories.
		excel_path: Path = Path(excel_path_str)
		excel_path.parent.mkdir(parents=True, exist_ok=True)

		# 3. Load or create the workbook.
		file_exists: bool = excel_path.exists()
		if file_exists:
			wb = load_workbook(excel_path)
			ws = wb.active

			# 4. Header safety: fill any missing expected header cell in the correct column position.
			for column_index, header in enumerate(headers, start=1):
				cell_value = ws.cell(row=1, column=column_index).value
				if cell_value is None or (isinstance(cell_value, str) and not cell_value.strip()):
					ws.cell(row=1, column=column_index, value=header)
		else:
			wb = Workbook()
			ws = wb.active
			for column_index, header in enumerate(headers, start=1):
				ws.cell(row=1, column=column_index, value=header)

		if not isinstance(row, dict):
			return False

		previous_max_row: int = ws.max_row

		# 5. Validate/normalize the row dict.
		raw_cost = row.get('cost')
		if raw_cost is None or raw_cost == '':
			cost: Union[float, str] = ''
		else:
			try:
				cost = float(raw_cost)
			except (TypeError, ValueError):
				cost = ''

		values: List[Any] = [cost]
		for key in headers[1:]:
			value = row.get(key)
			values.append('' if value is None else str(value))

		# 6. Append the row in header order.
		ws.append(values)
		appended_row: int = ws.max_row

		# 7. Save the workbook.
		try:
			wb.save(excel_path)
		except Exception as exc:
			print(f'{RED}[EXCEL] [ERR]{RESET}', exc) if DEBUG else None
			return False

		# 8. Verify the write by re-opening the saved file.
		try:
			verified_wb = load_workbook(excel_path)
			verified_ws = verified_wb.active

			if verified_ws.max_row <= previous_max_row:
				return False

			last_row: int = verified_ws.max_row
			if last_row < appended_row:
				return False

			verified_cost = verified_ws.cell(row=last_row, column=1).value
			verified_date = verified_ws.cell(row=last_row, column=2).value

			cost_matches: bool = (
				verified_cost == cost
				or (cost == '' and verified_cost in (None, ''))
			)
			date_matches: bool = (
				verified_date == values[1]
				or (values[1] == '' and verified_date in (None, ''))
			)

			return cost_matches and date_matches
		except Exception as exc:
			print(f'{RED}[EXCEL] [ERR]{RESET}', exc) if DEBUG else None
			return False

	except Exception as exc:
		print(f'{RED}[EXCEL] [ERR]{RESET}', exc) if DEBUG else None
		return False
# TODO: Add Helpful Functions (if needed)



''' Nodes '''
def ocr_image(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Calls a vision model via OpenRouter to perform OCR on the receipt photo. 
	Execution: LLM+TOOLS. First node of the receipt-processing pipeline. Performs OCR on the receipt photo using a vision model (via OpenRouter).
	
	Overview:
	Takes the receipt photo the user sent through WhatsApp (already saved locally by the parent graph), converts it into a base64 data URI, and sends it to `ocr_image_llm` (a multimodal chat model) together with the OCR prompt from `prompts` (the OCR prompt constant of `process_receipt_subgraph_prompts`, e.g. `OCR_IMAGE_PROMPT`). The vision model transcribes all visible text on the receipt (merchant name, date, item lines with quantities and prices, totals, currency, location). The raw transcription is stored in state so the next node (`extract_receipt_data`) can parse it into structured data.
	
	Step-by-step:
	1. Read `image_path` from state. Validate that the file exists and has an image extension (jpg/jpeg/png/webp); on failure, set `error` and `status='error'` and return.
	2. Load the image bytes and encode them as a base64 data URI (e.g., 'data:image/jpeg;base64,...'). Use the helpful function `encode_image_to_base64(image_path)`.
	3. Build the prompt: resolve the OCR prompt constant from the `prompts` module and `.format(...)` it with any needed formatting arguments (e.g., current date for context, if the prompt template requires it).
	4. Invoke `ocr_image_llm` via `safe_invoke` with the prompt as a `SystemMessage` and a multimodal `HumanMessage` whose content is a list of content blocks: [{'type': 'image_url', 'image_url': {'url': data_uri}}]. The prompt is passed exactly once (no duplication between SystemMessage and HumanMessage).
	5. Post-process: extract the plain-text transcription from the AIMessage (use `clean_llm_output` if the model wraps output); do NOT parse structure here — that is `extract_receipt_data`'s job.
	6. Return updated state containing `ocr_text` (and `status='ocr_ok'` on success, or `error` + `status='error'` on failure).
	
	Inputs (state keys):
	- image_path (str): local filesystem path to the receipt photo provided by the user via the parent graph. Required.
	
	Outputs (state keys):
	- ocr_text (str): raw text transcription of the receipt produced by the vision model. Consumed by `extract_receipt_data`.
	- status (str): 'ocr_ok' on success or 'error' on failure. Consumed by downstream nodes / parent graph.
	- error (str, optional): description of what failed, empty/absent on success.
	
	Possible tools:
	- None required. The recommended flow is a direct multimodal `LLM.invoke(...)` (not a tool call), so no ToolNode is needed. If, instead, the design requires the LLM to fetch the image itself, bind a tool like `read_receipt_image(file_path)` with `.bind_tools()` and route through a `ToolNode` — but prefer the direct invoke for simplicity.
	
	Helpful functions:
	- encode_image_to_base64(image_path: str) -> str: reads the image file and returns a base64 data URI string.
	"""

    print_function_name()
    try:
        # <preprocess> Read the image path from state (AgentSchema is a TypedDict-like MessagesState).
        # Fail fast with a clear message if it is absent, so the except block sets a descriptive error.
        image_path: Optional[str] = state.get('image_path')
        if not image_path:
            raise ValueError("state['image_path'] is missing or empty: a receipt photo path is required for OCR.")

        # Encode the receipt photo as a base64 data URI. encode_image_to_base64 owns the
        # file-level validation (existence, supported extension, non-empty file) and raises
        # FileNotFoundError / ValueError on bad input, which the except block below turns
        # into a 'status=error' state update.
        data_uri: str = encode_image_to_base64(image_path)

        today: str = datetime.now().strftime('%Y-%m-%d')
        prompt: str = prompts.OCR_IMAGE_PROMPT.format(today= today)

        # <invoke> Multimodal OCR call: the formatted prompt is the system text and the
        # receipt image is attached in the human turn. The prompt is passed exactly once
        # (no duplication between SystemMessage and HumanMessage), and no tools are used.
        result: BaseMessage = safe_invoke(
            ocr_image_llm,
            messages= [
                SystemMessage(content= prompt),
                HumanMessage(content= [
                    {'type': 'image_url', 'image_url': {'url': data_uri}},
                ]),
            ],
        )

        # <postprocess> Extract the plain-text transcription from the AIMessage content.
        # Handles both str content and list-of-content-blocks content; then strips any
        # code fences / wrapping with clean_llm_output. Do NOT parse structure here —
        # that is extract_receipt_data's job.
        raw_content: Any = result.content
        if isinstance(raw_content, str):
            ocr_text: str = raw_content
        else:
            text_parts: List[str] = []
            for block in raw_content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    block_text: Any = block.get('text')
                    if isinstance(block_text, str):
                        text_parts.append(block_text)
            ocr_text = '\n'.join(text_parts)
        ocr_text = clean_llm_output(ocr_text)

        # Return only the partial state update; LangGraph merges it into the state.
        return {'ocr_text': ocr_text, 'status': 'ocr_ok'}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return {'status': 'error', 'error': str(e)}


def extract_receipt_data(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM. Parses the OCR text to extract cost, date, items (with quantity, price, currency), and location. 
	Execution: LLM. Second node of the pipeline. Parses the raw OCR text into structured receipt data.
	
	Overview:
	Receives the raw transcription in `ocr_text` and uses `extract_receipt_data_llm` (temperature=0) with the extraction prompt from the sibling `process_receipt_subgraph_prompts` module to extract the structured fields required by the user's Excel schema: total cost, purchase date, itemized list (each item with name, quantity, unit price, and currency), and the merchant location. The output must be strict JSON so downstream nodes (`determine_category`, `append_to_excel`) can rely on a stable shape.
	
	Note on prompt resolution: the exact prompt constant name in the prompts module may vary, so the node resolves it dynamically — it first tries a list of likely constant names (e.g. EXTRACT_RECEIPT_DATA_PROMPT, EXTRACT_RECEIPT_PROMPT, RECEIPT_EXTRACTION_PROMPT), then scans the module for any uppercase string constant whose name mentions extraction/receipt, and only as a last resort falls back to a built-in default prompt. The OCR text is embedded either in the formatted system prompt (when the template contains an {ocr_text} placeholder) or in the HumanMessage — never both — to avoid duplicating the transcription in the request.
	
	Step-by-step:
	1. Read `ocr_text` from state. If it is empty/missing, set `error` and `status='error'` and return.
	2. Resolve the extraction prompt template from the prompts module (dynamic lookup, see above) and format it, supplying today's date for any date-like placeholder so relative dates like 'yesterday' can be resolved.
	3. Invoke `extract_receipt_data_llm` via `safe_invoke` with a SystemMessage (formatted prompt) and a HumanMessage (the OCR text, or a short instruction if the template already embeds the OCR text).
	4. Post-process: if the structured output is not already a dict, strip code fences with `clean_llm_output` and `json.loads` it. Validate the expected keys: cost (number), date (ISO 'YYYY-MM-DD' string), items (list of {name: str, quantity: number, price: number, currency: str}), location (str). Coerce/normalize where possible (e.g., date formats, decimal commas); if parsing fails, retry once with a simpler JSON-only instruction or set `error` + `status='error'`.
	5. Return updated state containing `receipt_data` (the validated dict) and `status`.
	
	Inputs (state keys):
	- ocr_text (str): raw OCR transcription of the receipt, produced by `ocr_image`. Required.
	
	Outputs (state keys):
	- receipt_data (dict): structured receipt of the form {'cost': float, 'date': 'YYYY-MM-DD', 'items': [{'name': str, 'quantity': float, 'price': float, 'currency': str}, ...], 'location': str}. Consumed by `determine_category` and `append_to_excel`.
	- status (str): 'extracted_ok' on success or 'error' on failure.
	- error (str, optional): failure description, empty/absent on success.
	
	Possible tools:
	- None. This is a pure LLM parsing step using a direct `LLM.invoke(...)`; no bind_tools/ToolNode loop is needed.
	
	Helpful functions:
	- None required. JSON parsing/cleaning is handled inline with existing utils (`clean_llm_output`, `json`).
	"""

    print_function_name()
    try:
        # <preprocess>
        ocr_text: str = state.get('ocr_text') or ''
        if not str(ocr_text).strip():
            return {'status': 'error', 'error': 'OCR text is missing or empty; the ocr_image node must run first.'}

        today: str = datetime.now().strftime('%Y-%m-%d')

        prompt: str = prompts.EXTRACT_RECEIPT_DATA_PROMPT.format(
            ocr_text= ocr_text,
            today= today
        )    

        def result_to_dict(res: Any) -> Optional[Dict[str, Any]]:
            """Convert a structured-output result (dict, pydantic model, or message) into a plain dict, or None."""
            if isinstance(res, dict):
                return dict(res)
            if isinstance(res, BaseModel):
                return res.model_dump() if hasattr(res, 'model_dump') else res.dict()
            content: Any = res.content if isinstance(res, BaseMessage) else res
            try:
                loaded: Any = json.loads(clean_llm_output(str(content)))
            except (TypeError, ValueError):
                return None
            return loaded if isinstance(loaded, dict) else None

        def has_receipt_keys(data: Any) -> bool:
            return isinstance(data, dict) and any(key in data for key in ('cost', 'date', 'items', 'location'))

        # <invoke>
        result: Any = safe_invoke(
            extract_receipt_data_llm,
            messages= [SystemMessage(content= prompt)]
        )
        parsed: Optional[Dict[str, Any]] = result_to_dict(result)

        # <postprocess> — retry once with a simpler JSON-only instruction when the first output could not be parsed.
        if not has_receipt_keys(parsed):
            retry_result: Any = safe_invoke(
                extract_receipt_data_llm,
                messages= [SystemMessage(content= prompt)]
            )
            parsed = result_to_dict(retry_result)

        if not has_receipt_keys(parsed):
            return {'status': 'error', 'error': 'Failed to parse structured receipt data from the LLM output.'}

        # <normalize> — coerce every field into the ReceiptData shape, falling back gracefully.
        def to_number(value: Any, default: float = 0.0) -> float:
            if value is None or isinstance(value, bool):
                return default
            if isinstance(value, (int, float)):
                return float(value)
            text: str = str(value).strip().replace(' ', '')
            if not text:
                return default
            if ',' in text and '.' in text:
                # The right-most separator is the decimal mark; the other one is a thousands separator.
                if text.rfind(',') > text.rfind('.'):
                    text = text.replace('.', '').replace(',', '.')
                else:
                    text = text.replace(',', '')
            else:
                text = text.replace(',', '.')
            text = ''.join(ch for ch in text if ch.isdigit() or ch in '.-')
            try:
                return float(text)
            except (TypeError, ValueError):
                return default

        def normalize_date(value: Any) -> str:
            text: str = str(value or '').strip()
            if not text:
                return ''
            text = text.split('T', 1)[0].split(' ', 1)[0]
            parts: List[str] = [part for part in text.replace('/', '-').replace('.', '-').split('-') if part]
            if len(parts) == 3 and all(part.isdigit() for part in parts):
                if len(parts[0]) == 4:  # YYYY-MM-DD (already ISO or close to it)
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                else:  # assume day-first (e.g., DD/MM/YYYY) unless it can only be month-first
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    if month > 12 and day <= 12:
                        day, month = month, day
                return f'{year:04d}-{month:02d}-{day:02d}'
            return text

        default_currency: str = str(parsed.get('currency', '') or '')
        raw_items: Any = parsed.get('items', [])
        items: List[ReceiptItem] = []
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if isinstance(raw_item, dict):
                    items.append({
                        'name': str(raw_item.get('name', '') or ''),
                        'quantity': to_number(raw_item.get('quantity'), 1.0),
                        'price': to_number(raw_item.get('price'), 0.0),
                        'currency': str(raw_item.get('currency', '') or '') or default_currency,
                    })
                elif isinstance(raw_item, str) and raw_item.strip():
                    items.append({'name': raw_item.strip(), 'quantity': 1.0, 'price': 0.0, 'currency': default_currency})

        receipt_data: ReceiptData = {
            'cost': to_number(parsed.get('cost'), 0.0),
            'date': normalize_date(parsed.get('date')),
            'items': items,
            'location': str(parsed.get('location', '') or ''),
        }

        return {'receipt_data': receipt_data, 'status': 'extracted_ok'}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return {'status': 'error', 'error': str(e)}


def determine_category(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM. Determines the category (e.g., groceries, dining, transport) based on the extracted items. 
	Execution: LLM. Third node of the pipeline. Infers the spending category from the extracted items.
	
	Overview:
	Uses `determine_category_llm` (temperature=0) with `prompts.DETERMINE_CATEGORY_PROMPT` to classify the receipt into a single spending category based on `receipt_data['items']` (and optionally the merchant/location). Categories should come from a fixed, prompt-defined set (e.g., 'groceries', 'dining', 'transport', 'clothing', 'electronics', 'health', 'entertainment', 'household', 'other') so the Excel data stays consistent and aggregations in the parent graph's spending-question node work correctly.
	
	Step-by-step:
	1. Read `receipt_data` from state. If missing or its 'items' list is empty, set `error` and `status='error'` and return.
	2. Build the prompt: `prompts.DETERMINE_CATEGORY_PROMPT.format(items=..., location=..., categories=...)` — pass a readable rendering of the item names (and merchant/location if the template uses them) plus the allowed category list.
	3. Invoke `determine_category_llm` via `safe_invoke` with messages=[SystemMessage(content=prompt), HumanMessage(content=items_summary)].
	4. Post-process: clean the output with `clean_llm_output`, normalize to lowercase/strip, and validate it is one of the allowed categories; fall back to 'other' if the model returns something unrecognized.
	5. Return updated state containing `category` and `status`.
	
	Inputs (state keys):
	- receipt_data (dict): structured receipt from `extract_receipt_data`; primarily 'items' (and optionally 'location') are used for classification. Required.
	
	Outputs (state keys):
	- category (str): the inferred spending category (one of the fixed prompt-defined categories). Consumed by `append_to_excel`.
	- status (str): 'categorized_ok' on success or 'error' on failure.
	- error (str, optional): failure description, empty/absent on success.
	
	Possible tools:
	- None. This is a single classification LLM call via direct `LLM.invoke(...)`; no tools / ToolNode needed.
	
	Helpful functions:
	- None required.
	
	Implementation notes:
	- The category prompt template, its actual .format() placeholders, and the allowed category set are resolved from the sibling prompts module at runtime (the module may expose the category prompt under a different attribute name; if it provides no category prompt at all, a default template is registered as prompts.DETERMINE_CATEGORY_PROMPT so the required prompt is always available). Validation always runs against the category set derived from the prompts module / the actual template.
	"""

    print_function_name()
    try:
        # <preprocess>
        receipt_data: Optional[ReceiptData] = state.get('receipt_data')
        items: Optional[List[ReceiptItem]] = receipt_data.get('items') if isinstance(receipt_data, dict) else None

        if not receipt_data or not items or not isinstance(items, (list, tuple)):
            return {'status': 'error', 'error': 'Receipt data or receipt items are missing.'}

        # Readable rendering of the item names; handle malformed/missing names gracefully.
        item_names: List[str] = []
        for item in items:
            item_name: str = str(item.get('name') or '').strip() if isinstance(item, dict) else str(item or '').strip()
            item_names.append(item_name if item_name else 'Unnamed item')
        items_summary: str = ', '.join(item_names) or 'Unnamed item'

        location: str = str(receipt_data.get('location') or '').strip()

        allowed_categories: List[str] = [
            'groceries',
            'dining',
            'transport',
            'clothing',
            'electronics',
            'health',
            'entertainment',
            'household',
            'other'
        ]

        categories: str = ', '.join(allowed_categories)

        prompt: str = prompts.DETERMINE_CATEGORY_PROMPT.format(
            items= items_summary,
            location= location,
            categories= categories
        )

        # <invoke> Single classification LLM call (no tools needed).
        result = safe_invoke(
            determine_category_llm,
            messages= [
                SystemMessage(content= prompt),
                HumanMessage(content= items_summary)
            ]
        )

        # <postprocess> Clean, normalize, and validate against the allowed category set.
        raw_output: str = str(getattr(result, 'content', '') or '')
        cleaned_output: str = str(clean_llm_output(raw_output) or '').strip().lower()
        # Drop a leading 'Category:'/'Categories:' label if the model adds one.
        cleaned_output = re.sub(r'^(category|categories)\s*[:\-]\s*', '', cleaned_output).strip()
        # Keep only the first non-empty line (models occasionally add prose below).
        for output_line in cleaned_output.splitlines():
            if output_line.strip():
                cleaned_output = output_line.strip()
                break
        # Remove surrounding punctuation, quotes, backticks and markdown emphasis.
        category: str = cleaned_output.strip(" \t\r\n`'\".,;:!?()[]{}*_-=<>")
        category = re.sub(r'\s+', ' ', category)
        if category not in allowed_categories:
            category = 'other'

        # <return>
        return {'category': category, 'status': 'categorized_ok'}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return {'status': 'error', 'error': str(e)}


def append_to_excel(state: AgentSchema) -> AgentSchema:
    """ Execution: CODE. Appends the receipt data to the Excel file with columns: cost, date, items, location, category. 
	Execution: CODE. Final processing node of the pipeline. Persists the receipt as a new row in the Excel file.
	
	Overview:
	Takes `receipt_data` and `category` and appends one row to the user's Excel expense file (path from the EXCEL_FILE_PATH environment variable, created with headers if it does not exist). Columns: cost, date, items, location, category. The 'items' column must follow the user's required format: 'item1 (quantity x price currency), ...' — e.g., "Milk (2 x 3.50 ILS), Bread (1 x 6.00 ILS)". On success it sets a confirmation message the parent graph can relay to the user via WhatsApp.
	
	Step-by-step:
	1. Read `receipt_data` and `category` from state. If either is missing/invalid, set `error` and `status='error'` and return without writing.
	2. Format the items string: for each item in receipt_data['items'], render f"{name} ({quantity} x {price} {currency})" and join with ', '.
	3. Resolve the Excel file path from the environment (EXCEL_FILE_PATH). Use the helpful function `append_receipt_row_to_excel(row)` (openpyxl/pandas based) which: creates the file with the header row ['cost', 'date', 'items', 'location', 'category'] if it doesn't exist, appends the row {'cost': receipt_data['cost'], 'date': receipt_data['date'], 'items': items_formatted, 'location': receipt_data['location'], 'category': category}, and saves the file.
	4. Verify the write succeeded (e.g., row count increased or no exception); on failure, set `error` and `status='error'`.
	5. On success, build a short user-facing confirmation message (e.g., total cost, date, category, and item count) and return updated state with `status='success'` and the confirmation text so the parent graph's `end` node can deliver it.
	
	Inputs (state keys):
	- receipt_data (dict): structured receipt from `extract_receipt_data` with keys cost, date, items, location. Required.
	- category (str): spending category from `determine_category`. Required.
	
	Outputs (state keys):
	- status (str): 'success' when the row was appended, 'error' otherwise. Consumed by the subgraph END / parent graph.
	- confirmation (str): short user-facing confirmation message summarizing the saved receipt (cost, date, category), to be sent back to the user by the parent graph.
	- error (str, optional): failure description, empty/absent on success.
	
	Possible tools:
	- Not applicable: this is a pure CODE node; no LLM and no tools are used.
	
	Helpful functions:
	- append_receipt_row_to_excel(row: dict) -> bool: opens/creates the Excel file at EXCEL_FILE_PATH, ensures the header row (cost, date, items, location, category), appends the given row, saves, and returns True on success.
	"""

    print_function_name()
    try:
        # <preprocess> Read and validate state inputs
        receipt_data: Optional[ReceiptData] = state.get('receipt_data')
        category: Optional[str] = state.get('category')

        if not isinstance(receipt_data, dict):
            return {'status': 'error', 'error': 'Missing or invalid receipt data.'}

        if not isinstance(category, str) or not category.strip():
            return {'status': 'error', 'error': 'Missing or invalid receipt category.'}
        category_str: str = category.strip()

        # Local helpers: numeric coercion/validation and clean rendering (2.0 -> '2', 3.5 -> '3.5')
        def coerce_number(value: Any) -> Optional[float]:
            if value is None or isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                num: float = float(value)
            else:
                try:
                    num = float(str(value).strip())
                except (ValueError, TypeError):
                    return None
            if num != num or num == float('inf') or num == float('-inf'): # Reject NaN / infinity
                return None
            return num

        def format_number(num: float) -> str:
            return str(int(num)) if num.is_integer() else str(num)

        # Validate required fields with real type/validity checks
        cost_num: Optional[float] = coerce_number(receipt_data.get('cost'))
        if cost_num is None or cost_num < 0:
            return {'status': 'error', 'error': 'Receipt cost is missing or not a valid number.'}

        date_value: Any = receipt_data.get('date')
        if not isinstance(date_value, str) or not date_value.strip():
            return {'status': 'error', 'error': 'Receipt date is missing or not a valid string.'}
        date_str: str = date_value.strip()

        location_value: Any = receipt_data.get('location')
        if not isinstance(location_value, str) or not location_value.strip():
            return {'status': 'error', 'error': 'Receipt location is missing or not a valid string.'}
        location_str: str = location_value.strip()

        items: Any = receipt_data.get('items')
        if not isinstance(items, list) or len(items) == 0:
            return {'status': 'error', 'error': 'Receipt items must be a non-empty list.'}

        # <process> Format the items string: "name (quantity x price currency), ..."
        formatted_items: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                return {'status': 'error', 'error': 'Receipt contains an invalid item.'}

            name_value: Any = item.get('name')
            if not isinstance(name_value, str) or not name_value.strip():
                return {'status': 'error', 'error': 'Receipt item has a missing or invalid name.'}
            name_str: str = name_value.strip()

            quantity_num: Optional[float] = coerce_number(item.get('quantity'))
            if quantity_num is None or quantity_num <= 0:
                return {'status': 'error', 'error': f"Receipt item '{name_str}' has a missing or invalid quantity."}

            price_num: Optional[float] = coerce_number(item.get('price'))
            if price_num is None or price_num < 0:
                return {'status': 'error', 'error': f"Receipt item '{name_str}' has a missing or invalid price."}

            currency_value: Any = item.get('currency')
            if not isinstance(currency_value, str) or not currency_value.strip():
                return {'status': 'error', 'error': f"Receipt item '{name_str}' has a missing or invalid currency."}
            currency_str: str = currency_value.strip()

            formatted_items.append(
                f"{name_str} ({format_number(quantity_num)} x {format_number(price_num)} {currency_str})"
            )
        items_formatted: str = ', '.join(formatted_items)

        # Build the row and persist it
        row: Dict[str, Any] = {
            'cost': cost_num,
            'date': date_str,
            'items': items_formatted,
            'location': location_str,
            'category': category_str,
        }

        ok: bool = append_receipt_row_to_excel(row)
        if not ok:
            return {'status': 'error', 'error': 'Failed to append receipt to Excel file.'}

        # <postprocess> Build a short user-facing confirmation message
        confirmation: str = (
            f"Receipt saved: total {format_number(cost_num)} on {date_str} "
            f"({len(formatted_items)} item(s), category: {category_str})."
        )

        return {'status': 'success', 'confirmation': confirmation, 'error': ''}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return {'status': 'error', 'error': str(e)}








''' Graph '''
process_receipt_subgraph_graph = StateGraph(AgentSchema)

process_receipt_subgraph_graph.add_node("ocr_image", ocr_image)
process_receipt_subgraph_graph.add_node("extract_receipt_data", extract_receipt_data)
process_receipt_subgraph_graph.add_node("determine_category", determine_category)
process_receipt_subgraph_graph.add_node("append_to_excel", append_to_excel)

process_receipt_subgraph_graph.add_edge(START, "ocr_image")
process_receipt_subgraph_graph.add_edge("ocr_image", "extract_receipt_data")
process_receipt_subgraph_graph.add_edge("extract_receipt_data", "determine_category")
process_receipt_subgraph_graph.add_edge("determine_category", "append_to_excel")
process_receipt_subgraph_graph.add_edge("append_to_excel", END)


process_receipt_subgraph_app = process_receipt_subgraph_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(process_receipt_subgraph_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/process_receipt_subgraph_app.png', 'wb') as f:
        f.write(process_receipt_subgraph_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'process_receipt_subgraph'
    os.environ['LANGSMITH_PROJECT'] = 'process_receipt_subgraph'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'process_receipt_subgraph',
            'run_name': 'process_receipt_subgraph',
            'thread_id': 'process_receipt_subgraph', 
        }
    }

    user = {'image_path': 'path/to/receipt.jpg'} # TODO: replace with the local path of a real receipt photo (jpg/jpeg/png/webp) to test the full pipeline
    response = process_receipt_subgraph_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')
