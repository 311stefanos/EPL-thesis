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
from creations.whatsapp_receipt_processor_and_query_bot import whatsapp_receipt_processor_and_query_bot_prompts as prompts

import pandas as pd
import requests

import re
from datetime import datetime
from paddleocr import PaddleOCR



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} whatsapp_receipt_processor_and_query_bot') if DEBUG else None



""" Schemas """

class ReceiptData(TypedDict):
	"""
	Structured representation of a parsed receipt. Contains the fields required for storage in the Excel file and for category assignment. Used as the value type for ocr_result and Excel rows.
	"""
	cost_eur: float # Total cost converted to EUR.
	date: str # Transaction date in YYYY-MM-DD format.
	items: str # Line items formatted as 'item1 (quantity x price currency), item2 (quantity x price currency), ...'.
	location: str # Merchant/store location extracted from the receipt.
	category: Literal['Groceries', 'Dining', 'Transport', 'Utilities', 'Entertainment', 'Shopping', 'Health', 'Other'] # AutoÃ¢â‚¬â€˜assigned expense category from the fixed taxonomy.




class AgentSchema(MessagesState):
	"""
	Main state schema for the WhatsApp receipt processor and query bot. Extends MessagesState to include conversation messages and additional keys needed for receipt processing and query handling. Used as the state type for the LangGraph workflow. Contains keys for messages, optional image path, parsed OCR result, error messages, workflow mode, and next expected user action.
	"""
	image_path: Optional[str] # Optional path to a receipt image when the image is not directly attached to the message.
	ocr_result: Optional[ReceiptData] # Parsed receipt data from OCR processing; populated by the ocr_processor node.
	error_message: Optional[str] # Error message from OCR or validation failures; used to inform the user.
	mode: Optional[str] # Current workflow mode (e.g., 'receipt_ingestion', 'query_handling') used for routing.
	next_action: Optional[str] # Expected next user action (e.g., 'confirm', 'provide more details') used to resume the conversation.




''' Tools '''
@tool
def excel_read_tool(file_path: str = "./receipts.xlsx") -> List[Dict[str, Any]]:
	"""
	Overview: Reads all receipt rows from the local Excel file `./receipts.xlsx` and returns them as a list of dictionaries, each representing a receipt with keys cost_eur, date, items, location, category.
	
	Caller LLM: The chat node uses this tool to fetch data for answering natural‑language spending queries.
	
	Outside-the-Tool Work (Tool Handler Function Responsibilities): The caller (chat node) will use the returned rows to parse the user's query and decide on aggregation. No state updates are performed here.
	
	Inside-the-Tool Work (Tool Responsibilities): The tool opens the Excel file using pandas, reads the sheet, converts each row to a dict, and returns the list. Handles missing file by returning an empty list.
	
	Instructions:
	1. Determine the file path (default "./receipts.xlsx").
	2. Use pandas.read_excel to load the sheet.
	3. Convert the DataFrame to a list of dictionaries with keys: cost_eur, date, items, location, category.
	4. Return the list.
	
	State Updates (on the caller function): None.
	
	Args:
	- file_path (str, optional): Path to the Excel file. Defaults to "./receipts.xlsx".
	
	Returns:
	- List[Dict[str, Any]]: List of receipt dictionaries.
	"""
	# Validate file path to prevent path traversal attacks
	if ".." in file_path or not file_path.endswith(".xlsx"):
		return []

	try:
		df: pd.DataFrame = pd.read_excel(file_path)
		# Normalize column names to match the expected schema keys
		expected_keys: set = {"cost_eur", "date", "items", "location", "category"}
		column_mapping: Dict[str, str] = {}
		for col in df.columns:
			normalized: str = str(col).strip().lower().replace(" ", "_").replace("-", "_")
			if normalized in expected_keys:
				column_mapping[col] = normalized
		df = df.rename(columns=column_mapping)
		# Select only the expected columns that exist in the file
		existing_keys: List[str] = [k for k in expected_keys if k in df.columns]
		df = df[existing_keys]
		rows: List[Dict[str, Any]] = df.to_dict('records')
		return rows
	except FileNotFoundError:
		return []
	except Exception as e:
		print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
		return []

@tool
def excel_write_tool(receipt_data: Dict[str, Any], file_path: str = "./receipts.xlsx") -> bool:
    """
    Overview: Appends a new receipt record to the Excel file `./receipts.xlsx`. If the file does not exist, it creates it with the appropriate headers.
    
    Caller LLM: The chat node calls this tool when a receipt has been parsed and validated.
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): The caller will generate a confirmation message for the user after the tool returns success. No state updates are performed here.
    
    Inside-the-Tool Work (Tool Responsibilities): The tool reads the existing file (if any), appends the new row, and writes back using pandas. Handles file creation if missing.
    
    Instructions:
    1. Determine the file path (default "./receipts.xlsx").
    2. Load existing data into a DataFrame (if file exists).
    3. Create a new DataFrame row from the receipt_data dictionary.
    4. Concatenate and write to Excel with headers.
    
    State Updates (on the caller function): None.
    
    Args:
    - receipt_data (Dict[str, Any]): Dictionary containing cost_eur, date, items, location, category.
    - file_path (str, optional): Path to the Excel file. Defaults to "./receipts.xlsx".
    
    Returns:
    - bool: True if write succeeded.
    """
    try:
        # Validate receipt_data has all required keys
        required_keys: List[str] = ["cost_eur", "date", "items", "location", "category"]
        for key in required_keys:
            if key not in receipt_data:
                print(f'{RED}[NODE] [ERR]{RESET} Missing required key in receipt_data: {key}') if DEBUG else None
                return False
        
        # Handle empty file_path
        if not file_path:
            file_path = "./receipts.xlsx"
        
        # Path traversal protection: ensure resolved path is within working directory
        resolved_path: Path = Path(file_path).resolve()
        if not resolved_path.is_relative_to(Path.cwd()):
            print(f'{RED}[NODE] [ERR]{RESET} File path is outside the working directory: {file_path}') if DEBUG else None
            return False
        
        # Load existing data or create empty DataFrame with headers
        if os.path.exists(file_path):
            existing_df: pd.DataFrame = pd.read_excel(file_path)
        else:
            existing_df = pd.DataFrame(columns=required_keys)
        
        # Create new row DataFrame from receipt_data
        new_row_df: pd.DataFrame = pd.DataFrame([receipt_data])
        
        # Concatenate and write to Excel with headers
        combined_df: pd.DataFrame = pd.concat([existing_df, new_row_df], ignore_index=True)
        combined_df.to_excel(file_path, index=False)
        return True
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return False

@tool
def filter_excel_rows_tool(category: Optional[str] = None, date_from: Optional[str] = None, date_to: Optional[str] = None, file_path: Optional[str] = None, rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Overview: Filters a list of receipt rows based on specified criteria such as category, date range, etc. If rows are not provided, the tool reads them from the Excel file.

    Caller LLM: The chat node uses this tool to narrow down receipts for aggregation.

    Outside-the-Tool Work (Tool Handler Function Responsibilities): The caller will use the filtered rows to compute aggregates. No state updates are performed here.

    Inside-the-Tool Work (Tool Responsibilities): The tool accepts optional rows list; if not provided, it reads from Excel. It then applies filters (category, date_from, date_to) and returns filtered rows.

    Instructions:
    1. If rows not provided, call excel_read_tool to get all rows.
    2. For each row, check if it matches all provided filter criteria (e.g., row['category'] == category, row['date'] >= date_from, row['date'] <= date_to).
    3. Return filtered list.

    State Updates (on the caller function): None.

    Args:
    - category (str, optional): Filter by category.
    - date_from (str, optional): Filter rows with date >= this (YYYY-MM-DD).
    - date_to (str, optional): Filter rows with date <= this.
    - file_path (str, optional): Path to Excel file. Defaults to "./receipts.xlsx".
    - rows (List[Dict[str, Any]], optional): Pre-loaded rows to filter (if not provided, tool loads from file).

    Returns:
    - List[Dict[str, Any]]: Filtered rows.
    """
    if rows is None:
        try:
            rows = excel_read_tool.invoke({"file_path": file_path or "./receipts.xlsx"})
        except Exception:
            return []
    filtered = []
    for row in rows:
        match = True
        if category is not None and row.get("category", "").lower() != category.lower():
            match = False
        if date_from is not None and row.get("date", "") < date_from:
            match = False
        if date_to is not None and row.get("date", "") > date_to:
            match = False
        if match:
            filtered.append(row)
    return filtered

@tool
def exchange_rate_api_tool(amount: float, from_currency: str, to_currency: str = "EUR") -> float:
	"""
	Overview: Converts a monetary amount from a source currency to a target currency (default EUR) using a free exchange rate API (e.g., exchangerate-api.com). Returns the converted amount.
	
	Caller LLM: The chat node may call this tool during receipt ingestion when the original currency is not EUR.
	
	Outside-the-Tool Work (Tool Handler Function Responsibilities): The caller will use the converted amount to populate the receipt's cost_eur field. No state updates are performed here.
	
	Inside-the-Tool Work (Tool Responsibilities): The tool constructs a request to the exchange rate API, retrieves the latest rates, performs conversion, and returns the result.
	
	Instructions:
	1. Validate amount and from_currency.
	2. Fetch exchange rates from the API (e.g., GET https://api.exchangerate-api.com/v4/latest/{from_currency}).
	3. Extract rate for target currency (default EUR).
	4. Compute converted_amount = amount * rate.
	5. Return converted_amount.
	
	State Updates (on the caller function): None.
	
	Args:
	- amount (float): Amount in source currency.
	- from_currency (str): Source currency code (e.g., "USD").
	- to_currency (str, optional): Target currency code. Defaults to "EUR".
	
	Returns:
	- float: Converted amount in target currency.
	"""
	try:
		if amount <= 0:
			return 0.0
		if not from_currency:
			return 0.0
		if not to_currency:
			return 0.0
		url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
		response = requests.get(url, timeout=10)
		response.raise_for_status()
		data = response.json()
		rate = data["rates"].get(to_currency.upper())
		if rate is None:
			return 0.0
		converted_amount = amount * rate
		return converted_amount
	except Exception:
		return 0.0
# TODO: Add Tools (if needed)



''' LLM '''
chat_llm = myChatOpenAI(
	temperature= 0.4
).bind_tools([excel_read_tool, excel_write_tool, filter_excel_rows_tool, exchange_rate_api_tool])



''' Helpful Functions '''

# TODO: Add Helpful Functions (if needed)



''' Nodes '''
def ocr_processor(state: AgentSchema) -> AgentSchema:
    """ Execution: CODE. Process an image attachment using PaddleOCR tool to extract raw text. Parse OCR output to identify total cost, date, line items, merchant/location, auto-assign a category from the fixed taxonomy (Groceries, Dining, Transport, Utilities, Entertainment, Shopping, Health, Other), convert cost to EUR via free exchange rate API if needed, and prepare structured data for storage. Returns extracted fields or error. """
    print_function_name()
    try:
        # Step 1: Determine the receipt source
        messages: list[BaseMessage] = state.get("messages", [])
        image_path: str | None = None

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                content = msg.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "image_url":
                            image_path = block.get("image_url", {}).get("url")
                            break
                elif isinstance(content, str):
                    image_path = content
                break

        if not image_path:
            image_path = state.get("image_path")

        if not image_path:
            state["error_message"] = "No image found in message or state"
            return state

        # Step 2: Extract raw text using PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(image_path, cls=True)

        raw_text: str = "\n".join([line[1] for page in result for line in page if line])

        if not raw_text.strip():
            state["error_message"] = "OCR returned no text"
            return state

        # Step 3: Parse the raw OCR text
        # Extract total cost with currency symbol captured near the total
        total_match = re.search(
            r'(?:Total|Amount|TOTAL)[:\s]*([€$£])?\s*([\d,]+\.?\d*)',
            raw_text, re.IGNORECASE
        )
        cost: float = float(total_match.group(2).replace(',', '')) if total_match else 0.0
        currency_symbol: str = total_match.group(1) if total_match and total_match.group(1) else ""

        # Determine original currency from symbol near total
        if currency_symbol == '$':
            orig_currency: str = "USD"
        elif currency_symbol == '£':
            orig_currency: str = "GBP"
        elif currency_symbol == '€':
            orig_currency: str = "EUR"
        else:
            # Fallback: check if € appears anywhere (likely EUR)
            if '€' in raw_text:
                orig_currency = "EUR"
            elif '$' in raw_text:
                orig_currency = "USD"
            elif '£' in raw_text:
                orig_currency = "GBP"
            else:
                orig_currency = "EUR"

        # Extract date with validation and normalization to YYYY-MM-DD
        date_str: str = ""
        date_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})', raw_text)
        if date_match:
            date_candidate: str = date_match.group(1)
            try:
                if '-' in date_candidate:
                    parsed_date = datetime.strptime(date_candidate, '%Y-%m-%d')
                else:
                    # Try DD/MM/YYYY first, then MM/DD/YYYY
                    try:
                        parsed_date = datetime.strptime(date_candidate, '%d/%m/%Y')
                    except ValueError:
                        parsed_date = datetime.strptime(date_candidate, '%m/%d/%Y')
                date_str = parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                state["error_message"] = f"Invalid date found in receipt: {date_candidate}"
                return state

        # Extract lines
        lines: list[str] = [l.strip() for l in raw_text.split('\n') if l.strip()]

        # Merchant/location = first meaningful line
        location: str = lines[0] if lines else "Unknown"

        # Line items = non-header/non-total/non-date lines
        items_lines: list[str] = [
            l for l in lines[1:]
            if not re.search(r'(total|amount|date)', l, re.IGNORECASE)
        ]
        items: str = " | ".join(items_lines)

        # Step 4: Auto-assign category from fixed taxonomy
        category_keywords: dict[str, list[str]] = {
            "Groceries": ["supermarket", "grocery", "market", "produce", "food"],
            "Dining": ["restaurant", "cafe", "coffee", "bakery", "food"],
            "Transport": ["taxi", "uber", "bus", "metro", "train", "fuel", "gas"],
            "Utilities": ["electric", "water", "internet", "phone"],
            "Entertainment": ["cinema", "movie", "concert", "game", "netflix"],
            "Shopping": ["store", "shop", "mall", "clothing", "amazon"],
            "Health": ["pharmacy", "hospital", "doctor", "medical"],
        }
        combined: str = (location + " " + items).lower()
        category: str = "Other"
        for cat, keywords in category_keywords.items():
            if any(kw in combined for kw in keywords):
                category = cat
                break

        # Step 5: Convert cost to EUR if original currency is not EUR
        cost_eur: float = cost
        if orig_currency != "EUR":
            cost_eur = exchange_rate_api_tool.invoke({
                "amount": cost,
                "from_currency": orig_currency,
            })

        # Step 6: Validate parsed data
        if cost_eur <= 0 or not date_str or not items.strip():
            state["error_message"] = "Validation failed: invalid parsed data"
            return state

        # Step 7: Prepare structured receipt dict
        receipt_data: ReceiptData = {
            "cost_eur": round(cost_eur, 2),
            "date": date_str,
            "items": items,
            "location": location,
            "category": category,
        }

        # Step 8: Append to Excel file
        excel_write_tool.invoke({"receipt_data": receipt_data})

        # Step 9: Return updated state with ocr_result
        state["ocr_result"] = receipt_data
        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        state["error_message"] = str(e)
        return state


def chat(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Core reactive step. Inspect the latest Human message (or OCR-processed data): if it contains an image (i.e., data from OCR), handle receipt ingestion; if plain text, interpret as a natural-language spending query, read and aggregate data from ./receipts.xlsx, and produce a concise plain-text summary. Calls tools for currency conversion, Excel I/O, and other operations as needed. Sets mode/next_action for any follow-up if required. """
    print_function_name()
    try:
        ocr_result = state.get("ocr_result")
        if ocr_result:
            # Receipt ingestion path
            cost_eur: float = ocr_result.get("cost_eur", 0)
            date: str = ocr_result.get("date", "")
            items: str = ocr_result.get("items", "")
            location: str = ocr_result.get("location", "")
            category: str = ocr_result.get("category", "")

            valid_categories = ["Groceries", "Dining", "Transport", "Utilities", "Entertainment", "Shopping", "Health", "Other"]

            if cost_eur <= 0 or not date or not items or category not in valid_categories:
                state["messages"].append(AIMessage(content="Error: Invalid receipt data. Missing or invalid fields."))
                return state

            success = excel_write_tool.invoke({"receipt_data": ocr_result})
            if success:
                confirmation = f"Receipt added. Total: {cost_eur} EUR in {category}."
            else:
                confirmation = "Failed to save receipt. Please try again."

            state["messages"].append(AIMessage(content=confirmation))
            state["mode"] = "receipt_ingestion"
            state["next_action"] = "confirm"
            return state

        # Plain-text query path
        messages: list[BaseMessage] = state.get("messages", [])
        query: str = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            state["messages"].append(AIMessage(content="I didn't understand your message. Could you please rephrase?"))
            return state

        system_prompt = prompts.CHAT_PROMPT.format(user_query= query)
        result = safe_invoke(
            chat_llm,
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
        )

        if result is None:
            result = AIMessage(content="I'm sorry, I couldn't process your request.")

        state["messages"].append(result)
        state["mode"] = "query_handling"
        state["next_action"] = None
        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        state["messages"].append(AIMessage(content="An error occurred while processing your request. Please try again."))
        return state




''' Conditional Functions '''
def from_start_to(state: AgentSchema) -> Literal["ocr_processor", "chat"]:
    """Route to ocr_processor if the latest message has image content or image_path is set; otherwise route to chat."""
    if state.get("image_path"):
        return "ocr_processor"

    messages = state.get("messages", [])
    if not messages:
        return "chat"

    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        content = last_message.content
        if isinstance(content, str) and "image" in content.lower():
            return "ocr_processor"
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "image_url" in block:
                    return "ocr_processor"

    return "chat"


def from_chat_to(state: AgentSchema) -> Literal["chat_tools_excel", "__end__"]:
    """Route to the tool node if the last AI message contains tool calls; otherwise end."""
    print_function_name()
    messages = state["messages"]
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "chat_tools_excel"
    return END


''' Graph '''
whatsapp_receipt_processor_and_query_bot_graph = StateGraph(AgentSchema)

whatsapp_receipt_processor_and_query_bot_graph.add_node("ocr_processor", ocr_processor)
whatsapp_receipt_processor_and_query_bot_graph.add_node("chat", chat)
whatsapp_receipt_processor_and_query_bot_graph.add_node("chat_tools_excel", ToolNode([excel_read_tool, excel_write_tool, filter_excel_rows_tool, exchange_rate_api_tool]))

whatsapp_receipt_processor_and_query_bot_graph.add_conditional_edges(
    "__start__",
    from_start_to,
    {   # Not needed just for clarity
        "ocr_processor": "ocr_processor",
        "chat": "chat",
    }
)
whatsapp_receipt_processor_and_query_bot_graph.add_conditional_edges(
    "chat",
    from_chat_to,
    {
        "chat_tools_excel": "chat_tools_excel",
        END: END,
    }
)
whatsapp_receipt_processor_and_query_bot_graph.add_edge("chat_tools_excel", "chat")
whatsapp_receipt_processor_and_query_bot_graph.add_edge("ocr_processor", "chat")


whatsapp_receipt_processor_and_query_bot_app = whatsapp_receipt_processor_and_query_bot_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(whatsapp_receipt_processor_and_query_bot_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/whatsapp_receipt_processor_and_query_bot_app.png', 'wb') as f:
        f.write(whatsapp_receipt_processor_and_query_bot_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'whatsapp_receipt_processor_and_query_bot'
    os.environ['LANGSMITH_PROJECT'] = 'whatsapp_receipt_processor_and_query_bot'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'whatsapp_receipt_processor_and_query_bot',
            'run_name': 'whatsapp_receipt_processor_and_query_bot',
            'thread_id': 'whatsapp_receipt_processor_and_query_bot', 
        }
    }

    user = '' # TODO: add
    response = whatsapp_receipt_processor_and_query_bot_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')