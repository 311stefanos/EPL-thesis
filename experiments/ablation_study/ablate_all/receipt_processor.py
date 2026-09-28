''' Imports '''

import base64
import os
import sys
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from utils.utils import myChatOpenAI, safe_invoke

from experiments.ablation_study.ablate_all import receipt_processor_prompts as prompts
from experiments.ablation_study.ablate_all.receipt_storage import append_receipt_record, format_items_string

''' Constants '''

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG', 'false').strip().lower() in ('1', 'true', 'yes')

# Vision-capable model called through OpenRouter to read receipt photos.
VISION_MODEL = os.getenv('OPENROUTER_VISION_MODEL', 'openai/gpt-4o-mini')

MIME_BY_SUFFIX = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
}
SUPPORTED_IMAGE_SUFFIXES = set(MIME_BY_SUFFIX)

# Maximum accepted photo size (vision providers commonly cap around 20 MB).
MAX_IMAGE_BYTES = 20 * 1024 * 1024

ALLOWED_CATEGORIES: List[str] = [
    'Groceries',
    'Dining',
    'Transport',
    'Utilities',
    'Shopping',
    'Health',
    'Entertainment',
    'Travel',
    'Education',
    'Other',
]

''' Schemas '''

class ReceiptItem(BaseModel):
    """A single purchased line item as printed on the receipt."""
    name: str = Field(description='Clean item name without commas or parentheses.')
    quantity: float = Field(description='Number of units purchased for this line.')
    unit_price: float = Field(description='Price of a single unit in the receipt currency.')


class ReceiptData(BaseModel):
    """Structured data extracted from one receipt photo."""
    location: str = Field(description='Store or merchant name, with branch or city when visible.')
    date: str = Field(description='Receipt date in ISO format (YYYY-MM-DD).')
    currency: str = Field(description='ISO 4217 currency code (e.g. USD, EUR, ILS) or the symbol found on the receipt.')
    total_cost: float = Field(description='Final total amount paid as printed on the receipt.')
    items: List[ReceiptItem] = Field(description='All purchased line items with quantity and unit price.')


class ReceiptCategory(BaseModel):
    """The single spending category assigned to a receipt."""
    category: str = Field(description='Exactly one category from the allowed category list.')


class ReceiptProcessorSchema(MessagesState):
    # Path of the receipt photo being processed.
    image_path: Optional[str]
    # Base64 data URL of the photo, produced by the image loader.
    image_data_url: Optional[str]
    # Structured receipt payload extracted by the vision model (see ReceiptData).
    receipt_data: Optional[dict]
    # Spending category assigned to the receipt.
    category: Optional[str]
    # Path of the Excel file the receipt was appended to.
    excel_path: Optional[str]
    # High-level outcome: 'receipt_saved' | 'error'.
    status: Optional[str]
    # Operational error description; empty when everything succeeded.
    error: Optional[str]

''' LLM '''

# Vision model with structured output that reads the receipt photo.
extraction_llm = myChatOpenAI(temperature=0.0, model=VISION_MODEL).with_structured_output(ReceiptData)

# Text model with structured output that assigns the spending category.
category_llm = myChatOpenAI(temperature=0.0).with_structured_output(ReceiptCategory)

''' Helpful Functions '''

def _encode_image(image_path: str) -> str:
    """Read an image file and return it as a base64 data URL for the vision model.

    Raises ValueError with a user-safe message when the path is missing, the
    file does not exist, the type is unsupported, or the file is too large.
    """
    if not image_path:
        raise ValueError('No receipt photo path was provided.')
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise ValueError(f'Receipt photo not found: {image_path}')
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ', '.join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise ValueError(f'Unsupported photo type "{suffix}". Supported types: {supported}.')
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError('The receipt photo is too large (the limit is 20 MB).')
    encoded = base64.b64encode(data).decode('ascii')
    return f'data:{MIME_BY_SUFFIX[suffix]};base64,{encoded}'


def _normalize_category(raw_category: str) -> str:
    """Map a model-proposed category onto the allowed list, defaulting to Other."""
    candidate = (raw_category or '').strip().lower()
    for allowed in ALLOWED_CATEGORIES:
        if candidate == allowed.lower():
            return allowed
    return 'Other'

''' Nodes '''

def load_image_node(state: ReceiptProcessorSchema) -> Dict[str, Any]:
    """Validate the photo path and encode the image for the vision model."""
    try:
        data_url = _encode_image(state.get('image_path') or '')
    except ValueError as exc:
        return {'error': str(exc), 'status': 'error'}
    except OSError:
        if DEBUG:
            traceback.print_exc()
        return {'error': 'The receipt photo could not be read.', 'status': 'error'}
    return {'image_data_url': data_url, 'error': ''}


def extract_receipt_node(state: ReceiptProcessorSchema) -> Dict[str, Any]:
    """Read the receipt photo with the vision model and extract structured data."""
    data_url = state.get('image_data_url') or ''
    if not data_url:
        return {'error': 'No receipt photo was loaded, so nothing could be extracted.', 'status': 'error'}
    today = date.today().isoformat()
    year = today[:4]
    prompt = prompts.EXTRACT_RECEIPT_PROMPT.format(today=today, year=year)
    message = HumanMessage(
        content=[
            {'type': 'text', 'text': 'Extract the data from this receipt photo.'},
            {'type': 'image_url', 'image_url': {'url': data_url}},
        ]
    )
    try:
        response = safe_invoke(extraction_llm, [SystemMessage(content=prompt), message])
    except Exception:
        if DEBUG:
            traceback.print_exc()
        return {'error': 'Reading the receipt photo failed; no data could be extracted.', 'status': 'error'}
    if response is None:
        return {'error': 'Reading the receipt photo failed; no data could be extracted.', 'status': 'error'}
    return {'receipt_data': response.model_dump(), 'error': ''}


def categorize_receipt_node(state: ReceiptProcessorSchema) -> Dict[str, Any]:
    """Determine the spending category for the extracted receipt items."""
    receipt = state.get('receipt_data') or {}
    items_text = format_items_string(receipt)
    location = str(receipt.get('location') or 'Unknown')
    prompt = prompts.CATEGORIZE_RECEIPT_PROMPT.format(
        items=items_text or 'No item details were extracted.',
        location=location,
        categories=', '.join(ALLOWED_CATEGORIES),
    )
    raw_category = ''
    try:
        response = safe_invoke(
            category_llm,
            [SystemMessage(content=prompt), HumanMessage(content='Assign the spending category.')],
        )
        raw_category = getattr(response, 'category', '')
    except Exception:
        if DEBUG:
            traceback.print_exc()
    return {'category': _normalize_category(raw_category)}


def save_receipt_node(state: ReceiptProcessorSchema) -> Dict[str, Any]:
    """Append the categorized receipt as a new row in the Excel file."""
    receipt = state.get('receipt_data') or {}
    category = state.get('category') or 'Other'
    try:
        excel_path = append_receipt_record(receipt, category)
    except Exception as exc:
        if DEBUG:
            traceback.print_exc()
        return {'error': f'Failed to write the receipt to the Excel file: {exc}', 'status': 'error'}
    return {'excel_path': str(excel_path), 'status': 'receipt_saved', 'error': ''}


def compose_confirmation_node(state: ReceiptProcessorSchema) -> Dict[str, Any]:
    """Build the deterministic confirmation message shown to the user."""
    receipt = state.get('receipt_data') or {}
    category = state.get('category') or 'Other'
    total_cost = float(receipt.get('total_cost') or 0.0)
    currency = str(receipt.get('currency') or '').strip()
    lines = [
        '✅ Receipt saved!',
        f'• Location: {receipt.get("location") or "Unknown"}',
        f'• Date: {receipt.get("date") or "Unknown"}',
        f'• Total: {total_cost:.2f} {currency}'.strip(),
        f'• Category: {category}',
        f'• Items: {format_items_string(receipt) or "No item details extracted"}',
    ]
    return {'messages': [AIMessage(content='\
'.join(lines))]}


def handle_error_node(state: ReceiptProcessorSchema) -> Dict[str, Any]:
    """Mark the failure; the root graph renders the user-facing message."""
    if DEBUG and state.get('error'):
        print(f'[receipt_processor] error: {state["error"]}')
    return {'status': 'error'}

''' Conditional Functions '''

def route_after_image(state: ReceiptProcessorSchema) -> Literal['extract_receipt', 'handle_error']:
    """Skip extraction when the photo could not be loaded."""
    return 'handle_error' if state.get('error') else 'extract_receipt'


def route_after_save(state: ReceiptProcessorSchema) -> Literal['compose_confirmation', 'handle_error']:
    """Only confirm success when the row was actually written."""
    return 'handle_error' if state.get('error') else 'compose_confirmation'

''' Graph '''

receipt_processor_graph = StateGraph(ReceiptProcessorSchema)

receipt_processor_graph.add_node('load_image', load_image_node)
receipt_processor_graph.add_node('extract_receipt', extract_receipt_node)
receipt_processor_graph.add_node('categorize_receipt', categorize_receipt_node)
receipt_processor_graph.add_node('save_receipt', save_receipt_node)
receipt_processor_graph.add_node('compose_confirmation', compose_confirmation_node)
receipt_processor_graph.add_node('handle_error', handle_error_node)

receipt_processor_graph.add_edge(START, 'load_image')
receipt_processor_graph.add_conditional_edges(
    'load_image',
    route_after_image,
    {'extract_receipt': 'extract_receipt', 'handle_error': 'handle_error'},
)
receipt_processor_graph.add_edge('extract_receipt', 'categorize_receipt')
receipt_processor_graph.add_edge('categorize_receipt', 'save_receipt')
receipt_processor_graph.add_conditional_edges(
    'save_receipt',
    route_after_save,
    {'compose_confirmation': 'compose_confirmation', 'handle_error': 'handle_error'},
)
receipt_processor_graph.add_edge('compose_confirmation', END)
receipt_processor_graph.add_edge('handle_error', END)

receipt_processor_app = receipt_processor_graph.compile()

''' Testing '''

if __name__ == '__main__':
    # Process a single receipt photo passed as the first command-line argument.
    if len(sys.argv) > 1:
        result = receipt_processor_app.invoke({'image_path': sys.argv[1]})
        print('Status:', result.get('status'))
        print('Category:', result.get('category'))
        print('Excel file:', result.get('excel_path'))
        for message in result.get('messages', []):
            if isinstance(message, AIMessage):
                print(message.content)
    else:
        print('Usage: python receipt_processor.py <path_to_receipt_photo>')
