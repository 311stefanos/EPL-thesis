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
import traceback, pytesseract, io, re, requests
import json
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS
from creations.menu_recommendation_workflow import menu_recommendation_workflow_prompts as prompts

from PIL import Image
from bs4 import BeautifulSoup
from urllib.parse import urlparse

''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'

print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} menu_recommendation_workflow') if DEBUG else None

""" Schemas """
class MenuItem(BaseModel):
	"""
	Represents a single menu item with its details. Used within the parsed_menu_data to structure menu items.
	"""
	name: str # The name of the menu item.
	description: Optional[str] # A brief description of the menu item.
	price: Optional[float] # The price of the menu item.
	category: Optional[str] # The category of the menu item (e.g., 'Appetizer', 'Main Course').
	ingredients: Optional[List[str]] # List of ingredients in the menu item.

class Recommendation(BaseModel):
	"""
	Represents a single menu recommendation with an explanation. Used within the recommendations key in AgentSchema.
	"""
	item: MenuItem # The recommended menu item.
	rank: int # The rank of the recommendation (1 being the highest).
	explanation: str # Explanation for why this item is recommended.

class AgentSchema(MessagesState):
	"""
	The main schema for the menu recommendation workflow. It holds the state of the graph, including parsed menu data, user preferences, conversation messages, and other metadata. This schema is used across all nodes to maintain and update the state during the workflow execution. The `messages` key is inherited from MessagesState and contains an ordered list of conversation messages. The `mode` key is a Literal type indicating the current mode or next action expected from the workflow.
	"""
	messages: List[BaseMessage] # Ordered list of conversation messages, including user inputs and assistant responses.
	parsed_menu_data: Optional[Dict[str, Any]] # Structured menu data extracted from the user's input (photo, link, or text).
	user_preferences: Optional[Dict[str, Any]] # User-specific food and drink preferences, including dietary restrictions and allergies.
	recommendations: Optional[List[Recommendation]] # Ranked list of menu recommendations with explanations for each suggestion.
	mode: Literal["parse_menu", "chat"] # Current mode or next action expected from the workflow.

''' Tools '''
@tool
def suggest_ranked_list(recommendations: List[Recommendation]) -> str:
    """
    Overview: Generates a ranked list of menu recommendations based on parsed menu items and user preferences. This tool does not modify the state or perform any actions; it simply returns a confirmation message indicating that the ranked list has been generated.
    Caller LLM: chat
    Instructions:
    1. Accept a list of menu items as input.
    2. Return a confirmation message indicating that the ranked list of recommendations has been generated.
    Outside-the-Tool Work (Caller Responsibilities):
    1. After invocation, update the state with the generated ranked list of recommendations.
    2. Add the TollMessage to the messages key.
    3. Go to the end node.
    Inside-the-Tool Work (Tool Responsibilities):
    1. Validate that recommendations is a list and that each element is a Recommendation instance.
    2. Do not perform file I/O, network calls, or state mutation.
    3. Do not access user memory directly; rely only on the provided recommendations argument.
    4. Return a compact confirmation string only.
    Args:
    - recommendations (List[Recommendation]): List of menu recommendations extracted from the user's input.
    Returns:
    - str: A confirmation message indicating that the ranked list of recommendations has been generated.
    """
    if not isinstance(recommendations, list):
        raise ValueError("Input must be a list")
    if len(recommendations) == 0:
        raise ValueError("Recommendations list cannot be empty")
    for item in recommendations:
        if not isinstance(item, Recommendation):
            raise ValueError(f"Item {item} is not a Recommendation instance")
    return f"Successfully generated {len(recommendations)} recommendations"

@tool
def fetch_user_memory(user_id: str) -> Dict[str, Any]:
    """
    Overview: Retrieves user preferences and feedback from a structured JSON file.
    Args:
        user_id (str): The unique identifier for the user.
    Returns:
        Dict[str, Any]: User-specific data or an empty dict if not found.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        print(f'{RED}[NODE] [ERR]{RESET} Invalid user_id: {user_id}') if DEBUG else None
        return {}

    file_path = "user_memory.json"
    try:
        with open(file_path, "r") as f:
            user_data = json.load(f)
        return user_data.get(user_id, {})
    except FileNotFoundError:
        print(f'{RED}[NODE] [ERR]{RESET} File not found: {file_path}') if DEBUG else None
        return {}
    except json.JSONDecodeError:
        print(f'{RED}[NODE] [ERR]{RESET} Invalid JSON in file: {file_path}') if DEBUG else None
        return {}
    except OSError as e:
        print(f'{RED}[NODE] [ERR]{RESET} Permission error accessing file: {file_path}, Error: {e}') if DEBUG else None
        return {}

@tool
def update_user_memory(user_id: str, new_memory: Dict[str, Any]) -> str:
    """
    Overview: Updates user preferences and feedback in a structured JSON file. This tool ensures the system can adapt over time based on user feedback.
    Caller LLM: chat
    Instructions:
    1. Accept a user ID and new memory data.
    2. Update the user's preferences and feedback in the JSON file.
    3. Return a string indicating the success or failure of the update.
    Outside-the-Tool Work (Caller Responsibilities):
    None
    Inside-the-Tool Work (Tool Responsibilities):
    1. Validate that user_id is a non-empty string and new_memory is a dict with JSON-serializable values.
    2. Load the existing JSON store safely, handle missing file or invalid JSON gracefully.
    3. Update only the specified user's record and preserve unrelated users' data.
    4. Write changes atomically when possible (write temp then replace) to reduce corruption risk.
    5. Return a compact status string indicating success or failure, and include an error reason on failure.
    Args:
    - user_id (str): The unique identifier for the user.
    - new_memory (Dict[str, Any]): New user preferences or feedback to be stored.
    Returns:
    - str: A message indicating the success or failure of the update.
    """
    # Validate inputs
    if not isinstance(user_id, str) or not user_id.strip():
        return "Error: Invalid user_id"
    if not isinstance(new_memory, dict):
        return "Error: new_memory must be a dictionary"

    # Validate JSON serialization
    try:
        json.dumps(new_memory)
    except (TypeError, ValueError) as e:
        return f"Error: new_memory contains non-serializable values: {str(e)}"

    # Define file paths
    file_path = Path("user_memory.json")
    temp_file_path = Path("user_memory_temp.json")
    backup_file_path = Path("user_memory_backup.json")

    try:
        # Create a backup of the original file if it exists
        if file_path.exists():
            with open(file_path, 'r') as f:
                backup_data = json.load(f)
            with open(backup_file_path, 'w') as f:
                json.dump(backup_data, f, indent=4)

        # Read existing data
        if file_path.exists():
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Update user data
        data[user_id] = new_memory

        # Write to temp file
        with open(temp_file_path, 'w') as f:
            json.dump(data, f, indent=4)

        # Replace original file
        temp_file_path.replace(file_path)

        return "Success: User memory updated"
    except PermissionError:
        return "Error: Permission denied while accessing the file"
    except IOError as e:
        return f"Error: File I/O error: {str(e)}"
    except OSError as e:
        return f"Error: OS error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def finalize_conversation() -> str:
    """
    Overview: Transitions the state into parse mode, indicating the end of the current conversation. This tool does not modify the state; it simply returns a confirmation message.
    Caller LLM: chat
    Instructions:
    1. Return a string indicating the success of the transition.
    Outside-the-Tool Work (Caller Responsibilities):
    1. If the graph needs to transition modes (for example: set state["next_action"] or a mode flag), do that in the caller node, not inside this tool.
    2. After receiving the confirmation string, send the final user-facing message and end the run.
    Inside-the-Tool Work (Tool Responsibilities):
    1. Do not read or write state.
    2. Do not perform file I/O, network calls, or any side effects.
    3. Return a compact confirmation string only.
    Args:
    - None
    Returns:
    - str: A message indicating the success of the transition.
    """
    return "Conversation finalized."
# TODO: Add Tools (if needed)

''' LLM '''
chat_llm = myChatOpenAI(
	temperature= 0.4
).bind_tools([suggest_ranked_list, fetch_user_memory, update_user_memory, finalize_conversation])

''' Helpful Functions '''
def parse_photo_menu(photo_data: bytes) -> Dict[str, Any]:
    """
    Overview: Extracts menu data from a photo format.
    Caller Node: parse_menu
    Instructions:
    1. Accept photo data in bytes format.
    2. Use OCR (Optical Character Recognition) to extract text from the photo.
    3. Parse the extracted text to identify menu items, descriptions, prices, categories, and ingredients.
    4. Structure the parsed data into a standardized format.
    5. Return the structured menu data as a dictionary.
    Args:
    - photo_data (bytes): The photo data containing the menu.
    Returns:
    - Dict[str, Any]: Structured menu data extracted from the photo.
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(photo_data))

        # Extract text using OCR
        raw_text = pytesseract.image_to_string(image)

        # Parse text into menu items
        menu_items: List[Dict[str, Any]] = []
        lines = raw_text.split('\n')

        current_category: Optional[str] = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if the line is a category
            if re.match(r'^[A-Z][a-zA-Z\s]+:$', line) or re.match(r'^[A-Z][a-zA-Z\s]+$', line):
                current_category = line.replace(':', '').strip()
                continue

            # Check if the line contains a price
            price_match = re.search(r'\$[\d,]+\.?\d*', line)
            if price_match:
                price = float(price_match.group()[1:].replace(',', ''))
                name = re.sub(r'\$[\d,]+\.?\d*', '', line).strip()

                # Extract description and ingredients if available
                description = None
                ingredients = None
                if ' - ' in name:
                    parts = name.split(' - ', 1)
                    name = parts[0]
                    description = parts[1]

                # Attempt to extract ingredients from description
                if description and 'Ingredients:' in description:
                    ingredients_part = description.split('Ingredients:')[1].strip()
                    ingredients = [ingredient.strip() for ingredient in ingredients_part.split(',')]

                # Create a menu item
                menu_item: Dict[str, Any] = {
                    'name': name,
                    'price': price,
                    'category': current_category,
                    'description': description,
                    'ingredients': ingredients
                }
                menu_items.append(menu_item)

        return {"items": menu_items}
    except Exception as e:
        print(f"Error during OCR or parsing: {e}")
        return {"items": []}

def parse_link_menu(link_url: str) -> Dict[str, Any]:
    """
    Overview: Extracts menu data from a link format.
    Caller Node: parse_menu
    Instructions:
    1. Accept a URL pointing to a menu.
    2. Fetch the content from the URL.
    3. Parse the fetched content to identify menu items, descriptions, prices, categories, and ingredients.
    4. Structure the parsed data into a standardized format.
    5. Return the structured menu data as a dictionary.
    Args:
    - link_url (str): The URL pointing to the menu.
    Returns:
    - Dict[str, Any]: Structured menu data extracted from the link.
    """
    
    try:
        # Validate the URL
        parsed_url = urlparse(link_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL format")

        # Fetch the HTML content from the URL with headers and timeout
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(link_url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Check if the content is HTML (case-insensitive)
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            raise ValueError("URL does not return HTML content")

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize the list to store menu items
        menu_items = []

        # Extract menu items from the HTML using flexible selectors
        for item in soup.find_all(['div', 'li', 'section'], class_=lambda x: x and 'menu' in x.lower()):
            name = None
            description = None
            price = None
            category = None
            ingredients = None

            # Extract name
            name_tag = item.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'p'], class_=lambda x: x and 'name' in x.lower())
            if name_tag:
                name = name_tag.text.strip()

            # Extract description
            description_tag = item.find(['p', 'span', 'div'], class_=lambda x: x and 'description' in x.lower())
            if description_tag:
                description = description_tag.text.strip()

            # Extract price
            price_tag = item.find(['span', 'div', 'p'], class_=lambda x: x and 'price' in x.lower())
            if price_tag:
                price_text = price_tag.text.strip().replace('$', '')
                try:
                    price = float(price_text)
                except ValueError:
                    price = None

            # Extract category
            category_tag = item.find(['span', 'div', 'p'], class_=lambda x: x and 'category' in x.lower())
            if category_tag:
                category = category_tag.text.strip()

            # Extract ingredients
            ingredients_tags = item.find_all(['li', 'span', 'div'], class_=lambda x: x and 'ingredient' in x.lower())
            if ingredients_tags:
                ingredients = [tag.text.strip() for tag in ingredients_tags]

            # Validate that at least the 'name' field exists before creating a menu item
            if name:
                menu_item = {
                    'name': name,
                    'description': description,
                    'price': price,
                    'category': category,
                    'ingredients': ingredients
                }
                menu_items.append(menu_item)

        # Structure the parsed data into a standardized format
        parsed_menu_data = {
            'items': menu_items
        }

        return parsed_menu_data

    except requests.RequestException as e:
        print(f'{RED}[NODE] [ERR]{RESET} Error fetching the URL: {str(e)}') if DEBUG else None
        return {'items': []}
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET} Error parsing the menu: {str(e)}') if DEBUG else None
        return {'items': []}

def parse_text_menu(text_data: str) -> Dict[str, Any]:
    """
    Overview: Extracts menu data from a text format.
    Caller Node: parse_menu
    Instructions:
    1. Accept text data containing the menu.
    2. Parse the text to identify menu items, descriptions, prices, categories, and ingredients.
    3. Structure the parsed data into a standardized format.
    4. Return the structured menu data as a dictionary.
    Args:
    - text_data (str): The text data containing the menu.
    Returns:
    - Dict[str, Any]: Structured menu data extracted from the text.
    """
    # Initialize the menu data structure
    menu_data: Dict[str, Any] = {
        "items": []
    }

    # Define regex patterns for extracting menu items, descriptions, prices, categories, and ingredients
    item_pattern = re.compile(r'^([A-Za-z\s]+?)\s*(?:-|\|)\s*(.*?)\s*(\$\d+\.\d{2}|\d+\.\d{2})', re.MULTILINE)
    category_pattern = re.compile(r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)', re.MULTILINE)
    ingredient_pattern = re.compile(r'\((.*?)\)|\[(.*?)\]')

    # Split the text into lines for processing
    lines = text_data.split('\n')

    current_category: Optional[str] = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            # Check if the line is a category header
            category_match = category_pattern.match(line)
            if category_match:
                current_category = category_match.group(1)
                continue

            # Check if the line is a menu item
            item_match = item_pattern.match(line)
            if item_match:
                name = item_match.group(1).strip()
                description = item_match.group(2).strip()
                price_str = item_match.group(3).strip()

                # Validate and parse price
                price = None
                if price_str.startswith('$'):
                    price_str = price_str[1:]
                try:
                    price = float(price_str)
                except ValueError:
                    continue  # Skip if price is malformed

                # Extract ingredients if present
                ingredients: List[str] = []
                ingredients_match = ingredient_pattern.search(description)
                if ingredients_match:
                    ingredients_str = ingredients_match.group(1) or ingredients_match.group(2)
                    ingredients = [ing.strip() for ing in ingredients_str.split(',')]

                # Create a menu item dictionary
                menu_item: Dict[str, Any] = {
                    "name": name,
                    "description": description,
                    "price": price,
                    "category": current_category,
                    "ingredients": ingredients
                }

                # Add the menu item to the menu data
                menu_data["items"].append(menu_item)
        except Exception as e:
            # Skip malformed lines and continue processing
            continue

    return menu_data
# TODO: Add Helpful Functions (if needed)

''' Nodes '''
def parse_menu(state: AgentSchema) -> AgentSchema:
    """ Execution: CODE. Parse the menu data provided by the user, supporting photo, link, or text formats.
    Parses the menu data provided by the user, supporting photo, link, or text formats.

    This node is responsible for extracting and structuring menu data from various input formats (photo, link, or text) to prepare it for further processing in the workflow.

    Steps:
    1. Extract the menu data from the input format (photo, link, or text).
    2. Validate the extracted data to ensure it is in a usable format.
    3. Structure the data into a standardized format for further processing.
    4. Update the state with the parsed menu data.

    Inputs:
    - state: AgentSchema
      - The current state of the workflow, which may include the raw menu data provided by the user.

    Outputs:
    - state: AgentSchema
      - The updated state with the parsed menu data included.

    Possible Tools:
    - None

    Possible Helpful Functions:
    - parse_photo_menu: Extracts menu data from a photo format.
    - parse_link_menu: Extracts menu data from a link format.
    - parse_text_menu: Extracts menu data from a text format.
    """
    print_function_name()
    try:
        messages: List[BaseMessage] = state.get('messages', [])
        if not messages:
            raise ValueError("No messages in state")

        last_message = messages[-1]
        parsed_menu_data = None

        if isinstance(last_message, HumanMessage):
            content = last_message.content
            if isinstance(content, str):
                # Handle text format
                if not content.strip():
                    raise ValueError("Empty text content")
                parsed_menu_data = parse_text_menu(content)
            elif isinstance(content, dict):
                # Handle link format
                if 'url' not in content:
                    raise ValueError("Dictionary content does not contain 'url' key")
                parsed_menu_data = parse_link_menu(content['url'])
            elif isinstance(content, bytes):
                # Handle photo format
                if not content:
                    raise ValueError("Empty photo content")
                parsed_menu_data = parse_photo_menu(content)
            else:
                raise ValueError("Unsupported message content format")
        else:
            raise ValueError("Last message is not a HumanMessage")

        # Validate parsed menu data structure
        if not parsed_menu_data or 'items' not in parsed_menu_data:
            raise ValueError("Invalid parsed menu data format")
        if not isinstance(parsed_menu_data['items'], list):
            raise ValueError("'items' must be a list")
        for item in parsed_menu_data['items']:
            if not isinstance(item, dict) or 'name' not in item:
                raise ValueError("Each menu item must be a dictionary with a 'name' field")

        # Update the state
        state['parsed_menu_data'] = parsed_menu_data
        state['mode'] = 'chat'

        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET} Error in parse_menu: {str(e)}') if DEBUG else None
        state['error'] = f"Error in parse_menu: {str(e)}"
        return state

def chat(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Process the user message, retrieve user preferences using user_preferences_manager, generate recommendations, and handle feedback using feedback_processor.
	Processes the user message, retrieves user preferences, generates recommendations, and handles feedback.

	This node interacts with the user, retrieves their preferences, and generates a ranked list of menu recommendations based on the parsed menu data and user feedback. The LLM can either respond in natural language or call tools to fetch or update user preferences, generate recommendations, or finalize the conversation.

	Steps:
	1. Format the prompt using the user's preferences, parsed menu data, and previous messages.
	2. Invoke the LLM with the formatted prompt.
	3. Based on the LLM's response, either:
	   - Append a natural language response to the state for the user.
	   - Call a tool to generate recommendations, fetch or update user preferences, or finalize the conversation.

	Inputs:
	- state: AgentSchema
	  - The current state of the workflow, which includes the parsed menu data, user preferences, and any previous feedback.
	  - messages: Ordered list of conversation messages, including user inputs and assistant responses.

	Outputs:
	- state: AgentSchema
	  - The updated state with the generated recommendations, user feedback, or natural language response included.
	  - latest: The latest response or recommendation to be sent to the user.

	Possible Tools:
	- suggest_ranked_list: Generates a ranked list of menu recommendations.
	- fetch_user_memory: Retrieves user preferences and feedback.
	- update_user_memory: Updates user preferences and feedback.
	- finalize_conversation: Transitions the state into parse mode.

	Possible Helpful Functions:
	- None
	"""
    print_function_name()
    try:
        # Extract user preferences, parsed menu data, and messages from the state
        user_preferences: Dict[str, Any] = state.get('user_preferences', {})
        parsed_menu_data: Dict[str, Any] = state.get('parsed_menu_data', {})
        messages: List[BaseMessage] = state.get('messages', [])

        # Format the prompt using the CHAT_PROMPT template
        prompt: str = prompts.CHAT_PROMPT.format(
            user_preferences=user_preferences,
            parsed_menu_data=parsed_menu_data,
            messages=messages
        )

        # Invoke the LLM with the formatted prompt and previous messages
        result: BaseMessage = safe_invoke(
            chat_llm,
            messages= [SystemMessage(content=prompt)]
        )

        # Append the LLM response to the messages
        state['messages'].append(result)

        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Add an error flag to the state
        state['error'] = f"Error in chat node: {str(e)}"
        return state

''' Conditional Functions '''
def from_start_to(state: AgentSchema) -> Literal["parse_menu", "chat"]:
    """Determine the initial routing logic based on the presence of parsed menu data in the state.

    Args:
        state: The current state of the workflow, which may contain parsed menu data.

    Returns:
        Literal["parse_menu", "chat"]: 'parse_menu' if the state does not contain parsed menu data, 'chat' otherwise.
    """
    print_function_name()
    if not state.get('parsed_menu_data'):
        return 'parse_menu'
    return 'chat'

def from_chat_to(state: AgentSchema) -> Literal["chat_tools_memory", "chat_tools_suggest", "chat_tools_finalize", "chat", "__end__"]:
    """
    Route to the correct tool node if the last AI message contains specific tool calls; otherwise route normally.
    """
    print_function_name() if DEBUG else None
    
    # Check if the conversation is finalized
    if state.get('mode') == 'parse_menu':
        return "__end__"
    
    messages = state.get('messages', [])
    if not messages:
        return "chat"
    
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return "chat"
    
    tool_calls = last_message.tool_calls
    if not tool_calls:
        return "__end__"
    
    # Iterate through all tool calls to handle multiple tool calls
    for tool_call in tool_calls:
        try:
            tool_name = tool_call['function']['name']
        except KeyError:
            # Skip malformed tool calls
            continue
        
        if tool_name == 'suggest_ranked_list':
            return "chat_tools_suggest"
        elif tool_name in ('fetch_user_memory', 'update_user_memory'):
            return "chat_tools_memory"
        elif tool_name == 'finalize_conversation':
            return "chat_tools_finalize"
    
    # If no valid tool calls are found, route to chat
    return "chat"
def chat_tools_suggest(state: AgentSchema) -> AgentSchema:
    """
    Handle suggest_ranked_list tool calls. Extract tool calls from last message, invoke the tool,
    append ToolMessage to messages, update recommendations in state, and return updated state.
    """
    print_function_name() if DEBUG else None
    try:
        # Get the last message from the state
        messages: List[BaseMessage] = state.get('messages', [])
        if not messages:
            raise ValueError("No messages in state")

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError("Last message is not an AIMessage")

        # Extract tool calls from the last message
        tool_calls = last_message.tool_calls
        if tool_calls is None:
            raise ValueError("Tool calls list is None")
        
        if not tool_calls:
            raise ValueError("No tool calls found in last message")

        # Iterate through tool calls to find the suggest_ranked_list tool call
        for tool_call in tool_calls:
            tool_name = tool_call['function']['name']
            if tool_name == 'suggest_ranked_list':
                # Parse the arguments for the tool call
                args = tool_call['function'].get('arguments', {})
                recommendations = parse_tool_arguments(args)['recommendations']

                # Validate recommendations
                if not isinstance(recommendations, list):
                    raise ValueError("Recommendations must be a list")
                for item in recommendations:
                    if not isinstance(item, Recommendation):
                        raise ValueError(f"Item {item} is not a Recommendation instance")

                # Invoke the suggest_ranked_list tool
                observation = suggest_ranked_list.invoke({'recommendations': recommendations})

                # Update the state with the recommendations
                state['recommendations'] = recommendations
                state['mode'] = 'parse_menu'

                # Append a ToolMessage to the messages list
                state['messages'].append(
                    ToolMessage(
                        content=observation,
                        name='suggest_ranked_list',
                        tool_call_id=tool_call['id']
                    )
                )

        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET} Error in chat_tools_suggest: {str(e)}') if DEBUG else None
        state['error'] = f"Error in chat_tools_suggest: {str(e)}"
        return state

def chat_tools_finalize(state: AgentSchema) -> AgentSchema:
    """
    Handle finalize_conversation tool calls. Extract tool calls from last message, invoke the tool,
    append ToolMessage to messages, and return updated state routing to __end__.
    """
    print_function_name() if DEBUG else None
    messages: List[BaseMessage] = state.get('messages', [])
    if not messages:
        print(f'{RED}[NODE] [ERR]{RESET} No messages in state') if DEBUG else None
        return state

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        print(f'{RED}[NODE] [ERR]{RESET} Last message is not an AIMessage') if DEBUG else None
        return state

    tool_calls = last_message.tool_calls
    if not tool_calls:
        print(f'{RED}[NODE] [ERR]{RESET} No tool calls in last message') if DEBUG else None
        return state

    for tool_call in tool_calls:
        try:
            tool_name = tool_call['function']['name']
            if tool_name == 'finalize_conversation':
                try:
                    observation = finalize_conversation.invoke({})
                    print(f'{GREEN}[NODE] [INFO]{RESET} Successfully invoked finalize_conversation') if DEBUG else None
                    state['messages'].append(
                        ToolMessage(
                            content=observation,
                            name=tool_name,
                            tool_call_id=tool_call['id']
                        )
                    )
                    state['mode'] = 'parse_menu'
                    break
                except Exception as e:
                    print(f'{RED}[NODE] [ERR]{RESET} Error invoking finalize_conversation: {str(e)}') if DEBUG else None
                    traceback.print_exc() if DEBUG else None
                    continue
        except KeyError:
            print(f'{RED}[NODE] [ERR]{RESET} Invalid tool call structure') if DEBUG else None
            continue

    return state

''' Graph '''
menu_recommendation_workflow_graph = StateGraph(AgentSchema)

menu_recommendation_workflow_graph.add_node("parse_menu", parse_menu)
menu_recommendation_workflow_graph.add_node("chat", chat)

# Add tool nodes
menu_recommendation_workflow_graph.add_node("chat_tools_memory", ToolNode([fetch_user_memory, update_user_memory]))
menu_recommendation_workflow_graph.add_node("chat_tools_suggest", chat_tools_suggest)
menu_recommendation_workflow_graph.add_node("chat_tools_finalize", chat_tools_finalize)

menu_recommendation_workflow_graph.add_conditional_edges(
    "start",
    from_start_to,
    {   # Not needed just for clarity
        "parse_menu": "parse_menu",
        "chat": "chat",
    }
)
menu_recommendation_workflow_graph.add_edge("parse_menu", "chat")

# Replace direct edge with conditional routing for tool handling
menu_recommendation_workflow_graph.add_conditional_edges(
    "chat",
    from_chat_to,
    {
        "chat_tools_memory": "chat_tools_memory",
        "chat_tools_suggest": "chat_tools_suggest",
        "chat_tools_finalize": "chat_tools_finalize",
        "chat": "chat",
        "__end__": "__end__"
    }
)

# Route tool nodes back to appropriate destinations
menu_recommendation_workflow_graph.add_edge("chat_tools_memory", "chat")
menu_recommendation_workflow_graph.add_edge("chat_tools_suggest", "__end__")
menu_recommendation_workflow_graph.add_edge("chat_tools_finalize", "__end__")

menu_recommendation_workflow_app = menu_recommendation_workflow_graph.compile(checkpointer= MemorySaver())

''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(menu_recommendation_workflow_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/menu_recommendation_workflow_app.png', 'wb') as f:
        f.write(menu_recommendation_workflow_app.get_graph().draw_mermaid_png())

    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'menu_recommendation_workflow'
    os.environ['LANGSMITH_PROJECT'] = 'menu_recommendation_workflow'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'menu_recommendation_workflow',
            'run_name': 'menu_recommendation_workflow',
            'thread_id': 'menu_recommendation_workflow',
        }
    }

    user = '' # TODO: add
    response = menu_recommendation_workflow_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')