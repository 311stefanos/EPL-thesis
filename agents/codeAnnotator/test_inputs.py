workflow = {
    "comments": "The user requested that 'generate_suggestions' and 'collect_feedback' nodes should loop back to themselves, and 'update_preferences' should be a tool used by 'collect_feedback'.",
    "root": {
        "type": "reactive_conversational",
        "name": "whatsapp_menu_suggestion_workflow",
        "nodes": [
            {
                "name": "start",
                "description": "Execution: CODE. Start node triggered by a user sending a WhatsApp message (text, image, link, or PDF). Initializes the conversation context and prepares to receive menu input.",
                "subgraph_id": None
            },
            {
                "name": "receive_menu_input",
                "description": "Execution: LLM+TOOLS. Extract text from menu input (photo, URL, or PDF) using OpenRouter vision API, requests/BeautifulSoup, and PyMuPDF/pdfplumber. Parse the extracted text into a structured format.",
                "subgraph_id": None
            },
            {
                "name": "generate_suggestions",
                "description": "Execution: LLM+TOOLS. Engage in free-form conversation to generate dish suggestions based on parsed menu items and user preferences stored in local JSON. Uses a tool called 'suggest_list(list: List)' to transition to the next node.",
                "subgraph_id": None
            },
            {
                "name": "send_suggestions",
                "description": "Execution: TOOLS. Send the ranked dish suggestions to the user via WhatsApp using the Twilio WhatsApp API.",
                "subgraph_id": None
            },
            {
                "name": "collect_feedback",
                "description": "Execution: LLM+TOOLS. Engage in free-form conversation to collect feedback on the meal and parse preference updates from the dialogue using an LLM. Uses a tool called 'update_preferences' to update user preferences.",
                "subgraph_id": None
            },
            {
                "name": "end",
                "description": "Execution: CODE. End node that signals the termination of the workflow when the user ends the conversation or no further input is received.",
                "subgraph_id": None
            }
        ],
        "edges": [
            {
                "source_name": "start",
                "target_name": "receive_menu_input",
                "description": "Transition triggered when a valid WhatsApp message is received from the user."
            },
            {
                "source_name": "receive_menu_input",
                "target_name": "generate_suggestions",
                "description": "Proceed to generate suggestions once the menu input has been successfully parsed."
            },
            {
                "source_name": "generate_suggestions",
                "target_name": "send_suggestions",
                "description": "Transition occurs when the LLM uses the tool 'suggest_list(list: List)' to generate and send suggestions."
            },
            {
                "source_name": "send_suggestions",
                "target_name": "collect_feedback",
                "description": "Engage in conversation to collect feedback after suggestions have been sent."
            },
            {
                "source_name": "collect_feedback",
                "target_name": "end",
                "description": "Terminate the workflow after feedback has been collected and preferences have been updated."
            },
            {
                "source_name": "generate_suggestions",
                "target_name": "generate_suggestions",
                "description": "Loop back to continue generating suggestions if more interaction is needed."
            },
            {
                "source_name": "collect_feedback",
                "target_name": "collect_feedback",
                "description": "Loop back to continue collecting feedback if more interaction is needed."
            }
        ],
        "description": "A reactive conversational workflow that processes user-provided menu inputs via WhatsApp, generates dish suggestions based on user preferences, sends these suggestions back to the user, collects feedback, and updates preferences. The workflow is triggered by user messages and operates in a streaming I/O mode."
    },
    "subgraphs": {}
}

file_path = '..\..\creations\whatsapp_menu_suggestion_workflow\whatsapp_menu_suggestion_workflow.py'

clarified_user_input = '''A WhatsApp-integrated food and drink preference agent that maintains persistent user profiles in local JSON storage. The agent receives menu inputs through three channels: photos processed via OpenRouter's free vision models (like `google/gemini-2.0-flash-exp:free`) for OCR text extraction, URLs scraped with requests/BeautifulSoup, and PDFs parsed with PyMuPDF/pdfplumber. It provides ranked dish suggestions with brief explanations by matching menu items against stored preferences: dietary restrictions, cuisine types, flavor profiles (spicy/sweet/salty/bitter/sour), price range tiers, favorite/disliked ingredients, and specific dish preferences. After meals, the agent engages in free-form conversation to collect feedback, uses an LLM to parse preference updates from the dialogue, and immediately appends these to the user's JSON profile. Multiple users are supported via username identification. The system runs as a Flask web app with Twilio WhatsApp integration, using ngrok for local webhook testing. Suggestions are generated in real-time upon receiving menu input, and preference memory updates occur after each feedback conversation session.

- **role**: Food/drink preference agent with menu analysis, suggestion generation, and dynamic memory updating
- **scope/boundaries**: WhatsApp messaging via Twilio API, local JSON file storage on hosting PC, handles photos/URLs/PDFs for menu input, provides personalized suggestions, parses free-form feedback conversations for preference updates, supports multiple username-based user profiles
- **inputs/data sources**: WhatsApp messages (text, images, links), OpenRouter vision API for OCR, web scraping with requests/BeautifulSoup for URL content, PyMuPDF/pdfplumber for PDF text extraction, local JSON file for user preference database
- **outputs/format**: Ranked list of menu items with brief explanations (e.g., "1. Margherita Pizza - matches your preference for vegetarian, mid-range, and mozzarella cheese"), conversational responses for feedback collection
- **constraints (cost/latency/safety/style/language)**: Use OpenRouter free tier models for cost efficiency, local JSON storage only (no cloud databases), natural language conversation style, handle one user conversation at a time, web scraping with reasonable rate limits
- **preference categories to track**: Dietary restrictions (allergies, vegetarian, vegan, gluten-free), cuisine preferences (Italian, Asian, Mexican, etc.), flavor profiles (spicy, sweet, salty, bitter, sour), price range (budget, mid-range, premium), favorite ingredients, disliked ingredients, specific dish preferences (e.g., "likes pasta with cream sauce")
- **feedback processing**: Free-form conversation about meals eaten, LLM-powered extraction of preference updates, immediate JSON append after conversation ends
- **technical implementation**: Flask web app with `/webhook` endpoint for Twilio, ngrok for local HTTPS tunneling, preference JSON structure with username keys, OCR fallback chain (OpenRouter vision → text extraction), web scraping with error handling
- **menu processing workflow**: Photo → OpenRouter vision → text extraction → parsing; URL → requests/BeautifulSoup → content extraction; PDF → PyMuPDF/pdfplumber → text extraction → parsing
- **suggestion ranking logic**: Match against dietary restrictions, score by cuisine/flavor/price alignment, exclude disliked items, rank by preference fit, include brief explanation for each item'''