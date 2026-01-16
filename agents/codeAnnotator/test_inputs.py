workflow = {
    "comments": "The workflow has been adjusted to parse the menu at the start, with the start node connecting to both the parse and chat nodes. This ensures menu data is processed early in the workflow.",
    "root": {
        "description": "A reactive conversational workflow that processes user messages to provide menu recommendations based on user preferences and feedback. The workflow starts by parsing the menu data and then processes the user message to generate recommendations.",
        "edges": [
            {
                "description": "Transition to parse the menu data before processing the user message.",
                "source_name": "start",
                "target_name": "parse_menu"
            },
            {
                "description": "Transition to the chat node to process the user message and generate a response.",
                "source_name": "start",
                "target_name": "chat"
            },
            {
                "description": "Transition to the chat node after parsing the menu data.",
                "source_name": "parse_menu",
                "target_name": "chat"
            },
            {
                "description": "Transition to the end node to terminate the workflow run after processing the user message.",
                "source_name": "chat",
                "target_name": "end"
            }
        ],
        "memory": True,
        "name": "menu_recommendation_workflow",
        "nodes": [
            {
                "description": "Execution: CODE. Route based on the mode/next_action stored in state and initiate menu parsing.",
                "name": "start"
            },
            {
                "description": "Execution: CODE. Parse the menu data provided by the user, supporting photo, link, or text formats.",
                "name": "parse_menu"
            },
            {
                "description": "Execution: LLM+TOOLS. Process the user message, retrieve user preferences using user_preferences_manager, generate recommendations, and handle feedback using feedback_processor.",
                "name": "chat"
            },
            {
                "description": "Execution: CODE. Terminate the workflow run.",
                "name": "end"
            }
        ],
        "type": "reactive_conversational"
    },
    "subgraphs": {}
}

file_path = '..\..\creations\menu_recommendation_workflow\menu_recommendation_workflow.py'

clarified_user_input = '''The agent will store and manage user-specific food and drink preferences, including dietary restrictions and allergies, in a structured JSON file. Upon receiving a menu (via photo, link, or text), it will parse the content in a single step and generate a ranked list of recommendations with clear explanations for each suggestion. The agent will support multiple user profiles, adapt over time based on feedback, and operate exclusively within the scope of menu items. It will engage users conversationally with a friendly tone in English, functioning as a WhatsApp chatbot without relying on external tools.

**Agent-Creation Essentials:**
- **Role:** Personalized menu recommendation assistant.
- **Scope/Boundaries:** Focused solely on food/drink menu items; no external tool usage.
- **Inputs/Data Sources:** Menu data (photo/link/text), user preferences (JSON file), user feedback.
- **Outputs/Format:** Ranked list of menu recommendations with explanations; conversational responses.
- **Constraints:**
  - Language: English only.
  - Style: Friendly, interactive tone.
  - Latency: Single-step menu parsing.
  - Safety: No external tools; data stored locally (JSON).
- **Key Preferences:**
  - Multi-user profile support.
  - Adaptive learning from feedback.
  - Deployment as a WhatsApp chatbot.
- **Deadlines:** None specified.'''