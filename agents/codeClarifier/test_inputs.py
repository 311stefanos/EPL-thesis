workflow = {
    "type": "reactive_conversational",
    "name": "Conversational WhatsApp Agent",
    "nodes": [
        {
            "name": "start",
            "description": "Execution: CODE. Capture incoming WhatsApp message.",
            "subgraph_id": None
        },
        {
            "name": "message_parser",
            "description": "Execution: LLM. Analyze user intent (query/recommendation request).",
            "subgraph_id": None
        },
        {
            "name": "database_query",
            "description": "Execution: CODE. Retrieve relevant review data from SQLite based on parsed intent.",
            "subgraph_id": None
        },
        {
            "name": "response_generator",
            "description": "Execution: LLM. Generate natural language response/recommendation.",
            "subgraph_id": None
        },
        {
            "name": "whatsapp_response_sender",
            "description": "Execution: TOOLS. Send WhatsApp reply to user.",
            "subgraph_id": None
        },
        {
            "name": "end",
            "description": "Execution: CODE. End of the conversational flow.",
            "subgraph_id": None
        }
    ],
    "edges": [
        {
            "source_name": "start",
            "target_name": "message_parser",
            "description": "Guard: message captured successfully. Pass message to parser."
        },
        {
            "source_name": "message_parser",
            "target_name": "database_query",
            "description": "Guard: intent requires data. Pass parsed intent to query."
        },
        {
            "source_name": "database_query",
            "target_name": "response_generator",
            "description": "Guard: data retrieved. Pass data to response generator."
        },
        {
            "source_name": "response_generator",
            "target_name": "whatsapp_response_sender",
            "description": "Guard: response generated. Pass response to sender."
        },
        {
            "source_name": "whatsapp_response_sender",
            "target_name": "end",
            "description": "Guard: response sent. End conversation flow."
        }
    ],
    "description": "Reactive conversational agent that handles incoming WhatsApp messages and provides review-based responses. Trigger: incoming WhatsApp message. I/O Mode: streaming."
}

filename = 'conversational_whatsapp_agent.py'
with open(f'../../creations/whatsapp_review_processing/{filename}', 'r') as f:
    code_structure = f.read()

clarified_user_input = '''
Develop a Unix-based agent that periodically fetches your Google Maps reviews using third-party APIs (with your provided credentials and place IDs), stores review details (text content, rating 1-5, location metadata, and timestamp) in a structured SQLite database, and provides a WhatsApp-based conversational interface 
for natural language queries about your review history and personalized recommendations based on your stored data.

- `role`: Review collection and analysis agent
- `scope/boundaries`:
  - Collects Google Maps reviews via third-party API
  - Stores data in local SQLite database
  - Provides conversational interface via WhatsApp for queries/recommendations
  - Excludes direct Google API access or web scraping
- `inputs/data sources`:
  - Third-party API credentials (SerpApi/Outscraper)
  - List of place IDs for frequented locations
  - User conversational queries/questions (sent via WhatsApp)
- `outputs/format`:
  - Structured SQLite database (reviews table with text, rating, location, timestamp, place_id)
  - Natural language responses to WhatsApp queries
  - Recommendations based on stored review patterns
- `constraints`:
  - Cost (third-party API usage fees)
  - Rate limits (API provider constraints)
  - Privacy (secure handling of API credentials)
  - Legal compliance (third-party ToS adherence)
  - No direct Google Maps API access or web scraping
  - WhatsApp integration limitations (message length, availability)
- `key preferences/deadlines`:
  - Prefer third-party API approach over scraping/detection
  - SQLite database storage
  - WhatsApp-based conversational interface
  - Recommendations based solely on user's stored history
- `additional technical requirements`:
  - Scheduled execution (cron jobs for periodic checks)
  - Natural language processing capabilities
  - Database schema design
  - API integration layer
  - WhatsApp integration (using Twilio/WhatsApp Business API)
  - Conversational interface implementation
  - Message parsing for WhatsApp queries'''