workflow = {
    "comments": "User requested to use LLM for workout_design, meal_plan_creation, and progress_guidance steps. Updated execution types accordingly.",
    "root": {
        "type": "linear_pipeline",
        "name": "Fitness Program Generator",
        "nodes": [
            {
                "name": "start_node",
                "description": "Execution: CODE. Triggered when the user provides complete input data (health status, equipment, session requirements, time slot preference, dietary preferences) via a structured form.",
                "subgraph_id": None
            },
            {
                "name": "input_collection",
                "description": "Execution: CODE. Collect and validate user inputs for program customization. Guard: All mandatory inputs are provided and valid.",
                "subgraph_id": None
            },
            {
                "name": "macro_calculation",
                "description": "Execution: CODE. Calculate daily calorie and macronutrient targets for simultaneous weight loss/muscle gain using standard formulas (e.g., Mifflin-St Jeor for BMR). Guard: Inputs are valid and complete.",
                "subgraph_id": None
            },
            {
                "name": "workout_design",
                "description": "Execution: LLM. Design a structured 5-day/week workout plan using bodyweight exercises and running, split between dawn/afternoon slots. Guard: Macro targets calculated and inputs valid.",
                "subgraph_id": None
            },
            {
                "name": "meal_plan_creation",
                "description": "Execution: LLM. Generate a weekly meal plan with portion guidance aligned with macro targets. Guard: Workout plan finalized.",
                "subgraph_id": None
            },
            {
                "name": "progress_guidance",
                "description": "Execution: LLM. Provide static progression/scaling rules (time-based), recovery guidelines, and metrics for self-tracking (weight, measurements). Guard: Meal plan created.",
                "subgraph_id": None
            },
            {
                "name": "output_generation",
                "description": "Execution: CODE. Deliver weekly workout schedules, exercise instructions, meal plans, and progress tracking guidelines in English. Guard: All steps complete.",
                "subgraph_id": None
            },
            {
                "name": "end_node",
                "description": "Execution: CODE. End of workflow.",
                "subgraph_id": None
            }
        ],
        "edges": [
            {
                "source_name": "start_node",
                "target_name": "input_collection",
                "description": "User provides complete input data (health status, equipment, session requirements, time slot preference, dietary preferences)."
            },
            {
                "source_name": "input_collection",
                "target_name": "macro_calculation",
                "description": "All mandatory inputs are collected and validated."
            },
            {
                "source_name": "macro_calculation",
                "target_name": "workout_design",
                "description": "Macro targets calculated and inputs valid."
            },
            {
                "source_name": "workout_design",
                "target_name": "meal_plan_creation",
                "description": "Workout plan finalized."
            },
            {
                "source_name": "meal_plan_creation",
                "target_name": "progress_guidance",
                "description": "Meal plan created."
            },
            {
                "source_name": "progress_guidance",
                "target_name": "output_generation",
                "description": "Progress guidance ready."
            },
            {
                "source_name": "output_generation",
                "target_name": "end_node",
                "description": "Workflow complete."
            }
        ],
        "description": "Linear workflow that generates a customized fitness program (workout and meal plans) and progress tracking guidelines. Triggered by user input via a structured form; runs in batch mode without human interaction. Uses LLM for workout design, meal plan creation, and progress guidance."
    },
    "subgraphs": {}
}

file_path = '../../creations/fitness_program_generator/fitness_program_generator.py'
with open(file_path, 'r') as f:
    code_structure = f.read()

clarified_user_input = '''<refined paragraph>
Design a comprehensive virtual AI-powered fitness and nutrition coaching agent that creates personalized, 
structured programs targeting simultaneous weight loss and muscle gain. The agent must utilize only bodyweight 
exercises and running as available equipment, accommodating 5 weekly sessions of 90 minutes each. 
Programs should be adaptable to either morning (dawn) or afternoon workout time slots within a free or minimal-cost model, 
delivered entirely in English. The solution requires no human interaction or location dependency, with integrated dietary planning and nutritional guidance.
- `role`: Virtual AI fitness coach and dietary assistant
- `scope/boundaries`:
   - Designs structured workout plans using bodyweight exercises and running
   - Creates integrated dietary plans for weight loss and muscle gain
   - Provides program guidance only (no execution or equipment provision)
   - Operates within free/minimal-cost constraints
   - Delivers content solely in English
- `inputs/data sources`:
   - User's health status (no conditions/injuries)
   - Available equipment: bodyweight exercises, running
   - Session requirements: 5 days/week, 90 minutes/session
   - Time flexibility: morning (dawn) or afternoon
   - Dietary preferences/allergies (if any)
- `outputs/format`:
   - Weekly workout plans (structured schedules)
   - Exercise instructions with form guidance
   - Integrated weekly meal plans with portion guidance
   - Macronutrient targets aligned with dual goals
   - Progress tracking metrics for both fitness and nutrition
- `constraints`:
   - **Cost**: Free or minimal-cost (freemium model)
   - **Equipment**: Bodyweight and running only
   - **Time**: Programs adaptable to dawn or afternoon slots
   - **Safety**: Safe for general healthy individuals
   - **Scope**: No medical diagnosis or prescription capabilities
   - **Language**: English-only delivery
   - **Dietary Limits**: Should avoid complex medical nutrition therapy
- `key preferences`:
   - Simultaneous weight loss and muscle gain focus
   - Integrated exercise and nutrition approach
   - Strict adherence to 5x90 minute weekly structure
- `additional requirements`:
   - Progression/scaling mechanisms for workouts and nutrition
   - Recovery and rest day guidelines
   - Form correction tips for injury prevention
   - Basic nutritional guidance with calorie/macro calculations
   - Hydration advice and food logging suggestions
   - Dietary flexibility options for preferences/restrictions'''