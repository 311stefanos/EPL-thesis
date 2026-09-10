from dotenv import load_dotenv
from pathlib import Path
from typing import Any
import os

from langchain_core.messages import SystemMessage

from utils.utils import myChatOpenAI, safe_invoke, clean_llm_output


''' Constants '''
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m'
RED = '\033[91m'
RESET = '\033[0m'


''' LLM '''
ablate_all_llm = myChatOpenAI(
    temperature=0.4
)


''' Prompt '''
ABLATE_ALL_PROMPT = """
You are a senior AI-agent software engineer.

Your task is to transform the user's natural-language request directly into one complete, runnable Python implementation of the requested AI agent.

This is a one-shot generation task. You have no follow-up agents, reviewers, code testers, prompt engineers, clarification agents, search tools, or repair loops. Therefore, produce the strongest complete implementation you can in this single response.

# User Request
<USER_REQUEST_START>
{user_request}
<USER_REQUEST_END>

# Requirements
- Return one complete Python source file.
- The implementation must be runnable and internally consistent.
- Include all required imports.
- Include schemas, tools, LLM declarations, helper functions, nodes, routing logic, graph construction, and compilation when required by the requested agent.
- Use LangGraph when the requested behavior requires an agentic or multi-step workflow.
- Use myChatOpenAI for LLM creation.
- Use safe_invoke for LLM invocation.
- Use clear state schemas and explicit node return values.
- Use prompts as normal Python strings inside this generated file when LLM instructions are required.
- Keep prompt construction simple: initialize required values first, then format the relevant prompt directly.
- Do not use dynamic prompt discovery, getattr-based prompt lookup, runtime prompt mutation, or fallback prompt-name searching.
- Bind tools only when the model actually needs to call them.
- Use ToolNode and conditional routing only when tool execution is required.
- Do not leave TODO implementations, pass statements, NotImplementedError, placeholder functions, or omitted sections.
- Do not explain the code outside the source file.
- Do not return markdown fences.

# Output
Return only the complete Python source code.
""".strip()


def ablate_all(user_request: str) -> str:
    """
    Generates the complete requested AI agent using exactly one LLM request.

    Args:
        user_request: Natural-language description of the requested agent.

    Returns:
        Complete generated Python source code.
    """
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError('user_request must be a non-empty string.')

    prompt: str = ABLATE_ALL_PROMPT.format(
        user_request=user_request.strip()
    )

    result: Any = safe_invoke(
        ablate_all_llm,
        messages=[
            SystemMessage(content=prompt)
        ]
    )

    generated_code: str = clean_llm_output(str(result.content))

    if not generated_code.strip():
        raise ValueError('The ablate-all model returned an empty response.')

    return generated_code


if __name__ == '__main__':
    user_request: str = (
        'I want a personal agent that I can send my receipts (in a photo) through a WhatsApp chat,'
        ' and it should read the photo, understand the receipt (cost, date, items, location, etc.), and insert the'
        ' data into an Excel file. The Excel file should have columns for cost, date, items (format: item1 (quantity'
        ' x price currency), ...), location, category. The category should be automatically determined from the'
        ' agent based on the items. It should also answer questions about spending when prompted by the user'
        ' (e.g. ‘how much did I spend on groceries this month?’, ‘what are the top 3 items I spent the most on?’,'
        ' etc.). For OCR processing, you may call a vision model through OpenRouter.'
        ' For the WhatsApp API, consider it out-of-scope.'
    )

    try:
        code = ablate_all(user_request)

        with open('../experiments/ablation_study/ablate_all/ablate_all.py', 'w', encoding= 'utf-8') as f:
            f.write(code)
        
    except Exception as exc:
        print(f'{RED}[ABLATE ALL] [ERROR]{RESET} {exc}')