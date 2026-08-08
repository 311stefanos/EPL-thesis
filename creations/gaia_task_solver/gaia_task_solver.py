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
from typing import Literal, List, Optional, Union
from pydantic import BaseModel

# General imports
from dotenv import load_dotenv
from pathlib import Path
import traceback
import json
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, parse_tool_arguments, clean_llm_output
from creations.gaia_task_solver import gaia_task_solver_prompts as prompts

import requests
import time
import base64



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} gaia_task_solver') if DEBUG else None



""" Schemas """
class ReviewAnswerSchema(BaseModel):
    """
    Structured output schema for the review_answer node. This schema defines the expected JSON structure from the review_answer_llm, containing the review decision (approve/revise) and optional actionable feedback for revision. Used to extract structured data from LLM output before updating AgentState.
    """
    review_decision: Literal['approve', 'revise'] # The outcome from the review process indicating whether the candidate answer is approved or needs revision.
    review_feedback: Optional[str] # Detailed, actionable feedback from the reviewer when review_decision is 'revise'. None when approved.

class FormatOutputSchema(BaseModel):
    """
    Structured output schema for the format_output node. This schema defines the expected JSON structure from the format_output_llm, containing the thinking process (reasoning behind the formatting) and the final GAIA-required answer string. Used to extract structured data from LLM output before updating AgentState.
    """
    thinking_process: str # The reasoning and thought process behind how the final answer was formatted to meet GAIA requirements.
    final_output: str # The exact GAIA-formatted answer string that meets all formatting, precision, and content requirements.

class AgentSchema(MessagesState):
    """
    The state schema for the GAIA task solver workflow. This TypedDict extends MessagesState to include all necessary fields for tracking the analysis, execution, review, and formatting phases of solving GAIA benchmark tasks. It maintains conversation history, task details, intermediate results, and review state throughout the workflow execution.
    """
    question: str # The original GAIA question string provided by the user.
    attachments: List[str] # List of file paths that may contain relevant data for solving the task.
    task_analysis: Union[str, dict] # The high-level plan produced by analyze_task node, identifying objectives, answer formats, files to inspect, and required tools.
    candidate_answer: str # The provisional answer produced by solve_task node before review.
    answer_format: str # The required format for the answer (e.g., 'number', 'text', 'date', etc.).
    steps_completed: List[str] # List of steps that were completed to reach the answer.
    evidence: List[str] # Supporting evidence or sources used to derive the answer.
    calculations: List[str] # Records of any calculations performed during the solving process.
    unresolved_issues: List[str] # Notes about any remaining uncertainties or issues with the solution.
    review_decision: Literal['approve', 'revise'] # The outcome from review_answer node indicating whether the candidate answer is approved or needs revision.
    review_feedback: Optional[str] # Detailed, actionable feedback from review_answer node when review_decision is 'revise'.
    review_count: int # Number of times the answer has been reviewed, used to limit revision loops to a maximum of 2.
    final_output: str # The GAIA-formatted answer string produced by format_output node after review approval or after maximum revisions.



''' Tools '''
@tool
def file_parser(file_path: str, instruction: str) -> dict:
    """
    Overview:
    Inspects an attached or downloaded file using the configured LLM through
    `myChatOpenAI` and `safe_invoke`.

    Text files are decoded locally. Images are sent as multimodal image content.
    PDFs are sent as file content. DOCX and XLSX files are converted to text
    locally when direct file input is unavailable.

    Args:
        file_path: Path to the file that must be parsed.
        instruction: Specific information to extract from the file.

    Returns:
        A dictionary containing extracted information, evidence, uncertainties,
        confidence, and file metadata.
    """
    SYSTEM_PROMPT = (
        "You are a meticulous file analysis expert. Inspect the provided file "
        "and extract all relevant information based on the user's instruction. "
        "Return only one valid JSON object with these keys:\n"
        "- 'extracted_info': the main extracted information, preserving exact text, numbers, and structure\n"
        "- 'evidence': a list of precise evidence locations such as pages, sheets, rows, cells, lines, or visual regions\n"
        "- 'uncertainties': a list of uncertainties or ambiguities\n"
        "- 'confidence': an integer from 0 to 100\n\n"
        "If the file contains tables, preserve them as structured data. "
        "If it contains images, describe only relevant visual elements. "
        "Do not invent evidence locations."
    )

    def error_result(message: str, uncertainty: str, *, metadata: Optional[dict] = None) -> dict:
        result = {
            'error': message,
            'extracted_info': None,
            'evidence': [],
            'uncertainties': [uncertainty],
            'confidence': 0
        }

        if metadata is not None:
            result['file_metadata'] = metadata

        return result

    # Validate the file path.
    if not os.path.exists(file_path):
        return error_result(f'File not found: {file_path}', 'File does not exist')

    # Keep the existing 10 MB limit.
    file_size = os.path.getsize(file_path)

    if file_size > 10 * 1024 * 1024:
        return error_result(f'File too large: {file_size} bytes (max 10MB)','File exceeds the 10MB limit')

    ext = os.path.splitext(file_path)[1].lower()

    MIME_MAP = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
        '.csv': 'text/csv',
        '.txt': 'text/plain',
        '.md': 'text/markdown',
        '.json': 'application/json',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    }

    mime_type = MIME_MAP.get(ext, 'application/octet-stream')

    metadata = {
        'path': file_path,
        'name': os.path.basename(file_path),
        'size': file_size,
        'mime_type': mime_type
    }

    try:
        with open(file_path, 'rb') as file:
            file_data = file.read()

    except Exception as e:
        return error_result(
            f'Failed to read file: {e}',
            'File could not be read',
            metadata=metadata
        )

    provider = (os.getenv('PROVIDER') or '').upper()

    image_extensions = {
        '.png',
        '.jpg',
        '.jpeg',
        '.gif',
        '.bmp',
        '.webp'
    }

    text_extensions = {
        '.csv',
        '.txt',
        '.md',
        '.json'
    }

    user_content = [{
        'type': 'text',
        'text': (
            f"Instruction:\n{instruction}\n\n"
            f"File name: {metadata['name']}\n"
            f"File MIME type: {mime_type}"
        )
    }]

    try:
        # Plain-text files can be included directly in the prompt.
        if ext in text_extensions:
            try:
                decoded_text = file_data.decode('utf-8-sig')

            except UnicodeDecodeError:
                decoded_text = file_data.decode('latin-1', errors='replace')

            max_text_chars = 250_000
            was_truncated = len(decoded_text) > max_text_chars
            decoded_text = decoded_text[:max_text_chars]

            truncation_note = (
                f"\n\n[FILE CONTENT TRUNCATED AFTER {max_text_chars} CHARACTERS]"
                if was_truncated else ''
            )

            user_content.append({
                'type': 'text',
                'text': f"File contents:\n{decoded_text} {truncation_note}"
            })

        # Images use the OpenAI-compatible multimodal image format.
        elif ext in image_extensions:
            encoded_file = base64.b64encode(file_data).decode('utf-8')
            data_url = f'data:{mime_type};base64,{encoded_file}'

            user_content.append({
                'type': 'image_url',
                'image_url': {'url': data_url}
            })

        # OpenRouter supports PDF file blocks.
        elif provider == 'OPENROUTER' and ext == '.pdf':
            encoded_file = base64.b64encode(file_data).decode('utf-8')
            data_url = f'data:{mime_type};base64,{encoded_file}'

            user_content.append({
                'type': 'file',
                'file': {
                    'filename': metadata['name'],
                    'file_data': data_url
                }
            })

        # Standard LangChain file block for providers supporting PDF input.
        elif ext == '.pdf':
            user_content.append({
                'type': 'file',
                'base64': base64.b64encode(file_data).decode('utf-8'),
                'mime_type': mime_type,
                'filename': metadata['name']
            })

        # Extract DOCX text locally, then send it through your configured API.
        elif ext == '.docx':
            import zipfile
            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(file_path) as archive:
                xml_data = archive.read('word/document.xml')

            root = ET.fromstring(xml_data)

            namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

            paragraphs = []

            for paragraph in root.findall('.//w:p', namespace):
                paragraph_text = ''.join(
                    node.text or ''
                    for node in paragraph.findall('.//w:t', namespace)
                ).strip()

                if paragraph_text:
                    paragraphs.append(paragraph_text)

            extracted_text = '\n'.join(paragraphs)

            max_text_chars = 250_000
            was_truncated = len(extracted_text) > max_text_chars
            extracted_text = extracted_text[:max_text_chars]

            if was_truncated:
                extracted_text += f"\n\n[DOCX CONTENT TRUNCATED AFTER {max_text_chars} CHARACTERS]"
                

            user_content.append({
                'type': 'text',
                'text': f'DOCX extracted text:\n{extracted_text}'
            })

        # Extract XLSX values locally, then send them through your API.
        elif ext == '.xlsx':
            try:
                from openpyxl import load_workbook

            except ImportError:
                return error_result(
                    'XLSX parsing requires the openpyxl package',
                    'The openpyxl dependency is not installed',
                    metadata=metadata
                )

            workbook = load_workbook(
                file_path,
                read_only=True,
                data_only=True
            )

            workbook_text = []
            max_rows_per_sheet = 10_000

            for worksheet in workbook.worksheets:
                workbook_text.append(
                    f'[SHEET: {worksheet.title}]'
                )

                for row_index, row in enumerate(
                    worksheet.iter_rows(values_only=True),
                    start=1
                ):
                    if row_index > max_rows_per_sheet:
                        workbook_text.append(
                            f'[SHEET TRUNCATED AFTER '
                            f'{max_rows_per_sheet} ROWS]'
                        )
                        break

                    values = [
                        str(value) if value is not None else ''
                        for value in row
                    ]

                    if any(values):
                        workbook_text.append(
                            f"Row {row_index}:\t" + '\t'.join(values)
                        )

            workbook.close()

            extracted_text = '\n'.join(workbook_text)
            max_text_chars = 250_000

            if len(extracted_text) > max_text_chars:
                extracted_text = (
                    extracted_text[:max_text_chars]
                    + f"\n\n[XLSX CONTENT TRUNCATED AFTER "
                    + f"{max_text_chars} CHARACTERS]"
                )

            user_content.append({
                'type': 'text',
                'text': f'XLSX extracted values:\n{extracted_text}'
            })

        else:
            return error_result(
                (
                    f'Unsupported file type for provider '
                    f'{provider or "UNKNOWN"}: '
                    f'{ext or "no extension"}'
                ),
                (
                    'The configured provider cannot receive this file type '
                    'and no local parser is available'
                ),
                metadata=metadata
            )

        # Uses PROVIDER, {PROVIDER}_API_KEY,
        # {PROVIDER}_BASE_URL, and MODEL_NAME.
        parser_llm = myChatOpenAI(
            temperature=0.1,
            model= 'google/gemini-2.5-flash-lite:batch',
            max_retries=0
        )

        response = safe_invoke(
            parser_llm,
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content)
            ]
        )

        raw_content = response.content

        # Some multimodal providers return a list of content blocks.
        if isinstance(raw_content, list):
            text_parts = []

            for block in raw_content:
                if isinstance(block, str):
                    text_parts.append(block)

                elif isinstance(block, dict):
                    block_text = (
                        block.get('text')
                        or block.get('content')
                    )

                    if block_text:
                        text_parts.append(str(block_text))

            response_text = '\n'.join(text_parts)

        else:
            response_text = str(raw_content or '')

        response_text = clean_llm_output(response_text)

        try:
            parsed = json.loads(response_text)

        except json.JSONDecodeError:
            return {
                'extracted_info': response_text,
                'evidence': [],
                'uncertainties': [
                    'Model response was not valid JSON; raw text was returned'
                ],
                'confidence': 50,
                'file_metadata': metadata
            }

        confidence = parsed.get('confidence', 0)

        try:
            confidence = max(
                0,
                min(100, int(confidence))
            )

        except (TypeError, ValueError):
            confidence = 0

        return {
            'extracted_info': parsed.get('extracted_info'),
            'evidence': parsed.get('evidence') or [],
            'uncertainties': parsed.get('uncertainties') or [],
            'confidence': confidence,
            'file_metadata': metadata
        }

    except Exception as e:
        return error_result(
            f'File parser failed: {e}',
            (
                f'{e.__class__.__name__} while invoking '
                f'the configured LLM API'
            ),
            metadata=metadata
        )


@tool
def web_search(query: str) -> list[dict]:
    """
    Overview: 
    Discovers relevant webpages returning title, URL, snippet, and publication date. This tool is used to search for information on the web when solving GAIA tasks that require up-to-date or external knowledge not present in the attached files or internal knowledge.
    
    Caller LLM: solve_task_llm
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): 
    None - The tool handler only needs to append the result as a ToolMessage to the messages state.
    
    Inside-the-Tool Work (Tool Responsibilities): 
    Execute a web search using the provided query, retrieve relevant results, and format them as a list of dictionaries containing title, URL, snippet, and publication date for each result.
    
    Instructions: 
    1. Receive a search query string.
    2. Execute the query against a web search engine.
    3. Retrieve the top relevant results.
    4. For each result, extract the title, URL, snippet/text content, and publication date.
    5. Return a list of dictionaries, each containing these four fields for one search result.
    
    State Updates (on the caller function): 
    None
    
    Args: 
    query: str - The search query to execute for finding relevant information.
    
    Returns: 
    list[dict] - A list where each element is a dictionary with keys 'title', 'url', 'snippet', and 'publication_date' representing a search result.
    """
    try:
        # Try using Tavily API first
        tavily_key = os.getenv('TAVILY_API_KEY')
        if tavily_key:
            response = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_key, "query": query, "max_results": 5}
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for r in data.get('results', []):
                results.append({
                    "title": r.get('title'),
                    "url": r.get('url'),
                    "snippet": r.get('content'),  # Map 'content' to 'snippet'
                    "publication_date": r.get('published_date')
                })
            return results

        # If no API key is configured, return an error indicator
        return [{"error": "No API key configured"}]

    except Exception as e:
        # Return a list with an error message on any failure
        return [{"error": str(e)}]

@tool
def open_url(url: str, instruction: str) -> dict:
    """
    Overview: 
    Opens webpages with extraction instructions or downloads files to the workspace. This tool is used to retrieve content from specific web URLs or download files during the GAIA task solving process. It can either extract specific information from a webpage based on instructions or download the file content for further analysis.
    
    Caller LLM: solve_task_llm
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): 
    None - The tool handler only needs to append the result as a ToolMessage to the messages state.
    
    Inside-the-Tool Work (Tool Responsibilities): 
    Access the specified URL, either extract information from the webpage using the provided instruction or download the file to the workspace, and return the result in a structured format.
    
    Instructions: 
    1. Receive a URL to access and an instruction specifying what to do with the content.
    2. Access the URL (webpage or file).
    3. If the instruction indicates extraction: extract the specified information from the webpage content.
    4. If the instruction indicates download: download the file to the workspace and return its local path.
    5. Return a dictionary containing either the extracted information from the webpage or details about the downloaded file including its local path.
    
    State Updates (on the caller function): 
    None
    
    Args: 
    url: str - The URL to open (could be a webpage or direct file link).
    instruction: str - Guidance on whether to extract information from the URL or download a file, and what specific information to extract if applicable.
    
    Returns: 
    dict - A dictionary containing either the extracted information from the webpage or details about the downloaded file including its local path.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=60)
        status_code = response.status_code
        
        if status_code != 200:
            return {
                'url': url,
                'status_code': status_code,
                'error': f'Failed to access URL. Status code: {status_code}'
            }
        
        if 'download' in instruction.lower():
            # Download mode: save file to workspace
            filename = Path(url).name
            if not filename or '.' not in filename:
                filename = f"downloaded_{int(time.time())}.dat"
            
            # Create downloads directory if it doesn't exist
            download_dir = os.path.join(os.getcwd(), 'downloads')
            os.makedirs(download_dir, exist_ok=True)
            
            local_path = os.path.join(download_dir, filename)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            return {
                'url': url,
                'local_path': local_path,
                'status_code': status_code
            }
        else:
            # Extraction mode: return webpage content
            return {
                'url': url,
                'extracted_info': response.text,
                'status_code': status_code
            }
    
    except requests.exceptions.Timeout:
        return {
            'url': url,
            'status_code': None,
            'error': 'Request timed out after 60 seconds'
        }
    except requests.exceptions.ConnectionError as e:
        return {
            'url': url,
            'status_code': None,
            'error': f'Connection error: {str(e)}'
        }
    except requests.exceptions.RequestException as e:
        return {
            'url': url,
            'status_code': None,
            'error': f'Request failed: {str(e)}'
        }
    except Exception as e:
        return {
            'url': url,
            'status_code': None,
            'error': f'Unexpected error: {str(e)}'
        }

@tool
def run_python(code: str) -> dict:
    """
    Execute Python code inside an isolated Docker container.

    The container has:
    - no network access
    - no access to project or host files
    - a read-only filesystem
    - a non-root user
    - no Linux capabilities
    - limited CPU, memory, processes, files, and execution time

    Args:
        code: Python code to execute.

    Returns:
        A dictionary containing success, stdout, stderr, exit code,
        timeout status, and execution time.
    """
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    import time
    import uuid
    from pathlib import Path

    timeout_seconds = 10
    docker_image = "python:3.11-slim"
    max_code_size = 100_000
    max_output_size = 20_000

    if not isinstance(code, str) or not code.strip():
        return {
            "success": False,
            "output": "",
            "error": "Code must be a non-empty string.",
            "exit_code": None,
            "timed_out": False,
            "execution_time": 0.0,
        }

    if len(code.encode("utf-8")) > max_code_size:
        return {
            "success": False,
            "output": "",
            "error": (
                f"Code exceeds the {max_code_size}-byte limit."
            ),
            "exit_code": None,
            "timed_out": False,
            "execution_time": 0.0,
        }

    if shutil.which("docker") is None:
        return {
            "success": False,
            "output": "",
            "error": "Docker is not installed or unavailable on PATH.",
            "exit_code": None,
            "timed_out": False,
            "execution_time": 0.0,
        }

    container_name = f"gaia-python-{uuid.uuid4().hex}"
    started_at = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="gaia_python_"
    ) as temp_directory:
        sandbox_directory = Path(temp_directory)
        script_path = sandbox_directory / "submitted_code.py"

        script_path.write_text(
            code,
            encoding="utf-8",
        )

        try:
            os.chmod(sandbox_directory, 0o755)
            os.chmod(script_path, 0o644)
        except OSError:
            pass

        docker_command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,

            # Proper signal handling.
            "--init",

            # Do not download or update an image during execution.
            "--pull",
            "never",

            # No internet or local network access.
            "--network",
            "none",

            # Resource limits.
            "--memory",
            "128m",
            "--memory-swap",
            "128m",
            "--cpus",
            "0.5",
            "--pids-limit",
            "32",

            # Limit open files and generated processes.
            "--ulimit",
            "nofile=64:64",
            "--ulimit",
            "nproc=32:32",

            # Remove container privileges.
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",

            # Run as the unprivileged nobody user.
            "--user",
            "65534:65534",

            # Prevent modification of the container filesystem.
            "--read-only",

            # Small writable temporary filesystem.
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16m",

            # Mount only the submitted script.
            # The host directory is read-only inside the container.
            "--mount",
            (
                f"type=bind,"
                f"source={sandbox_directory.resolve()},"
                f"target=/sandbox,"
                f"readonly"
            ),

            "--workdir",
            "/sandbox",

            # Avoid writing Python cache files.
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "HOME=/tmp",

            docker_image,
            "python",
            "-I",
            "/sandbox/submitted_code.py",
        ]

        try:
            process = subprocess.Popen(
                docker_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            stdout, stderr = process.communicate(
                timeout=timeout_seconds
            )

        except subprocess.TimeoutExpired:
            try:
                subprocess.run(
                    [
                        "docker",
                        "rm",
                        "-f",
                        container_name,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
            except Exception:
                pass

            execution_time = time.perf_counter() - started_at

            return {
                "success": False,
                "output": "",
                "error": (
                    f"Execution timed out after "
                    f"{timeout_seconds} seconds."
                ),
                "exit_code": None,
                "timed_out": True,
                "execution_time": round(
                    execution_time,
                    6,
                ),
            }

        except Exception as exc:
            try:
                subprocess.run(
                    [
                        "docker",
                        "rm",
                        "-f",
                        container_name,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3,
                    check=False,
                )
            except Exception:
                pass

            execution_time = time.perf_counter() - started_at

            return {
                "success": False,
                "output": "",
                "error": (
                    f"Failed to start the Docker sandbox: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "exit_code": None,
                "timed_out": False,
                "execution_time": round(
                    execution_time,
                    6,
                ),
            }

    execution_time = time.perf_counter() - started_at

    stdout = stdout or ""
    stderr = stderr or ""

    output_truncated = len(stdout) > max_output_size
    error_truncated = len(stderr) > max_output_size

    if output_truncated:
        stdout = (
            stdout[:max_output_size]
            + "\n[stdout truncated]"
        )

    if error_truncated:
        stderr = (
            stderr[:max_output_size]
            + "\n[stderr truncated]"
        )

    return {
        "success": process.returncode == 0,
        "output": stdout,
        "error": stderr or None,
        "exit_code": process.returncode,
        "timed_out": False,
        "execution_time": round(
            execution_time,
            6,
        ),
        "output_truncated": output_truncated,
        "error_truncated": error_truncated,
    }

@tool
def submit_final_answer(answer: str, format: str, steps_completed: list[str], evidence: list[str], calculations: list[str], unresolved_issues: list[str]) -> str:
    """
    Records the candidate answer, format, completed steps, evidence, calculations, and unresolved issues, then triggers the review process. This tool is called by the solver when it believes it has reached a solution to the GAIA task. It packages up the current state of the solution including the answer itself, the reasoning steps taken, supporting evidence, any calculations performed, and notes about unresolved issues, then transitions control to the review_answer node.
    """
    # Package the provided solution components into a structured JSON string
    submission_package = json.dumps({
        "answer": answer,
        "format": format,
        "steps_completed": steps_completed,
        "evidence": evidence,
        "calculations": calculations,
        "unresolved_issues": unresolved_issues
    })
    
    # Return a confirmation message indicating that the final answer has been submitted for review.
    # The requirements state: "Include the answer and format in the confirmation."
    return f"Final answer '{answer}' (format: {format}) has been submitted for review."

@tool
def think_tool(thought: str) -> str:
    """
    Overview: 
    A simple echo tool for internal reasoning that allows the LLM to record its thoughts or reasoning steps without making external calls. This tool supports the solver's internal reasoning process by providing a way to document intermediate thoughts, considerations, or self-reflections during the problem-solving process.
    
    Caller LLM: solve_task_llm
    
    Outside-the-Tool Work (Tool Handler Function Responsibilities): 
    None - The tool handler only needs to append the result as a ToolMessage to the messages state.
    
    Inside-the-Tool Work (Tool Responsibilities): 
    Simply return the provided thought string as-is, allowing the LLM to record its internal reasoning in the conversation history.
    
    Instructions: 
    1. Receive a string representing the LLM's thought or reasoning step.
    2. Return the exact same string without modification.
    3. This allows the thought to be preserved in the tool call history within the messages state for later review.
    
    State Updates (on the caller function): 
    None
    
    Args: 
    thought: str - A string representing the LLM's internal thought, reasoning step, or consideration during the solving process.
    
    Returns: 
    str - The exact same thought string that was provided as input.
    """
    return thought
# TODO: Add Tools (if needed)



''' LLM '''
analyze_task_llm = myChatOpenAI(
    temperature= 0.0
)

solve_task_llm = myChatOpenAI(
    temperature= 0.5
).bind_tools([file_parser, web_search, open_url, run_python, submit_final_answer, think_tool])

review_answer_llm = myChatOpenAI(
    temperature= 0.3
).with_structured_output(ReviewAnswerSchema)

format_output_llm = myChatOpenAI(
    temperature= 0.0
).with_structured_output(FormatOutputSchema)




''' Helpful Functions '''

# TODO: Add Helpful Functions (if needed)



''' Nodes '''
def analyze_task(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM. Produces a high-level, step-by-step plan for solving the task, identifying objectives, answer formats, files to inspect, and required tools. 
    **Overview**
    The `analyze_task` node is the first step in the GAIA task solver. It takes the raw user question, any attached file paths, and the current conversation history, and produces a concise, highÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“level plan that will guide the subsequent solver. The plan identifies the exact objective, the required finalÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“answer format, which attachments need to be inspected, the main procedural steps, and the set of tools that will likely be required.
    
    **StepÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“byÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“step**
    1. Read the following keys from `state`:
       - `messages`: the ordered conversation log.
       - `question`: the original GAIA question string.
       - `attachments`: list of file paths that may contain relevant data.
    2. Construct a prompt by formatting `prompts.ANALYZE_TASK_PROMPT` with the question, attachments, and any other contextual information.
    3. Invoke `analyze_task_llm` via `safe_invoke`, passing the prompt and the current `messages`.
    4. Parse the LLM output to extract a structured plan (JSON or plain text). The plan should be stored in `state['task_analysis']`.
    5. Append an `AIMessage` containing the plan to `state['messages']`.
    6. Return the updated state.
    
    **Possible inputs**
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the conversation history.
    - `state['question']`: `str` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the GAIA question.
    - `state['attachments']`: `List[str]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ file paths.
    
    **Possible outputs**
    - `state['task_analysis']`: `str` or `dict` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the highÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“level plan.
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ appended with the plan.
    
    **Tools**
    - None. This node is purely LLMÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“only.
    
    **Helpful functions**
    - None
    """
    print_function_name()
    try:
        # Extract state variables
        messages: List[BaseMessage] = state.get('messages', [])
        question: str = state.get('question', '')
        attachments: List[str] = state.get('attachments', [])
        
        # Prepare attachments string for prompt
        attachments_str: str = ', '.join(attachments) if attachments else 'None'
        
        # Format the prompt with question and attachments
        prompt: str = prompts.ANALYZE_TASK_PROMPT.format(
            question=question,
            attachments=attachments_str
        )
        
        # Invoke the LLM with SystemMessage and current messages
        result = safe_invoke(analyze_task_llm, messages=[SystemMessage(content=prompt)] + messages)
        
        # Extract the plan from the LLM output
        plan: str = result.content if hasattr(result, 'content') else str(result)
        
        # Clean the LLM output if needed
        plan = clean_llm_output(plan)
        
        # Store the plan in task_analysis
        # Append an AIMessage with the plan to messages
        return {
            'task_analysis': plan,
            'messages': [AIMessage(content=plan)]
        }
    
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state


def solve_task(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM+TOOLS. Executes the task by making a sequence of tool calls (file_parser, web_search, open_url, run_python) and reacts to feedback. Calls submit_final_answer to trigger review.
    **Overview**
    `solve_task` is the core execution node. It receives the plan produced by `analyze_task` and carries out the necessary steps to arrive at a candidate answer. The solver operates in a loop of LLM reasoning and tool execution, inspecting each tool result before deciding the next action. When the LLM calls `submit_final_answer`, the solver records the candidate answer and triggers review.
    
    **StepÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“byÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“step**
    1. Read the following keys from `state`:
       - `messages` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ conversation history.
       - `question` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ original GAIA question.
       - `attachments` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ file paths.
       - `task_analysis` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the plan.
       - `review_feedback` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ any feedback from a previous review (may be empty).
       - `review_count` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ number of times the answer has been reviewed.
    2. Construct a prompt by formatting `prompts.SOLVE_TASK_PROMPT` with the plan, question, attachments, and prior feedback.
    3. Invoke `solve_task_llm` via `safe_invoke`, binding the following tools:
       - `file_parser`
       - `web_search`
       - `open_url`
       - `run_python`
       - `submit_final_answer`
       - `think_tool` (a simple echo tool for internal reasoning).
    4. The LLM may propose a sequence of tool calls. For each tool call:
       - The `ToolNode` will execute the tool and append a `ToolMessage` to `state['messages']`.
       - The LLM is reÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“invoked with the updated context to decide the next action.
    5. When the LLM calls `submit_final_answer`, the tool writes the candidate answer, evidence, steps, and any unresolved issues to `state['candidate_answer']` and triggers the `review_answer` node.
    6. Append an `AIMessage` summarizing the current action to `state['messages']`.
    7. Return the updated state.
    
    **Possible inputs**
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ conversation history.
    - `state['question']`: `str` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the GAIA question.
    - `state['attachments']`: `List[str]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ file paths.
    - `state['task_analysis']`: `str` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the plan.
    - `state['review_feedback']`: `Optional[str]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ feedback from previous review.
    - `state['review_count']`: `int` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ number of review iterations.
    
    **Possible outputs**
    - `state['candidate_answer']`: `str` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the provisional answer.
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ appended with AI and Tool messages.
    - `state['review_count']`: `int` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ may be incremented when review is triggered.
    
    **Tools**
    - `file_parser`
    - `web_search`
    - `open_url`
    - `run_python`
    - `submit_final_answer`
    - `think_tool`
    
    **Helpful functions**
    - None
    """
    print_function_name()
    try:
        # Extract required state variables
        messages: List[BaseMessage] = state.get('messages', [])
        question: str = state.get('question', '')
        attachments: List[str] = state.get('attachments', [])
        task_analysis: Union[str, dict] = state.get('task_analysis', '')
        review_feedback: Optional[str] = state.get('review_feedback')
        review_count: int = state.get('review_count', 0)

        # Format the prompt with the plan, question, attachments, and prior feedback
        # Handle None review_feedback by using empty string
        prior_feedback = review_feedback if review_feedback else "No prior feedback."
        
        # Convert task_analysis to string if it's a dict
        plan_str = str(task_analysis) if task_analysis else "No plan available."
        
        # Format attachments as a readable string
        attachments_str = "\n".join(attachments) if attachments else "No attachments."

        prompt = prompts.SOLVE_TASK_PROMPT.format(
            plan=plan_str,
            question=question,
            attachments=attachments_str,
            prior_feedback=prior_feedback
        )

        # Invoke the LLM with the system prompt and current messages
        # The LLM has tools bound: file_parser, web_search, open_url, run_python, submit_final_answer, think_tool
        llm_messages = [SystemMessage(content=prompt)] + messages
        result = safe_invoke(solve_task_llm, messages=llm_messages)

        # Return the state update (LangGraph state is immutable; return a new dict)
        if result:
            return {'messages': [result]}
        return {}

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state


def review_answer(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM. Inspects the full execution history and candidate answer. Returns 'approve' or 'evise' with actionable feedback. 
    **Overview**
    After `solve_task` calls `submit_final_answer`, the `review_answer` node inspects the entire execution history and the candidate answer. It evaluates whether the answer satisfies all GAIA requirements, checks evidence, verifies calculations, ensures formatting, and determines if the answer can be approved or needs revision.
    
    **StepÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“byÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“step**
    1. Read the following keys from `state`:
       - `messages` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ all conversation, tool results, and candidate answer.
       - `candidate_answer` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the provisional answer.
       - `review_count` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ how many times the answer has been reviewed.
    2. Construct a prompt by formatting `prompts.REVIEW_ANSWER_PROMPT` with the candidate answer, messages, and any prior feedback.
    3. Invoke `review_answer_llm` via `safe_invoke`.
    4. Parse the LLM output to extract a JSON object with two fields:
       - `review_decision`: either ``"approve"`` or ``"revise"``.
       - `review_feedback`: a detailed, actionable comment if revision is required.
    5. Store these in `state['review_decision']` and `state['review_feedback']`.
    6. Append an `AIMessage` summarizing the review to `state['messages']`.
    7. Return the updated state.
    
    **Possible inputs**
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ conversation history.
    - `state['candidate_answer']`: `str` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the provisional answer.
    - `state['review_count']`: `int` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ number of reviews.
    
    **Possible outputs**
    - `state['review_decision']`: `Literal["approve", "revise"]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the review outcome.
    - `state['review_feedback']`: `Optional[str]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ feedback for revision.
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ appended with the review.
    
    **Tools**
    - None. This node is purely LLMÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“only.
    
    **Helpful functions**
    - None
    """
    print_function_name()
    try:
        # 1. Read from state
        messages = state['messages']
        candidate_answer = state['candidate_answer']
        prior_feedback = state.get('review_feedback', "")

        # 2. Convert messages to a string representation for the prompt
        # We'll extract the content of each message and join with newlines
        messages_str = "\n".join([getattr(msg, 'content', str(msg)) for msg in messages])

        # 3. Format the prompt with candidate answer, messages string, and prior feedback
        prompt = prompts.REVIEW_ANSWER_PROMPT.format(
            candidate_answer=candidate_answer,
            messages=messages_str,
            prior_feedback=prior_feedback
        )

        # 4. Invoke LLM with the formatted prompt as a SystemMessage
        # We do not pass the messages list again to avoid overloading
        result: ReviewAnswerSchema = safe_invoke(review_answer_llm, messages=[SystemMessage(content=prompt)])

        # 5. Parse LLM output (result is already structured via with_structured_output)
        review_decision = result.review_decision
        review_feedback = result.review_feedback

        # 6. Store in state
        # 7. Increment review_count
        # 8. Append AIMessage summarizing review
        summary = f"Review Decision: {review_decision}. Feedback: {review_feedback}"
        # 9. Return updated state
        return {
            'review_decision': review_decision,
            'review_feedback': review_feedback,
            'review_count': state['review_count'] + 1,
            'messages': [AIMessage(content= summary)]
        }
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state


def format_output(state: AgentSchema) -> AgentSchema:
    """ Execution: LLM. Produces the final, exact answer string required by the GAIA task, adhering to strict formatting and precision rules. 
    **Overview**
    The `format_output` node receives the final approved answer (or the best attempt after two revisions) and produces the GAIAÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“required output string. It must strip any extraneous planning or explanation, preserve exact spelling, punctuation, units, and ordering, and ensure the answer matches the specified format.
    
    **StepÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“byÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“step**
    1. Read the following keys from `state`:
       - `messages` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ full history.
       - `candidate_answer` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the current best answer.
       - `review_decision` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ should be ``"approve"`` or ``"revise"``.
       - `review_feedback` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ any comments (ignored if approved).
       - `review_count` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ number of revisions performed.
    2. Construct a prompt by formatting `prompts.FORMAT_OUTPUT_PROMPT` with the candidate answer and the required format instructions.
    3. Invoke `format_output_llm` via `safe_invoke`.
    4. Parse the LLM output to extract the final answer string.
    5. Store this string in `state['final_output']`.
    6. Append an `AIMessage` containing the formatted answer to `state['messages']`.
    7. Return the updated state.
    
    **Possible inputs**
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ conversation history.
    - `state['candidate_answer']`: `str` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the current best answer.
    - `state['review_decision']`: `Literal["approve", "revise"]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ review outcome.
    - `state['review_feedback']`: `Optional[str]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ review comments.
    - `state['review_count']`: `int` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ number of revisions.
    
    **Possible outputs**
    - `state['final_output']`: `str` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ the GAIAÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“formatted answer.
    - `state['messages']`: `List[BaseMessage]` ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ appended with the final output.
    
    **Tools**
    - None. This node is purely LLMÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‹Å“only.
    
    **Helpful functions**
    - None
    """
    print_function_name()
    try:
        # Read required state fields
        messages: List[BaseMessage] = state['messages']
        candidate_answer: str = state['candidate_answer']
        review_decision: str = state['review_decision']
        review_feedback: str = state['review_feedback'] if state['review_feedback'] is not None else  ' '
        review_count: int = state['review_count']
        answer_format: str = state['answer_format']
        
        # Format the prompt with candidate answer, answer format, and review context
        prompt: str = prompts.FORMAT_OUTPUT_PROMPT.format(
            candidate_answer=candidate_answer,
            answer_format=answer_format,
            review_decision=review_decision,
            review_feedback=review_feedback,
            review_count=review_count
        )
        
        # Invoke the LLM with the formatted prompt and current messages
        result: FormatOutputSchema = safe_invoke(
            format_output_llm,
            messages=[SystemMessage(content=prompt), *messages]
        )
        
        # Extract thinking_process and final_output from the structured output
        thinking_process: str = result.thinking_process
        final_output: str = result.final_output
        
        # Return a new state dict with the final output and appended message (LangGraph merges these)
        return {
            'final_output': final_output,
            'messages': [AIMessage(content=final_output)],
        }
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET} ', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state





''' Conditional Functions '''
def from_review_answer_to(state: AgentSchema) -> Literal["solve_task", "format_output"]:
    """Route based on review decision and count: approve -> format_output; revise with count <2 -> solve_task; otherwise format_output."""
    print_function_name()
    review_decision = state.get('review_decision', 'approve')
    review_count = state.get('review_count', 0)
    
    if review_decision == 'approve':
        return 'format_output'
    elif review_decision == 'revise' and review_count < 2:
        return 'solve_task'
    else:
        return 'format_output'
    
def from_solve_task_to(state: AgentSchema) -> Literal["solve_task_tools_type_a", "solve_task_tools_submit_final_answer", "review_answer"]:
    """ 
    Routes the workflow after solve_task based on the last AI message's tool calls.
    - If the last AI message contains tool calls for submit_final_answer, route to 'solve_task_tools_submit_final_answer'.
    - If the last AI message contains tool calls for other tools (file_parser, web_search, open_url, run_python, think_tool), route to 'solve_task_tools_type_a'.
    - If the last AI message does NOT contain any tool calls, route to 'review_answer'.
    """
    print_function_name() if DEBUG else None

    # Guard against empty messages list
    messages: List[BaseMessage] = state.get('messages', [])
    if not messages:
        return "review_answer"

    last_message: BaseMessage = messages[-1]

    # Extract tool calls - check both sources robustly
    tool_calls: list = []

    # Check direct tool_calls attribute first (LangChain canonical format)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_calls = last_message.tool_calls
    else:
        # Fall back to additional_kwargs (raw API format)
        tool_calls = last_message.additional_kwargs.get("tool_calls", []) or []

    # Extract tool names, handling both formats
    tool_names: List[str] = []
    for tc in tool_calls:
        if "name" in tc:
            # Direct format: {"name": "...", "args": ..., "id": ...}
            tool_names.append(tc.get("name"))
        elif "function" in tc:
            # additional_kwargs format: {"function": {"name": "...", "arguments": "..."}}
            tool_names.append(tc.get("function", {}).get("name"))

    # Check for submit_final_answer
    if "submit_final_answer" in tool_names:
        return "solve_task_tools_submit_final_answer"
    # Check for other tools
    elif any(name in ["file_parser", "web_search", "open_url", "run_python", "think_tool"] for name in tool_names):
        return "solve_task_tools_type_a"
    else:
        return "review_answer"



''' Tool Handlers '''
def solve_task_tools_submit_final_answer(state: AgentSchema) -> AgentSchema:
    """
    Handle an AI message containing submit_final_answer.

    This node:
    1. Finds the submit_final_answer tool call.
    2. Extracts its arguments into the agent state.
    3. Executes only submit_final_answer.
    4. Skips any other tool calls in the same AI message.
    5. Returns one ToolMessage for every requested tool call.

    Other tools are skipped because their results cannot change the answer
    that was already submitted in the same AI message.
    """
    print_function_name() if DEBUG else None

    try:
        messages: List[BaseMessage] = state.get('messages', [])
        if not messages:
            raise ValueError('Cannot handle submit_final_answer because the message history is empty.')

        last_message = messages[-1]

        # Support both LangChain's normalized tool-call format and
        # the raw OpenAI-compatible additional_kwargs format.
        if getattr(last_message, 'tool_calls', None):
            raw_tool_calls = last_message.tool_calls
        else:
            raw_tool_calls = last_message.additional_kwargs.get('tool_calls', []) or []

        if DEBUG:
            print(json.dumps(raw_tool_calls, indent=4))

        normalized_tool_calls = []

        for raw_tool_call in raw_tool_calls:
            # LangChain normalized format:
            # {
            #     "name": "submit_final_answer",
            #     "args": {...},
            #     "id": "..."
            # }
            if 'name' in raw_tool_call:
                tool_name = raw_tool_call.get('name')
                tool_args = (
                    raw_tool_call.get('args', {})
                    or raw_tool_call.get('arguments', {})
                    or {}
                )
                tool_call_id = raw_tool_call.get('id')

            # Raw OpenAI-compatible format:
            # {
            #     "id": "...",
            #     "function": {
            #         "name": "submit_final_answer",
            #         "arguments": "{...}"
            #     }
            # }
            elif 'function' in raw_tool_call:
                function_data = raw_tool_call.get('function', {})

                tool_name = function_data.get('name')
                tool_args = function_data.get('arguments', {}) or {}
                tool_call_id = raw_tool_call.get('id')

            else:
                continue

            if isinstance(tool_args, str):
                tool_args = parse_tool_arguments(tool_args)

            if not isinstance(tool_args, dict):
                tool_args = {}

            normalized_tool_calls.append({
                'name': tool_name,
                'args': tool_args,
                'id': tool_call_id,
            })

        submit_calls = [
            tool_call
            for tool_call in normalized_tool_calls
            if tool_call['name'] == 'submit_final_answer'
        ]

        if not submit_calls:
            raise ValueError('This handler was called, but no submit_final_answer tool call was found.')

        # Only one final submission should be accepted.
        # Use the first one deterministically.
        submit_call = submit_calls[0]
        submit_args = submit_call['args']

        # Execute only the submission tool.
        # Other tools in this AI message are deliberately skipped.
        confirmation = submit_final_answer.invoke(submit_args)

        tool_messages: List[ToolMessage] = []

        for tool_call in normalized_tool_calls:
            tool_name = tool_call['name']
            tool_call_id = tool_call['id']

            if tool_name == 'submit_final_answer':
                if tool_call is submit_call:
                    content = str(confirmation)
                else:
                    content = 'Skipped duplicate submit_final_answer call. Only the first submission was accepted.'

            else:
                content = (
                    f'Skipped {tool_name!r} because '
                    'submit_final_answer was called in the same response. '
                    'Run required tools before submitting the final answer.'
                )

            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=tool_name or 'unknown_tool',
                    tool_call_id=tool_call_id,
                )
            )

        return {
            'messages': tool_messages,
            'candidate_answer': submit_args.get('answer', ''),
            'answer_format': submit_args.get('format', ''),
            'steps_completed': submit_args.get(
                'steps_completed',
                []
            ),
            'evidence': submit_args.get('evidence', []),
            'calculations': submit_args.get(
                'calculations',
                []
            ),
            'unresolved_issues': submit_args.get(
                'unresolved_issues',
                []
            ),
        }

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None

        traceback.print_exc() if DEBUG else None

        return state



''' Graph '''
gaia_task_solver_graph = StateGraph(AgentSchema)

gaia_task_solver_graph.add_node("analyze_task", analyze_task)
gaia_task_solver_graph.add_node("solve_task", solve_task)
gaia_task_solver_graph.add_node("review_answer", review_answer)
gaia_task_solver_graph.add_node("format_output", format_output)
gaia_task_solver_graph.add_node("solve_task_tools_type_a", ToolNode([file_parser, web_search, open_url, run_python, think_tool]))
gaia_task_solver_graph.add_node("solve_task_tools_submit_final_answer", solve_task_tools_submit_final_answer)

gaia_task_solver_graph.add_edge(START, "analyze_task")
gaia_task_solver_graph.add_edge("analyze_task", "solve_task")
gaia_task_solver_graph.add_conditional_edges(
    "solve_task",
    from_solve_task_to,
    {   # Not needed just for clarity
        "solve_task_tools_type_a": "solve_task_tools_type_a",
        "solve_task_tools_submit_final_answer": "solve_task_tools_submit_final_answer",
        "review_answer": "review_answer",
    }
)
gaia_task_solver_graph.add_edge("solve_task_tools_type_a", "solve_task")
gaia_task_solver_graph.add_edge("solve_task_tools_submit_final_answer", "review_answer")
gaia_task_solver_graph.add_edge("review_answer", "format_output")
gaia_task_solver_graph.add_edge("format_output", END)


gaia_task_solver_app = gaia_task_solver_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(gaia_task_solver_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/gaia_task_solver_app.png', 'wb') as f:
        f.write(gaia_task_solver_app.get_graph().draw_mermaid_png())
    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'gaia_task_solver'
    os.environ['LANGSMITH_PROJECT'] = 'gaia_task_solver'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'gaia_task_solver',
            'run_name': 'gaia_task_solver',
            'thread_id': 'gaia_task_solver', 
        }
    }

    user = '' # TODO: add
    response = gaia_task_solver_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')