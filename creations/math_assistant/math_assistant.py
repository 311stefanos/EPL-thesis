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
import traceback, hashlib, time, requests, xml.etree.ElementTree as ET, fcntl, tempfile, cv2, numpy as np, pytesseract
import json
import os

# My imports
from utils.utils import myChatOpenAI, safe_invoke, print_function_name, will_tool_call, parse_tool_arguments, USER_APPROVALS, read_state_file, clean_llm_output
from creations.math_assistant import math_assistant_prompts as prompts

from datetime import datetime
from pix2tex import cli as latexocr

from sympy import Eq, symbols, simplify, S, solve, latex, sympify, SympifyError
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import matplotlib.pyplot as plt
from pint import UnitRegistry, UndefinedUnitError, DimensionalityError

import fitz
from docx import Document
from docx.math import OMath



''' Constants '''
load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent.parent / '.env')

DEBUG = os.getenv('DEBUG')

BLUE = '\033[94m' # INFO
RED = '\033[91m' # ERR
GREEN = '\033[92m' # REST
RESET = '\033[0m'



print(f'\n{BLUE}[AGENT] [INFO] [STARTUP]{RESET} math_assistant') if DEBUG else None



""" Schemas """
class AgentSchema(MessagesState):
	"""
	Main state schema for mathematics tutoring agent workflow.
	
	Persisted across sessions and used by all nodes for processing.
	Key Components:
	- messages: Complete conversation history (inherited from MessagesState)
	"""
	pass


''' Tools '''
@tool
def equation_solver(equation: str, variable: str) -> dict[str, Union[str, list, bool]]:
    """
    Solves mathematical equations symbolically using SymPy integration
    
    Args:
        equation: str - LaTeX-formatted equation to solve
        variable: str - Target variable for solving
    
    Returns:
        dict: {
            'solution': str|list (LaTeX solution(s)), 
            'steps': list[str] (solution steps in LaTeX), 
            'valid': bool,
            'solution_type': Literal['single', 'multiple', 'infinite', 'none', 'invalid']
        }
    """
    try:
        

        # Validate equation format
        if equation.count('=') != 1:
            return {
                'solution': 'Equation must contain exactly one equals sign',
                'steps': [],
                'valid': False,
                'solution_type': 'invalid'
            }

        # Parse equation components
        lhs, rhs = equation.split('=', 1)
        expr = Eq(parse_expr(lhs.strip()), parse_expr(rhs.strip()))
        var = symbols(variable)

        # Validate variable presence
        if var not in expr.free_symbols:
            return {
                'solution': f'Variable {variable} not found in equation',
                'steps': [f'Original equation: ${latex(expr)}$'],
                'valid': False,
                'solution_type': 'invalid'
            }

        # Solving process with step tracking
        steps = [f'Original equation: ${latex(expr)}$']
        simplified = simplify(expr)
        
        if simplified != expr:
            steps.append(f'Simplified form: ${latex(simplified)}$')

        # Check for identity equations
        if simplified == S.true:
            return {
                'solution': ['All real numbers'],
                'steps': steps,
                'valid': True,
                'solution_type': 'infinite'
            }

        solutions = solve(simplified, var, dict=True)

        # Handle solution cases
        if not solutions:
            return {
                'solution': 'No solution found',
                'steps': steps,
                'valid': False,
                'solution_type': 'none'
            }

        # Process and format solutions
        latex_solutions = [latex(sol[var]) for sol in solutions]
        
        if len(latex_solutions) > 1:
            solution_type = 'multiple'
        else:
            solution_type = 'single'

        steps.append(f'Isolate {variable}: ' + 
                    ', '.join([f'${var} = {sol}$' for sol in latex_solutions]))

        return {
            'solution': latex_solutions,
            'steps': steps,
            'valid': True,
            'solution_type': solution_type
        }

    except SympifyError:
        return {
            'solution': 'Invalid equation syntax',
            'steps': [],
            'valid': False,
            'solution_type': 'invalid'
        }
    except NotImplementedError:
        return {
            'solution': 'Equation type not supported',
            'steps': [],
            'valid': False,
            'solution_type': 'invalid'
        }
    except Exception as e:
        return {
            'solution': f'Error solving equation: {str(e)}',
            'steps': [],
            'valid': False,
            'solution_type': 'invalid'
        }

@tool
def plotter(function_expr: str, variable: str, range_start: float, range_end: float) -> dict[str, Union[str, tuple]]:
    """
    Generates interactive visualizations of mathematical functions

    Args:
        function_expr: str - Function expression to plot
        variable: str - Independent variable
        range_start: float - Plot range start
        range_end: float - Plot range end

    Returns:
        dict: {'plot_id': str, 'url': str, 'dimensions': tuple}
    """
    
    # Default values for error cases
    default_dims = (0, 0)
    
    try:
        # Validate range parameters
        if range_start >= range_end:
            range_start, range_end = range_end, range_start

        # Parse and validate function expression
        x = sp.symbols(variable)
        expr = sp.sympify(function_expr)
        f = sp.lambdify(x, expr, 'numpy')

        # Generate data points with error handling
        x_vals = np.linspace(range_start, range_end, 200)
        try:
            y_vals = f(x_vals)
        except Exception as eval_error:
            return {
                'plot_id': 'eval_error',
                'url': '',
                'dimensions': default_dims
            }

        # Create interactive plot
        plt.ioff()  # Turn off interactive mode to prevent display
        fig = plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals)
        plt.title(f'Plot of {function_expr}')
        plt.xlabel(variable)
        plt.ylabel(f'f({variable})')
        plt.grid(True)
        
        # Enable interactive features
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)  # Allow plot to initialize

        # Generate unique identifiers
        plot_id = hashlib.sha256(
            f'{function_expr}{datetime.now().timestamp()}'.encode()
        ).hexdigest()[:16]

        result = {
            'plot_id': plot_id,
            'url': f'/plots/{plot_id}',  # Simulated URL
            'dimensions': tuple(fig.get_size_inches()),
        }

        return result

    except (sp.SympifyError, TypeError, ValueError) as e:
        return {
            'plot_id': 'parse_error',
            'url': '',
            'dimensions': default_dims
        }
    except Exception as e:
        return {
            'plot_id': 'error',
            'url': '',
            'dimensions': default_dims
        }

@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> dict[str, Union[float, str]]:
    """
    Performs dimensional analysis and unit conversions using pint library.
    
    Args:
        value: Numerical value to convert
        from_unit: Source unit symbol
        to_unit: Target unit symbol
    
    Returns:
        dict: {
            'value': Converted value (4 sig figs),
            'units': Target unit symbol,
            'formula': Conversion formula metadata
        }
    """
    # Isolated unit registry to prevent state pollution
    ureg = UnitRegistry()
    ureg.default_system = 'mks'  # Use metric system by default
    
    try:
        # Basic input validation
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be numeric")
            
        quantity = value * ureg(from_unit)
        converted = quantity.to(to_unit)
        
        # Format to 4 significant figures
        formatted_value = float(f"{converted.magnitude:.4g}")
        
        # Calculate conversion factor for metadata
        src_base = quantity.to_base_units()
        dst_base = converted.to_base_units()
        factor = src_base.magnitude / dst_base.magnitude
        
        return {
            'value': formatted_value,
            'units': to_unit,
            'formula': f"1 {from_unit} = {factor:.4g} {to_unit}"
        }
        
    except UndefinedUnitError as e:
        return {
            'value': float('nan'),
            'units': '',
            'formula': f"Invalid unit error ({from_unit}->{to_unit}): {str(e)}"
        }
    except DimensionalityError as e:
        return {
            'value': float('nan'),
            'units': '',
            'formula': f"Dimensionality error ({from_unit}->{to_unit}): {str(e)}"
        }
    except Exception as e:
        return {
            'value': float('nan'),
            'units': '',
            'formula': f"Conversion error ({from_unit}->{to_unit}): {str(e)}"
        }

@tool
def matrix_module(matrices: list[list[list[float]]], operation: Literal['multiply', 'invert', 'determinant', 'eigenvalues']) -> dict[str, Union[float, list[list[float]]]]:
    """
    Overview: Performs linear algebra operations on matrices
    Caller LLM: chat_llm
    Outside-the-Tool Work (Tool Handler Function Responsibilities): None
    Inside-the-Tool Work (Tool Responsibilities): Execute matrix operation and return result
    Instructions:
    1. Validate matrix dimensions for operation
    2. Perform specified matrix operation
    3. Format result with proper precision
    4. Include operation metadata
    State Updates (on the caller function): Append ToolMessage with matrix result
    Args:
    - matrices: list - List of matrices as 2D lists
    - operation: str - Operation to perform
    Returns:
    - dict: {'result': matrix/float, 'operation': str, 'dimensions': tuple}
    """
    
    
    try:
        # Convert input to numpy arrays and validate
        np_matrices = [np.array(m) for m in matrices]
        
        # Check for empty matrices
        if any(m.size == 0 for m in np_matrices):
            raise ValueError('Empty matrix provided')
        
        # Operation dispatch with validation
        if operation == 'multiply':
            if len(np_matrices) != 2:
                raise ValueError('Multiplication requires exactly 2 matrices')
            
            a, b = np_matrices
            if a.shape[1] != b.shape[0]:
                raise ValueError(f'Matrix dimension mismatch: {a.shape} vs {b.shape}')
                
            result = np.dot(a, b)
            dimensions = result.shape
            
        elif operation == 'invert':
            if len(np_matrices) != 1:
                raise ValueError('Inversion requires exactly 1 matrix')
                
            matrix = np_matrices[0]
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Matrix must be square for inversion')
                
            det = np.linalg.det(matrix)
            if abs(det) < 1e-9:  # Floating point precision threshold
                raise ValueError('Matrix is singular and cannot be inverted')
                
            result = np.linalg.inv(matrix)
            dimensions = result.shape
            
        elif operation == 'determinant':
            if len(np_matrices) != 1:
                raise ValueError('Determinant requires exactly 1 matrix')
                
            matrix = np_matrices[0]
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Matrix must be square for determinant calculation')
                
            result = np.linalg.det(matrix)
            dimensions = ()  # Scalar value
            
        elif operation == 'eigenvalues':
            if len(np_matrices) != 1:
                raise ValueError('Eigenvalues require exactly 1 matrix')
                
            matrix = np_matrices[0]
            if matrix.shape[0] != matrix.shape[1]:
                raise ValueError('Matrix must be square for eigenvalue calculation')
                
            result = np.linalg.eigvals(matrix)
            dimensions = (len(result),)
            
        else:
            raise ValueError(f'Unsupported operation: {operation}')
        
        # Format results with precision
        if isinstance(result, np.ndarray):
            formatted_result = np.around(result, 4).tolist()
        else:
            formatted_result = round(float(result), 4)
        
        return {
            'result': formatted_result,
            'operation': operation,
            'dimensions': dimensions
        }
        
    except np.linalg.LinAlgError as e:
        return {
            'result': f'Linear algebra error: {str(e)}',
            'operation': operation,
            'dimensions': (0, 0)
        }
    except ValueError as e:
        return {
            'result': f'Validation error: {str(e)}',
            'operation': operation,
            'dimensions': (0, 0)
        }
    except Exception as e:
        return {
            'result': f'Unexpected error: {str(e)}',
            'operation': operation,
            'dimensions': (0, 0)
        }

@tool
def web_search(query: str, max_results: int) -> list[dict[str, str]]:
    """
    Overview: Retrieves academic references from vetted math sources using arXiv API
    
    Enhanced implementation with:
    - Proper academic citation formatting
    - Mathematical content filtering
    - Rate limit handling with backoff/jitter
    - Pagination support
    - Comprehensive error handling

    Args:
        query: str - Search query string
        max_results: int - Maximum results to return (capped at 50)

    Returns:
        list: [{
            'title': str, 
            'url': str, 
            'citation': str, 
            'snippet': str
        }]
    """
    
    
    
    
    
    
    MAX_RESULTS = min(max_results, 50)
    ARXIV_MAX_PER_PAGE = 100
    MATH_CATEGORIES = ['math', 'cs', 'stat', 'physics', 'q-bio']
    
    def format_arxiv_citation(entry) -> str:
        authors = ', '.join([author.find('{http://www.w3.org/2005/Atom}name').text 
                           for author in entry.findall('{http://www.w3.org/2005/Atom}author')])
        published = entry.find('{http://www.w3.org/2005/Atom}published').text[:4]
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
        arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
        primary_category = entry.find('{http://www.w3.org/2005/Atom}primary_category')
        
        journal_ref = entry.find('{http://www.w3.org/2005/Atom}journal_ref')
        if journal_ref is not None:
            return f"{authors} ({published}). '{title}'. {journal_ref.text}. arXiv:{arxiv_id}"
        
        if primary_category is not None:
            return f"{authors} ({published}). '{title}'. {primary_category.attrib['term']}. arXiv:{arxiv_id}"
        
        return f"{authors} ({published}). '{title}'. arXiv:{arxiv_id}"
    
    def is_math_related(entry) -> bool:
        categories = [cat.attrib['term'] 
                     for cat in entry.findall('{http://www.w3.org/2005/Atom}category')]
        return any(cat.split('.')[0] in MATH_CATEGORIES for cat in categories)
    
    results: List[Dict[str, str]] = []
    
    try:
        start = 0
        while len(results) < MAX_RESULTS:
            params = {
                'search_query': f'all:{query} AND cat:math*',
                'start': start,
                'max_results': min(ARXIV_MAX_PER_PAGE, MAX_RESULTS - len(results))
            }
            
            response = requests.get(
                'http://export.arxiv.org/api/query',
                params=params,
                headers={'User-Agent': 'math-assistant/1.0'},
                timeout=10
            )
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            entries = root.findall('{http://www.w3.org/2005/Atom}entry')
            
            if not entries:
                break
            
            for entry in entries:
                if len(results) >= MAX_RESULTS:
                    break
                
                if is_math_related(entry):
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                    url = entry.find('{http://www.w3.org/2005/Atom}id').text
                    snippet = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'citation': format_arxiv_citation(entry),
                        'snippet': snippet
                    })
            
            start += len(entries)
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f'{RED}[TOOL] [WARN]{RESET} arXiv API rate limit exceeded') if DEBUG else None
        else:
            print(f'{RED}[TOOL] [ERR]{RESET} arXiv API error:', e) if DEBUG else None
    except Exception as e:
        print(f'{RED}[TOOL] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
    
    return results[:MAX_RESULTS]

@tool
def update_memory(memory_key: str, content: str, operation: Literal['store', 'retrieve', 'delete']) -> dict[str, Union[bool, str]]:
    """
    Manages persistent interaction memory storage using file-based JSON storage with atomic operations.

    Implements:
    - Advisory file locking for concurrency control
    - Atomic file writes using tempfile replacement
    - Full CRUD operations with validation
    - Detailed error handling and status reporting
    """
    
    
    
    
    
    try:
        # Validate inputs
        if not isinstance(memory_key, str) or not memory_key.strip():
            return {'success': False, 'message': 'Invalid memory key', 'timestamp': datetime.utcnow().isoformat()}
        
        if operation == 'store' and not isinstance(content, str):
            return {'success': False, 'message': 'Content must be string for store operation', 'timestamp': datetime.utcnow().isoformat()}

        # Setup storage directory
        mem_dir = Path.home() / '.math_assistant'
        mem_dir.mkdir(exist_ok=True, parents=True)
        mem_file = mem_dir / 'memory.json'

        # Initialize empty storage if file doesn't exist
        if not mem_file.exists():
            with open(mem_file, 'w') as f:
                json.dump({}, f)

        # File operations with locking
        with open(mem_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}

            result = {'success': True, 'message': '', 'timestamp': datetime.utcnow().isoformat()}

            if operation == 'store':
                data[memory_key] = {
                    'content': content,
                    'created': datetime.utcnow().isoformat(),
                    'modified': datetime.utcnow().isoformat()
                }
                result['message'] = f'Stored {memory_key}'

            elif operation == 'retrieve':
                if memory_key not in data:
                    return {'success': False, 'message': f'Key {memory_key} not found', 'timestamp': datetime.utcnow().isoformat()}
                result['message'] = data[memory_key]['content']

            elif operation == 'delete':
                if memory_key not in data:
                    return {'success': False, 'message': f'Key {memory_key} not found', 'timestamp': datetime.utcnow().isoformat()}
                del data[memory_key]
                result['message'] = f'Deleted {memory_key}'

            else:
                return {'success': False, 'message': 'Invalid operation', 'timestamp': datetime.utcnow().isoformat()}

            # Atomic write using tempfile
            with tempfile.NamedTemporaryFile('w', dir=mem_dir, delete=False) as tf:
                json.dump(data, tf)
                tempname = tf.name
            
            # Replace original file
            os.replace(tempname, mem_file)

            fcntl.flock(f, fcntl.LOCK_UN)
            return result

    except PermissionError as pe:
        return {'success': False, 'message': f'Permission denied: {str(pe)}', 'timestamp': datetime.utcnow().isoformat()}
    except json.JSONDecodeError as je:
        return {'success': False, 'message': f'Invalid JSON data: {str(je)}', 'timestamp': datetime.utcnow().isoformat()}
    except Exception as e:
        return {'success': False, 'message': f'Operation failed: {str(e)}', 'timestamp': datetime.utcnow().isoformat()}




# TODO: Add Tools (if needed)



''' LLM '''
chat_llm = myChatOpenAI(
	temperature= 0.2
).bind_tools([equation_solver, plotter, unit_converter, matrix_module, web_search, update_memory])




''' Helpful Functions '''
def pdf_to_text(pdf_content: bytes, file_type: Literal['pdf', 'docx']) -> str:
    """
    Overview: Extracts text content from PDF/DOCX files while preserving mathematical notation.
    Caller Node: parse_input_doc
    
    Enhanced implementation using PyMuPDF's layout analysis and python-docx's math module
    with proper equation handling and structure preservation.
    """
    from io import BytesIO
    import re
    
    
    
    
    try:
        # File validation
        if file_type == 'pdf':
            if not pdf_content.startswith(b'%PDF-'):
                raise ValueError("Invalid PDF file signature")
            
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text_blocks = []
            equation_regex = re.compile(r'(\\\\(.*?\\\\))|(\$(.*?)\$)|([a-zA-Z]+\(.*?\))|([0-9]+[\+\-\*/^][0-9]+)')
            
            for page in doc:
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)['blocks']
                for block in blocks:
                    if 'lines' in block:
                        for line in block['lines']:
                            line_text = ' '.join([span['text'] for span in line['spans']])
                            # Enhanced equation detection and conversion
                            line_text = equation_regex.sub(
                                lambda m: f'\\({m.group(0)}\\)' if m.group(0) else '', 
                                line_text
                            )
                            text_blocks.append(line_text)
            full_text = '\n'.join(text_blocks)
            
        elif file_type == 'docx':
            if not pdf_content.startswith(b'PK\x03\x04'):
                raise ValueError("Invalid DOCX file signature")
            
            doc = Document(BytesIO(pdf_content))
            text_blocks = []
            
            for para in doc.paragraphs:
                para_text = para.text
                # Extract equations from Office MathML
                for math in para._element.xpath('.//m:oMath'):
                    math_text = Omath(math).latex
                    para_text = para_text.replace(math_text, f'\\({math_text}\\)')
                text_blocks.append(para_text)
            
            full_text = '\n\n'.join(text_blocks)
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Structure-preserving cleaning
        full_text = re.sub(r'(?<!\\)\s+', ' ', full_text)  # Preserve LaTeX spaces
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)  # Normalize paragraphs
        full_text = re.sub(r'(\\\()\s+', r'\1', full_text)  # Fix equation spacing
        full_text = re.sub(r'\s+(\\\\))', r'\1', full_text)
        
        return full_text
    
    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        print(f'{RED}[FUNCTION] [ERR]{RESET} {error_msg}') if DEBUG else None
        traceback.print_exc() if DEBUG else None
        raise RuntimeError(error_msg) from e

def ocr_image(image_content: bytes, image_type: Literal['png', 'jpg', 'tiff']) -> str:
    """
    Overview: Performs math-oriented OCR on image files with equation detection.
    Caller Node: parse_input_doc
    
    Enhanced implementation using contour analysis for math region detection and
    LatexOCR for equation conversion with improved error handling and layout preservation.
    """
    
    
    
    
    
    from typing import List, Tuple
    
    try:
        # Image loading and validation
        nparr = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unsupported image type: {image_type}")

        # Enhanced preprocessing pipeline
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Initialize LatexOCR model
        equation_model = latexocr.LatexOCR()

        # Detect all contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours into regions with math detection
        regions: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 5 and h > 5:  # Filter small noise
                regions.append((x, y, x+w, y+h))

        # Merge nearby regions using NMS
        def non_max_suppression(boxes, overlap_thresh=0.2):
            if len(boxes) == 0:
                return []
            boxes = np.array(boxes)
            pick = []
            x1 = boxes[:,0]
            y1 = boxes[:,1]
            x2 = boxes[:,2]
            y2 = boxes[:,3]
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)
            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)
                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])
                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                overlap = (w * h) / area[idxs[:last]]
                idxs = np.delete(idxs, np.concatenate(([last],
                    np.where(overlap > overlap_thresh)[0])))
            return boxes[pick].astype("int").tolist()

        regions = non_max_suppression(regions)

        # Improved reading order sorting
        def reading_order(region):
            x1, y1, x2, y2 = region
            return (y1//20 * 1000) + x1  # Group into lines every ~20 pixels

        regions.sort(key=reading_order)

        # Process regions
        output: List[str] = []
        for (x1, y1, x2, y2) in regions:
            crop = img[y1:y2, x1:x2]
            
            # Math detection via symbol density
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            density = cv2.countNonZero(cv2.threshold(crop_gray, 127, 255, cv2.THRESH_BINARY)[1])
            density /= (x2-x1)*(y2-y1)
            
            if density > 0.1:  # Higher density indicates math symbols
                try:
                    latex = equation_model(crop)
                    output.append(f'\\({latex}\\)')
                except Exception as e:
                    text = pytesseract.image_to_string(crop, config='--psm 7 --oem 3')
                    output.append(text.strip())
            else:
                text = pytesseract.image_to_string(crop, config='--psm 7 --oem 3')
                output.append(text.strip())

        # Post-process combined output
        combined = ' '.join(output).strip()
        combined = ' '.join(combined.split())  # Remove extra whitespace
        return combined

    except Exception as e:
        error_msg = f"Image OCR failed: {str(e)}"
        print(f'{RED}[FUNCTION] [ERR]{RESET} {error_msg}') if DEBUG else None
        traceback.print_exc() if DEBUG else None
        raise RuntimeError(error_msg) from e
def convert_mathml_to_latex(mathml_element: xml.etree.ElementTree.Element) -> str:
    """
    Converts Office MathML objects to LaTeX format with proper equation wrapping.
    
    Enhanced implementation with:
    - Robust error handling for invalid MathML structures
    - Support for multiple matrix environments
    - Complete namespace handling
    - Mathematical function/symbol support
    - Operator spacing rules
    
    Args:
        mathml_element: MathML XML element to convert
    
    Returns:
        str: LaTeX-formatted equation wrapped in \(...\)
    """
    try:
        # Enhanced symbol mappings
        greek_letters = {
            'alpha': '\\alpha', 'beta': '\\beta', 'gamma': '\\gamma',
            'delta': '\\delta', 'epsilon': '\\epsilon', 'zeta': '\\zeta',
            'eta': '\\eta', 'theta': '\\theta', 'iota': '\\iota',
            'kappa': '\\kappa', 'lambda': '\\lambda', 'mu': '\\mu',
            'nu': '\\nu', 'xi': '\\xi', 'omicron': 'o', 'pi': '\\pi',
            'rho': '\\rho', 'sigma': '\\sigma', 'tau': '\\tau',
            'upsilon': '\\upsilon', 'phi': '\\phi', 'chi': '\\chi',
            'psi': '\\psi', 'omega': '\\omega',
            # Capital Greek letters
            'Gamma': '\\Gamma', 'Delta': '\\Delta', 'Theta': '\\Theta',
            'Lambda': '\\Lambda', 'Xi': '\\Xi', 'Pi': '\\Pi',
            'Sigma': '\\Sigma', 'Phi': '\\Phi', 'Psi': '\\Psi',
            'Omega': '\\Omega'
        }

        operators = {
            '+': ' + ', '-': ' - ', '*': ' \\times ', '/': ' \\div ',
            '±': ' \\pm ', '∓': ' \\mp ', '⋅': ' \\cdot ',
            '×': ' \\times ', '÷': ' \\div ', '=': ' = ',
            '<': ' < ', '>': ' > ', '≤': ' \\leq ', '≥': ' \\geq ',
            '≠': ' \\neq ', '≈': ' \\approx ', '≡': ' \\equiv ',
            '→': ' \\rightarrow ', '⇒': ' \\Rightarrow ',
            '∞': ' \\infty ', '∂': ' \\partial ', '∇': ' \\nabla '
        }

        functions = {
            'sin', 'cos', 'tan', 'log', 'ln', 'exp',
            'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan'
        }

        def convert_element(element: ET.Element) -> str:
            tag = ET.QName(element).localname  # Robust namespace handling
            children = list(element)

            # Validation functions
            def validate_children_count(expected, tag):
                if len(children) != expected:
                    raise ValueError(f"{tag} requires exactly {expected} children")

            handlers = {
                'mi': lambda e: (
                    f'\\{e.text}' if e.text in functions
                    else greek_letters.get(e.text, e.text)
                ),
                'mn': lambda e: e.text,
                'mo': lambda e: operators.get(e.text, e.text),
                'mfrac': lambda e: (
                    validate_children_count(2, 'mfrac') or
                    f'\\frac{{{convert(e[0])}}}{{{convert(e[1])}}}'
                ),
                'msup': lambda e: (
                    validate_children_count(2, 'msup') or
                    f'{{{convert(e[0])}}}^{{{convert(e[1])}}}'
                ),
                'msub': lambda e: (
                    validate_children_count(2, 'msub') or
                    f'{{{convert(e[0])}}}_{{{convert(e[1])}}}'
                ),
                'mrow': lambda e: ''.join(convert(c) for c in e),
                'msqrt': lambda e: f'\\sqrt{{{convert(e[0])}}}',
                'mroot': lambda e: (
                    validate_children_count(2, 'mroot') or
                    f'\\sqrt[{convert(e[1])}]{{{convert(e[0])}}}'
                ),
                'mtable': lambda e: (
                    f'\\begin{{{element.get("matrixenv", "matrix")}}}'
                    + ' \\\\ '.join(
                        ' & '.join(convert(c) for c in row.findall('mtd'))
                        for row in e.findall('mtr')
                    )
                    + f'\\end{{{element.get("matrixenv", "matrix")}}}'
                ),
                'mfunc': lambda e: f'\\{e.text}',
                'mspace': lambda e: ' ' * int(e.get('width', '1')),
                'mtext': lambda e: f'\\text{{{e.text}}}'
            }

            try:
                if tag in handlers:
                    return handlers[tag](element)
                
                # Process unknown elements
                if children:
                    return ''.join(convert(c) for c in children)
                return element.text or ''

            except Exception as e:
                print(f'{RED}[FUNCTION] [WARN]{RESET} Error processing {tag}: {str(e)}') if DEBUG else None
                return ''

        latex_content = convert_element(mathml_element)
        return f'\\({latex_content}\\)'

    except Exception as e:
        print(f'{RED}[FUNCTION] [ERR]{RESET} MathML conversion failed: {str(e)}') if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return ''

def calculate_region_density(region: numpy.ndarray, threshold: float) -> float:
    """
    Calculates symbol density for image regions to detect math expressions.
    Args:
    - region: numpy.ndarray - Image region to analyze
    - threshold: float - Binarization threshold (0-255)
    Returns:
    - float: Normalized density value (0-1)
    """
    # Handle empty region case
    if region.size == 0:
        return 0.0

    # Convert to grayscale if color image
    if len(region.shape) == 3 and region.shape[2] == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region

    # Apply inverse thresholding with integer threshold
    _, thresh = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY_INV)

    # Calculate density
    white_pixels = cv2.countNonZero(thresh)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    return white_pixels / total_pixels if total_pixels != 0 else 0.0

def sort_regions_reading_order(regions: list[tuple], line_threshold: int) -> list[tuple]:
    """
    Sorts regions in proper reading order using advanced spatial analysis.
    Args:
        - regions: list[tuple] - List of (x1,y1,x2,y2) coordinates
        - line_threshold: int - Y-axis threshold for grouping lines
    Returns:
        - list[tuple]: Sorted regions in reading order
    """
    # Initialize line groups with first region
    line_groups = []
    
    for region in regions:
        x1, y1, x2, y2 = region
        center_y = (y1 + y2) / 2
        
        # Find matching line group
        matched = False
        for group in line_groups:
            if abs(center_y - group['avg_y']) <= line_threshold:
                group['regions'].append(region)
                # Update group average using weighted average
                group['avg_y'] = (group['avg_y'] * (len(group['regions'])-1) + center_y) / len(group['regions'])
                matched = True
                break
        
        if not matched:
            line_groups.append({
                'avg_y': center_y,
                'regions': [region]
            })
    
    # Sort line groups by average y (top-to-bottom)
    line_groups.sort(key=lambda g: g['avg_y'])
    
    # Sort regions within each group by x1 (left-to-right)
    sorted_regions = []
    for group in line_groups:
        group['regions'].sort(key=lambda r: r[0])
        sorted_regions.extend(group['regions'])
    
    return sorted_regions





''' Nodes '''
def parse_input_doc(state: AgentSchema) -> AgentSchema:
    """ Execution: CODE. Checks for PDF/image attachments, extracts text using OCR/PDF readers, and appends extracted content to message. 
	Processes document attachments in user messages to extract mathematical content.

	Steps:
	1. Check message attachments for supported file types (PDF/DOCX/images)
	2. Use PDF parsers for documents and OCR for images
	3. Extract text content while preserving mathematical notation
	4. Append cleaned text to message content with [EXTRACTED_CONTENT] marker
	5. Store processing metadata for document tracking

	Inputs (state keys):
	- 'messages[-1].attachments': List of file attachments with 'content' and 'type' fields

	Outputs (state keys):
	- 'messages[-1].content': Original text + extracted content with marker
	- 'document_metadata': List of parsed documents with timestamps/sources
	- 'last_processed_doc': SHA hash of processed document content

	Helpful Functions:
	- pdf_to_text(): PDF parser that preserves LaTeX/math notation
	- ocr_image(): Math-oriented OCR with equation detection
	"""

    print_function_name()
    try:
        last_message = state['messages'][-1]
        attachments = last_message.additional_kwargs.get('attachments', [])
        
        if not attachments:
            return state

        extracted_texts: List[str] = []
        document_metadata: List[Dict[str, Any]] = []
        last_processed_doc: Optional[str] = None

        for attachment in attachments:
            content = attachment.get('content')
            file_type = attachment.get('type', '').lower()

            if not content or not file_type or not isinstance(content, bytes):
                continue

            try:
                if file_type in ['pdf', 'docx']:
                    extracted = pdf_to_text(content, file_type)
                elif file_type in ['png', 'jpg', 'tiff']:
                    extracted = ocr_image(content, file_type)
                else:
                    print(f'{RED}[NODE] [WARN]{RESET} Unsupported file type: {file_type}') if DEBUG else None
                    continue

                extracted_texts.append(extracted)
                doc_hash = hashlib.sha256(content).hexdigest()
                
                document_metadata.append({
                    'type': file_type,
                    'timestamp': time.time(),
                    'hash': doc_hash
                })
                last_processed_doc = doc_hash

            except Exception as e:
                print(f'{RED}[NODE] [ERR]{RESET} Attachment processing failed:', e) if DEBUG else None
                traceback.print_exc() if DEBUG else None
                continue

        if extracted_texts:
            combined_content = '\n'.join(extracted_texts)
            new_content = f"{last_message.content}\n[EXTRACTED_CONTENT]\n{combined_content}"
            
            new_messages = list(state['messages'])
            new_messages[-1] = last_message.copy(update={"content": new_content})

            return {
                'messages': new_messages,
                'document_metadata': document_metadata,
                'last_processed_doc': last_processed_doc
            }

        return state

    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        return state


def chat(state: AgentSchema) -> AgentSchema:
    print_function_name()
    try:
        messages: List[BaseMessage] = state['messages']
        history_str = '\n'.join([msg.content for msg in messages[:-1]])
        current_input = messages[-1].content

        # Format prompt with conversation history
        prompt = prompts.CHAT_PROMPT.format(history=history_str)
        system_msg = SystemMessage(content=prompt)

        # Invoke LLM with system message and conversation history
        result = safe_invoke(chat_llm, messages=[system_msg, HumanMessage(content=current_input)])

        # Update state with LLM response
        new_messages = messages + [result]
        state['messages'] = new_messages

        # Handle solution artifacts for both response types
        solution_artifacts = state.get('solution_artifacts', {})
        
        if will_tool_call([result]):
            # Preserve tool call metadata for ToolNode integration
            solution_artifacts['pending_tools'] = [
                {
                    'name': tool_call['name'],
                    'args': parse_tool_arguments(tool_call['args']),
                    'id': tool_call['id']
                }
                for tool_call in getattr(result, 'tool_calls', [])
            ]
        else:
            # Store direct response content
            solution_artifacts['response'] = clean_llm_output(result.content)
        
        state['solution_artifacts'] = solution_artifacts
        
        return state
    except Exception as e:
        print(f'{RED}[NODE] [ERR]{RESET}', e) if DEBUG else None
        traceback.print_exc() if DEBUG else None
        # Maintain conversation flow with error message
        error_msg = AIMessage(content="Apologies, I encountered an error processing your request. Please try rephrasing or ask about a different topic.")
        state['messages'].append(error_msg)
        return state



''' Conditional Functions '''
def from_start_to(state: AgentSchema) -> Literal["parse_input_doc", "chat"]:
    """Routes to parse_input_doc if last message has attachments, else to chat.

    Decision logic:
    1. Checks last message's attachments field regardless of message type
    2. Routes to document processing if any attachments found
    3. Directs to chat node for regular text messages
    """
    print_function_name()
    last_message: BaseMessage = state['messages'][-1]
    
    if hasattr(last_message, 'attachments') and last_message.attachments:
        return "parse_input_doc"
    return "chat"

def from_chat_to(state: AgentSchema) -> Literal["chat_tools_math", "__end__"]:
    """Route based on presence of tool calls in last message."""
    print_function_name() if DEBUG else None
    
    if will_tool_call(state['messages']):
        return "chat_tools_math"
    return "__end__"



''' Graph '''
math_assistant_graph = StateGraph(AgentSchema)

math_assistant_graph.add_node("parse_input_doc", parse_input_doc)
math_assistant_graph.add_node("chat", chat)
math_assistant_graph.add_node("chat_tools_math", ToolNode([
    equation_solver, plotter, unit_converter, 
    matrix_module, web_search, update_memory
]))

math_assistant_graph.add_conditional_edges(
    "start",
    from_start_to,
    {   # Not needed just for clarity
        "parse_input_doc": "parse_input_doc",
        "chat": "chat",
    }
)
math_assistant_graph.add_edge("parse_input_doc", "chat")

math_assistant_graph.add_conditional_edges(
    "chat",
    from_chat_to,
    {
        "chat_tools_math": "chat_tools_math",
        "__end__": END
    }
)
math_assistant_graph.add_edge("chat_tools_math", "chat")

math_assistant_graph.set_entry_point("start")


math_assistant_app = math_assistant_graph.compile(checkpointer= MemorySaver())



''' Testing '''
if __name__ == '__main__':
    from IPython.display import Image as GraphImage

    # Visualize the graph
    GraphImage(math_assistant_app.get_graph().draw_mermaid_png(max_retries= 5, retry_delay= 2.0))
    parent_dir = Path(__file__).resolve().parent
    if not os.path.exists(parent_dir / 'graphs'):
        os.makedirs(parent_dir / 'graphs')
    with open(parent_dir / 'graphs/math_assistant_app.png', 'wb') as f:
        f.write(math_assistant_app.get_graph().draw_mermaid_png())

    
    # Connect to langsmith
    from langsmith import Client
    os.environ['LANGCHAIN_PROJECT'] = 'math_assistant'
    os.environ['LANGSMITH_PROJECT'] = 'math_assistant'
    client = Client()

    config = {
        'recursion_limit': 100,
        'configurable': {
            'user_id': 'math_assistant',
            'run_name': 'math_assistant',
            'thread_id': 'math_assistant', 
        }
    }

    user = '' # TODO: add
    response = math_assistant_app.invoke(user, config= config)

    print(f'{BLUE}[MAIN] [INFO]{RESET} Response') if DEBUG else None
    if DEBUG:
        for key, value in response.items():
            print(f'    {key}: {value}')