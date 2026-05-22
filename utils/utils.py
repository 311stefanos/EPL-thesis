from openai import APIConnectionError, InternalServerError, RateLimitError, BadRequestError, AuthenticationError
from pydantic_core._pydantic_core import ValidationError as PydanticValidationError
from json.decoder import JSONDecodeError

from langchain_core.messages import BaseMessage
from langgraph.prebuilt import tools_condition
from typing import Protocol, Any
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from time import sleep
from pathlib import Path
import inspect
import json 
import os
import re



load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent / '.env')

DEBUG = os.getenv('DEBUG')



# A constant for user approvals
USER_APPROVALS = ['y', 'ye', 'yea', 'yes', 'ok', 'okay', 'k', '', 'true', 'True']




''' Helpful General Functions '''
# Print the name of the function that is being executed
def print_function_name(colour: str= '\033[93m') -> None:
    '''
    `print_function_name` is a function that prints the name of the function that is being executed

    `Args:`
        colour (str): The colour of the text
    '''
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    filename = os.path.splitext(os.path.basename(frame.f_code.co_filename))[0]
    print(f'\n{colour}[NODE]\033[0m {filename}/{func_name}')

# Check if the last message will or should call a tool
def will_tool_call(messages: list[BaseMessage], instruction_texts: list[str] = [], actually_called: bool= False) -> bool:
    '''
    Check if the last message will call a tool.

    ### Args:
    - `messages`: the list of messages up to now
        - **note:** remember to add the last message if the state is not updated yet
    - `instruction_text`: the text that the LLM is instructed to respond with when calling a tool
    - `actually_call`: whether it actually called the tool (by **only** searching the additional kwargs and tool_calls)
        - **default**: False

    ### Returns:
    - True if the last message will call a tool

    ### Tool Calls:
    - 'Will use tavily_search to gather context'
        - Skipped if actually_call is True
    - last_message.tool_calls exists and not empty
    - last_message.additional_kwargs.tool_calls exists and not empty
    - tools_condition(last_message) == tools
    '''
    last_message = messages[-1]
    instruction_text_bool = any(instruction_text in last_message.content.lower() for instruction_text in instruction_texts)
    return (
        # If actually_called is set, we should check wheather the last message is a tool call, not the content
        instruction_text_bool and not actually_called or 
        hasattr(last_message, 'tool_calls') and last_message.tool_calls or
        hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('tool_calls', False) or
        tools_condition({'messages': messages}) == 'tools'
    )

# Function to parse tool arguments (when they come in additional_kwargs)
def parse_tool_arguments(args):
    # If the SDK already gave you a dict, use it
    if isinstance(args, dict):
        return args

    s = str(args).strip()

    # Normalize line endings
    s = s.replace('\r\n', '\n')
    # Replace any unescaped newlines with a space (JSON doesn't allow raw newlines)
    #    (?<!\\)\n  = a newline not preceded by a backslash
    s = re.sub(r'(?<!\\)\n', ' ', s)
    # Remove other control characters that are illegal in JSON
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', ' ', s)
    # Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Optional: last-resort escape of remaining bare backslashes before quote/newline
        s2 = re.sub(r'\\(?![\\/"bfnrtu])', r'\\\\', s)
        return json.loads(s2)  # will raise again if truly broken
    
# Remove heading and trailing tags or markdown special characters
def clean_llm_output(code: str) -> str:
    '''
    `clean_llm_output` removes heading and trailing tags or markdown special characters from the LLM's output

    `Args:`
        code (str): The LLM's output

    `Returns:`
        code: str
    '''
    code = code.strip()
    if not code:
        return code

    # Remove possible tags or markdown special characters from the LLM's output
    while code[0] in ['<', '`'] or code[-1] in ['>', '`']:
        # Removing tags
        while code.strip().startswith('<'):
            # Remove the line
            index = code.find('\n')
            code = code[index + 1:].strip()
            
        while code.strip().endswith('>'):
            for i, char in enumerate(reversed(code)):
                if char == '<':
                    index = len(code) - i
                    code = code[:index].strip()
                    break

        # Removing markdown ```
        while code.strip().startswith('`'):
            # Remove the line
            index = code.find('\n')
            code = code[index + 1:].strip()

        while code.strip().endswith('`'):
            for i, char in enumerate(reversed(code)):
                if char == '`':
                    index = len(code) - i - 1
                    code = code[:index].strip()
                    break

    return code.strip()

# Reads the contents of state['file_path']
def read_state_file(state) -> str:
    '''
    `read_state_file` reads the contents of state['file_path']
    
    `Args:`
        state: The state of the agent. Must have the key 'file_path'.

    `Returns:`
        code: str
    '''
    with open(state['file_path'], 'r', encoding='utf-8') as f:
        code = f.read()
    return code





''' Helpful LLM Classes/Functions '''
# A class that extends the ChatOpenAI class, that automatically inputs the base url, api keys etc
class myChatOpenAI(ChatOpenAI):
    '''
    A class that extends the ChatOpenAI class, that automatically inputs some parametres
    '''
    def __init__(
            self, 
            provider: str|None = None,
            base_url: str|None = None, 
            api_key: str|None = None,
            model: str|None = None,
            temperature: float = 0.7,
            *args,
            **kwargs
        ):
        provider = provider or os.getenv('PROVIDER')
        provider = provider.upper()
        kwargs['base_url'] = base_url or os.getenv(f'{provider}_BASE_URL')
        kwargs['api_key'] = api_key or os.getenv(f'{provider}_API_KEY')
        kwargs['model'] = model or os.getenv('MODEL_NAME')
        kwargs['temperature'] = temperature
        super().__init__(*args, **kwargs)





# An exception that is raised when the LLM cannot be reached after too many tries
class TooManyTriesException(Exception): ...

# A Protocol class that requires the `invoke` method
class Invokable(Protocol):
    '''
    A Protocol class that requires the `invoke` method
    '''
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...

# A function that invokes an LLM and handles errors
def safe_invoke(llm: Invokable, messages: list[BaseMessage], *args, retry_interval: int = 6, max_retries: int = 7, raise_pydantic= False) -> BaseMessage:
    '''
    `safe_invoke` is a function that invokes an LLM and handles errors

    `Args:`
        llm (Invokable): The LLM to invoke
        messages (list[BaseMessage]): The messages to pass to the LLM
        *args (Any): The arguments to pass to the LLM
        retry_interval (int) = 5: The number of seconds to wait between retries
        max_retries (int) = 5: The maximum number of retries
    
    `Returns:`
        (BaseMessage) The result of the LLM invocation

    `Raises:` The errors raised by the LLM
        AuthenticationError: If the API key is invalid
        other Exceptions: If the LLM returns an error that is not catched

    '''
    retry_counter = 0
    while retry_counter < max_retries:
        try:
            return llm.invoke(messages, *args)
        
        # Nothing to do, just raise the error
        except (AuthenticationError,) as e:
            raise e
        
        # Try again
        except PydanticValidationError as e:
            if raise_pydantic:
                raise e
            else:
                print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
                retry_counter += 1
                sleep(retry_interval)
        
        # Try again
        except (BadRequestError, APIConnectionError, InternalServerError, JSONDecodeError) as e:
            print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
            # Also print the reason
            print(e.response.json()['error']['message'])
            retry_counter += 1
            sleep(retry_interval)

        # Try again, with a little more handling
        except RateLimitError as e:
            error: dict = e.response.json()['error']
            # Check if it's a rate limit because of the server
            if 'is temporarily rate-limited upstream' in error.get('metadata', {}).get('raw', ''):
                cause = '(Upstream rate limit)'
                sleep_for = retry_interval

            # Or becuase of free-models-per-min
            elif 'Rate limit exceeded:' in error.get('message', '') and 'free-models-per-min' in error.get('message', ''):
                cause = '(Rate limit per minute exceeded)'
                # UFor a minute
                sleep_for = 60

            # Or because of free-models-per-day (daily quota)
            elif 'Rate limit exceeded:' in error.get('message', '') and ('free-models-per-day' in error.get('message', '') or 'per day' in error.get('message', '')):
                cause = '(Rate limit per day exceeded)'
                # For daily limits we do not want to sleep here — re-raise so the caller can handle long waits
                print(f'RateLimitError {cause}, raising to caller') if DEBUG else None
                raise e
            
            # Or fallback
            else: 
                cause = ''
                sleep_for = retry_interval
            
            print(f'RateLimitError {cause}, retrying in {sleep_for} seconds...') if DEBUG else None
            retry_counter += 1
            sleep(sleep_for)

        except ValueError as e:
            error_dict: dict = e.args[0] if e.args else None
            if isinstance(error_dict, str):
                print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
                retry_counter += 1
                sleep(retry_interval)
            elif error_dict.get('code', -1) == 500 and error_dict.get('message', '') == 'Internal Server Error':
                print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
                retry_counter += 1
                sleep(retry_interval)

        except TypeError as e:
            print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
            retry_counter += 1
            sleep(retry_interval)

        except KeyboardInterrupt as e:
            print(f'{e.__class__.__name__}, exiting...') if DEBUG else None
            exit()

        # Something went wrong, raise it
        except Exception as e:
            raise e
        
    raise TooManyTriesException(f'Could not get a response from the LLM after {max_retries} tries.')


