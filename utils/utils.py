
from openai import (OpenAIError, APIConnectionError, InternalServerError,
                    RateLimitError, BadRequestError, AuthenticationError)
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from typing import Protocol, Any
from dotenv import load_dotenv
from time import sleep, time
from pathlib import Path
import os

load_dotenv(dotenv_path= Path(__file__).resolve().parent.parent / '.env')

DEBUG = os.getenv('DEBUG')



# A class that extends the ChatOpenAI class, that automatically inputs the base url, api keys etc
class myChatOpenAI(ChatOpenAI):
    '''
    A class that extends the ChatOpenAI class, that automatically inputs some parametres
    '''
    def __init__(
            self, 
            base_url: str = 'https://openrouter.ai/api/v1', 
            api_key: str|None = None,
            model: str|None = None,
            temperature: float = 0.7,
            *args,
            **kwargs
        ):
        kwargs['base_url'] = base_url
        kwargs['api_key'] = api_key or os.getenv('OPENROUTER_API_KEY')
        kwargs['model'] = model or os.getenv('MODEL_NAME')
        kwargs['temperature'] = temperature
        super().__init__(*args, **kwargs)




# An exception that is raised when the LLM cannot be reached after too many tries
class TooManyTriesException(Exception): pass

# A Protocol class that requires the `invoke` method
class Invokable(Protocol):
    '''
    A Protocol class that requires the `invoke` method
    '''
    def invoke(self, *args: Any, **kwargs: Any) -> Any: ...

# A function that invokes an LLM and handles errors
def safe_invoke(llm: Invokable, *args, retry_interval: int = 5, max_retries: int = 5) -> BaseMessage:
    '''
    `safe_invoke` is a function that invokes an LLM and handles errors

    `Args:`
        llm (Invokable): The LLM to invoke
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
            return llm.invoke(*args)
        
        # Nothing to do, just raise the error
        except (AuthenticationError,) as e:
            raise e
        
        # Try again
        except (BadRequestError, APIConnectionError, InternalServerError) as e:
            print(f'{e.__class__.__name__}, retrying in {retry_interval} seconds...') if DEBUG else None
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
                # Until the next minute
                reset_time = int(error['metadata']['headers']['X-RateLimit-Reset'])
                sleep_for = (reset_time - int(time() * 1000)) / 1000

            # Or fallback
            else: 
                cause = ''
                sleep_for = retry_interval
            
            print(f'RateLimitError {cause}, retrying in {sleep_for} seconds...') if DEBUG else None
            retry_counter += 1
            sleep(sleep_for)

        # Something went wrong, raise it
        except Exception as e:
            raise e
        
    raise TooManyTriesException('Could not get a response from the LLM after {max_retries} tries.')


