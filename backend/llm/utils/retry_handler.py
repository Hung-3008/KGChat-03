# backend/llm/utils/retry_handler.py

import asyncio
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

class RetryHandler:
    """
    A flexible decorator for retrying asynchronous functions with exponential backoff.
    
    This decorator can be applied to any async function to make it resilient
    to transient errors like network issues or temporary API unavailability.
    """
    
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0, jitter: bool = True):
        """
        Initializes the RetryHandler decorator.

        Args:
            max_retries: The maximum number of times to retry the function.
            initial_delay: The delay in seconds before the first retry.
            backoff_factor: The factor by which the delay increases for each subsequent retry.
            jitter: If True, adds a small random amount to the delay to prevent
                    multiple clients from retrying in sync (thundering herd problem).
        """
        if max_retries < 0:
            raise ValueError("max_retries must be a non-negative number.")
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def __call__(self, func: Callable) -> Callable:
        """
        Makes the class instance callable, allowing it to be used as a decorator.
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """The wrapper that implements the retry logic."""
            delay = self.initial_delay
            last_exception = None
            
            # The total number of attempts will be max_retries + 1 (the initial attempt)
            for attempt in range(self.max_retries + 1):
                try:
                    # Await the actual coroutine
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt >= self.max_retries:
                        # If we've exhausted all retries, break the loop and re-raise
                        break

                    logger.warning(
                        f"Attempt {attempt + 1} of {self.max_retries + 1} failed for {func.__name__}. "
                        f"Error: {e}. Retrying in {delay:.2f} seconds..."
                    )
                    
                    # Calculate sleep time with optional jitter
                    sleep_time = delay
                    if self.jitter:
                        sleep_time += asyncio.get_event_loop().run_in_executor(None, lambda: __import__('random').uniform(0, delay * 0.1))
                    
                    await asyncio.sleep(sleep_time)
                    
                    # Increase delay for the next potential retry
                    delay *= self.backoff_factor
            
            # If the loop finished without returning, it means all retries failed.
            logger.error(f"Function {func.__name__} failed after {self.max_retries} retries.")
            raise last_exception from last_exception

        return wrapper
