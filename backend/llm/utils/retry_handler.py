import asyncio
from functools import wraps
from typing import Callable, Any
import random


class RetryHandler:
    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = self.initial_delay
            last_exception = None
            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt >= self.max_retries:
                        break
                    # add small jitter
                    sleep_time = delay + random.uniform(0, delay * 0.1)
                    await asyncio.sleep(sleep_time)
                    delay *= self.backoff_factor

            raise last_exception

        return wrapper
