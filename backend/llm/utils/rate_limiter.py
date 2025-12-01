from typing import Callable, Any


def noop_rate_limiter(func: Callable) -> Callable:
	"""A no-op rate limiter decorator kept as placeholder.

	This allows callers to import a rate-limiter symbol without enforcing any limits.
	Replace with a real implementation when needed.
	"""
	def wrapper(*args: Any, **kwargs: Any):
		return func(*args, **kwargs)

	return wrapper
