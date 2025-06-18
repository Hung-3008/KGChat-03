class LLMError(Exception):
    pass

class ClientNotFoundError(LLMError):
    pass

class AuthenticationError(LLMError):
    pass