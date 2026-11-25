"""The errors the practice version of Chroma raises. They have the same names as in the real Chroma."""


class ChromaError(Exception):
    pass


class InternalError(ChromaError):
    pass


class NotFoundError(ChromaError):
    pass


class InvalidArgumentError(ChromaError):
    pass


class DuplicateIDError(ChromaError):
    pass


class NotInPracticeVersion(AttributeError):
    """Raised for a method or attribute the practice version leaves out. The message says so in a plain sentence."""
