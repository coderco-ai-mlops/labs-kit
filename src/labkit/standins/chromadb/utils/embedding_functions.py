"""Embedding functions for the practice version of Chroma."""


def _missing(what):
    return (f"The practice version of Chroma on the labs site does not include {what}. "
            "It is in the real Chroma: see Go further in this lab.")


class EmbeddingFunction:
    """Turns a list of texts into a list of vectors. Write your own by defining __call__(self, input)."""

    def __class_getitem__(cls, item):
        return cls

    def __call__(self, input):
        raise NotImplementedError("An embedding function needs a method `__call__(self, input)` that returns one vector for each text.")


class ONNXMiniLM_L6_V2(EmbeddingFunction):
    """The model the real Chroma downloads (about 80 MB) when a collection has no embedding function."""

    def __call__(self, input):
        raise RuntimeError(
            "This collection has no embedding function, so Chroma would use its built-in model, all-MiniLM-L6-v2, "
            "which the practice version of Chroma on the labs site does not include. "
            "Pass embedding_function=... when you get the collection. This happened"
        )

    @staticmethod
    def name():
        return "default"


class DefaultEmbeddingFunction(ONNXMiniLM_L6_V2):
    def __call__(self, input):
        return ONNXMiniLM_L6_V2.__call__(self, input)


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module 'chromadb.utils.embedding_functions' has no attribute '{name}'")
    raise ImportError(_missing(f"embedding_functions.{name}"))
