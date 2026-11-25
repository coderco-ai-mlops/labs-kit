"""bentoml.sklearn: save a scikit-learn model into the model store, and load it back."""

from bentoml._practice import Model, load, module_getattr, not_included, save

__all__ = ["save_model", "load_model"]


def save_model(name, model, **options) -> Model:
    """Save the model under this name as a new version, and return the stored model."""
    for option in options:
        raise TypeError(not_included(f"the `{option}` option of bentoml.sklearn.save_model"))
    return save(name, model)


def load_model(bento_model):
    """Load a stored model, ready to predict. Takes a tag such as "name:latest", or a stored model."""
    return load(bento_model)


__getattr__ = module_getattr("bentoml.sklearn")
