"""The working parts of the practice version of BentoML: the model store, a service and its API methods."""

import base64
import functools
import os
import pickle

GO_FURTHER = "If the spelling is right, it is in the real BentoML: see Go further in this lab."


def not_included(what: str) -> str:
    return f"The practice version of BentoML on the labs site does not include {what}. {GO_FURTHER}"


def module_getattr(module: str):
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(not_included(f"`{module}.{name}`"))

    return __getattr__


class BentoMLException(Exception):
    pass


class NotFound(BentoMLException):
    pass


# ---- the model store ---------------------------------------------------------

class Tag:
    """A model's name and version, written name:version."""

    def __init__(self, name: str, version: str | None = None):
        self.name, self.version = name, version

    @classmethod
    def from_taglike(cls, value) -> "Tag":
        if isinstance(value, Tag):
            return value
        if not isinstance(value, str) or not value:
            raise BentoMLException(f"A tag is text such as \"housing_predictor:latest\". It was given {value!r}.")
        name, _, version = value.partition(":")
        return cls(name, version or None)

    def __str__(self):
        return self.name if self.version is None else f"{self.name}:{self.version}"

    def __repr__(self):
        return f"Tag(name={self.name!r}, version={self.version!r})"

    def __eq__(self, other):
        return isinstance(other, Tag) and (self.name, self.version) == (other.name, other.version)

    def __hash__(self):
        return hash((self.name, self.version))


class Model:
    """One saved model in the store."""

    def __init__(self, tag: Tag, data: bytes):
        self.tag, self._data = tag, data

    def __repr__(self):
        return f'Model(tag="{self.tag}")'


STORES: dict = {}   # store folder -> {model name -> its saved versions, oldest first}. Kept in memory for the visit.


def store() -> dict:
    """The model store in use. As in the real BentoML, the BENTOML_HOME setting names the store's folder,
    and without it the store is the folder called bentoml in the home folder. Each folder is its own store,
    so a lab that names a folder of its own never sees a model that was saved somewhere else."""
    home = os.environ.get("BENTOML_HOME") or os.path.join(os.path.expanduser("~"), "bentoml")
    return STORES.setdefault(home, {})


def save(name, model) -> Model:
    tag = Tag.from_taglike(name)
    version = tag.version or base64.b32encode(os.urandom(10)).decode("ascii").lower()
    saved = Model(Tag(tag.name, version), pickle.dumps(model))
    store().setdefault(tag.name, []).append(saved)
    return saved


def get(tag) -> Model:
    tag = Tag.from_taglike(tag)
    versions = store().get(tag.name, [])
    if not versions:
        raise NotFound(f"no Models with name '{tag.name}' exist in BentoML store")
    if tag.version in (None, "latest"):
        return versions[-1]
    for saved in versions:
        if saved.tag.version == tag.version:
            return saved
    raise NotFound(f"Model '{tag}' is not found in BentoML store")


def every() -> list:
    return [saved for versions in store().values() for saved in versions]


def load(bento_model):
    if isinstance(bento_model, BentoModel):
        # As in the real BentoML: reach it through the service, self.bento_model, which hands over the stored model.
        raise TypeError("argument of type 'BentoModel' is not iterable")
    saved = bento_model if isinstance(bento_model, Model) else get(bento_model)
    return pickle.loads(saved._data)


class BentoModel:
    """A model a service declares that it needs: `bento_model = bentoml.models.BentoModel("name:latest")`.

    Read from an instance, as self.bento_model, it is the stored model. Read from the class it is this declaration.
    """

    def __init__(self, tag):
        self.tag = Tag.from_taglike(tag)

    def resolve(self) -> Model:
        try:
            return get(self.tag)
        except NotFound:
            raise BentoMLException(f"Model {self.tag} could not be found locally and failed to fetch from remote because No cloud context default found.") from None

    def __get__(self, instance, owner=None):
        return self if instance is None else self.resolve()

    def __str__(self):
        return str(self.tag)

    def __repr__(self):
        return f"BentoModel(tag={self.tag!r})"


# ---- services ----------------------------------------------------------------

class APIMethod:
    """What @bentoml.api turns a method into. Called on a service instance, it runs the method."""

    def __init__(self, func, route: str | None = None, name: str | None = None):
        self.func = func
        self.name = name or func.__name__
        self.route = route or f"/{self.name}"
        if not self.route.startswith("/"):
            self.route = "/" + self.route
        self.doc = func.__doc__
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return functools.partial(self.func, instance)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(not_included(f"`{name}` on an API method"))

    def __repr__(self):
        return f"<APIMethod name={self.name!r} route={self.route!r}>"


_API_OPTIONS = ("input_spec", "output_spec", "batchable", "batch_dim", "max_batch_size", "max_latency_ms")


def api(func=None, *, route: str | None = None, name: str | None = None, **options):
    for option in options:
        if option in _API_OPTIONS:
            raise TypeError(not_included(f"the `{option}` option of @bentoml.api"))
        raise TypeError(f"api() got an unexpected keyword argument '{option}'")

    def wrap(method):
        if not callable(method):
            raise TypeError(f"@bentoml.api goes on a method of a service class. It was given a {type(method).__name__}.")
        return APIMethod(method, route=route, name=name)

    return wrap if func is None else wrap(func)


class Service:
    """What @bentoml.service turns a class into. Calling it builds the class, the way `bentoml serve` does once per worker."""

    def __init__(self, inner, name: str | None, config: dict):
        self.inner = inner
        self.name = name or inner.__name__
        self.config = config
        self.apis = {}
        for attribute in dir(inner):
            value = getattr(inner, attribute, None)
            if isinstance(value, APIMethod):
                self.apis[value.name] = value
        self.models = [value for value in vars(inner).values() if isinstance(value, BentoModel)]

    def __call__(self):
        return self.inner()

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        if hasattr(self.inner, name):
            raise AttributeError(f"'Service' object has no attribute '{name}'")
        raise AttributeError(not_included(f"`{name}` on a service") + f" A Service has no attribute '{name}'.")

    def __repr__(self):
        return f"<Service name={self.name!r}>"


_SERVICE_OPTIONS = ("image", "description", "path_prefix", "envs", "labels", "cmd", "service_class")


def service(inner=None, /, *, name: str | None = None, **config):
    for option in config:
        if option in _SERVICE_OPTIONS:
            raise TypeError(not_included(f"the `{option}` option of @bentoml.service"))

    def wrap(cls):
        if not isinstance(cls, type):
            raise TypeError(f"@bentoml.service goes on a class. It was given a {type(cls).__name__}.")
        return Service(cls, name, dict(config))

    return wrap if inner is None else wrap(inner)
