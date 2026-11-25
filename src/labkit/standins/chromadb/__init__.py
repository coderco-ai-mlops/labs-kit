"""A practice version of Chroma for the labs site, where the real chromadb cannot be installed.

It covers what lab s5-07b uses: Client and PersistentClient, collections with an embedding
function, add, query, get, count, upsert, update, delete, and data that is saved in a folder.
Searches compare the question with every stored document, which is exact and fast enough
for a lab. The real Chroma uses an index so that it stays fast with millions of documents.
"""

import json
import math
import os
import re
import sys

from chromadb import errors, utils
from chromadb.errors import DuplicateIDError, InternalError, InvalidArgumentError, NotFoundError, NotInPracticeVersion
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction, EmbeddingFunction

__version__ = "1.5.9"
_FILE = "chroma.json"          # the real Chroma saves to a file called chroma.sqlite3
_MEMORY = {"collections": {}}  # every chromadb.Client() in one program shares this, as in the real Chroma
_OPEN: dict = {}               # folder -> the data loaded from it
_WHERE_OPS = ("$gt", "$gte", "$lt", "$lte", "$ne", "$eq", "$in", "$nin")
_SPACES = ("l2", "cosine", "ip")


def _missing(what):
    return (f"The practice version of Chroma on the labs site does not include {what}. "
            "It is in the real Chroma: see Go further in this lab.")


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(f"module 'chromadb' has no attribute '{name}'")
    raise ImportError(_missing(f"chromadb.{name}"))   # ImportError, so the sentence also survives `from chromadb import x`


class _MissingModules:
    """Asked last, when Python cannot find a module such as `chromadb.config`: says so in a plain sentence."""

    package = "chromadb"

    def find_spec(self, name, path=None, target=None):
        if name.startswith("chromadb."):
            raise ImportError(_missing(name))
        return None


if not any(getattr(finder, "package", None) == "chromadb" for finder in sys.meta_path):
    sys.meta_path.append(_MissingModules())


def _ef_name(function):
    name = getattr(function, "name", None)
    return name() if callable(name) else type(function).__name__


def _listed(value, what, method, kinds=str):
    """Chroma accepts one item or a list of items. Always hand back a list."""
    if value is None:
        return None
    items = [value] if isinstance(value, kinds) else list(value)
    if what == "ids":
        for item in items:
            if not isinstance(item, str):
                raise ValueError(f"Expected ID to be a str, got {item!r} in {method}.")
    return items


def _matches(metadata, where, method):
    if where is None:
        return True
    if not isinstance(where, dict) or len(where) != 1:
        raise ValueError(f"Expected where to have exactly one operator, got {where} in {method}.")
    (key, want), = where.items()
    if key in ("$and", "$or"):
        results = [_matches(metadata, part, method) for part in want]
        return all(results) if key == "$and" else any(results)
    have = (metadata or {}).get(key)
    if not isinstance(want, dict):
        return have == want
    for op, value in want.items():
        if op not in _WHERE_OPS:
            raise ValueError(f"Expected where operator to be one of $gt, $gte, $lt, $lte, $ne, $eq, $in, $nin, $contains, $not_contains, got {op} in {method}.")
        if op == "$ne":
            ok = have != value
        elif op == "$nin":
            ok = have not in value
        elif have is None:
            ok = False
        else:
            ok = {"$eq": lambda: have == value, "$in": lambda: have in value, "$gt": lambda: have > value,
                  "$gte": lambda: have >= value, "$lt": lambda: have < value, "$lte": lambda: have <= value}[op]()
        if not ok:
            return False
    return True


def _distance(space, a, b):
    if len(a) != len(b):
        raise InvalidArgumentError(f"Collection expecting embedding with dimension of {len(b)}, got {len(a)}")
    dot = sum(x * y for x, y in zip(a, b))
    if space == "ip":
        return 1.0 - dot
    if space == "cosine":
        size = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
        return 1.0 - dot / size if size else 1.0
    return sum((x - y) ** 2 for x, y in zip(a, b))   # "l2", the default: the squared distance


class Collection:
    """One named set of documents, like one VectorStore from lab s5-07."""

    def __init__(self, client, name, embedding_function):
        self._client, self.name, self._embedding_function = client, name, embedding_function
        self._uid = self._data()["uid"]

    def __repr__(self):
        return f"Collection(name={self.name})"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotInPracticeVersion(_missing(f"collection.{name}"))

    def _data(self):
        found = self._client._store["collections"].get(self.name)
        if found is None or found["uid"] != getattr(self, "_uid", found["uid"]):
            raise NotFoundError(f"Collection [{self.name}] does not exist.")
        return found

    @property
    def metadata(self):
        return self._data()["metadata"]

    def _embed(self, texts, method):
        function = self._embedding_function or DefaultEmbeddingFunction()
        try:
            vectors = function(list(texts))
        except Exception as err:             # like the real Chroma, say which call the embedding function failed in
            raise type(err)(f"{err} in {method}.") from None
        try:
            vectors = [[float(x) for x in vector] for vector in vectors]
        except (TypeError, ValueError):
            vectors = None
        if vectors is None or len(vectors) != len(texts):
            raise ValueError(f"Expected the embedding function to return one list of numbers for each text. For {len(texts)} text(s) it returned {vectors if vectors is not None else 'something else'!r:.80} in {method}.")
        return vectors

    def _records(self, method, ids, documents, metadatas, embeddings):
        ids = _listed(ids, "ids", method)
        documents = _listed(documents, "documents", method)
        metadatas = _listed(metadatas, "metadatas", method, dict)
        if embeddings is not None and embeddings and not isinstance(embeddings[0], (list, tuple)):
            embeddings = [embeddings]
        if documents is None and embeddings is None and (method != "update" or metadatas is None):
            raise ValueError(f"You must provide either embeddings or documents in {method}.")
        sizes = {"ids": ids, "embeddings": embeddings, "metadatas": metadatas, "documents": documents}
        lengths = {key: len(value) for key, value in sizes.items() if value is not None}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Unequal lengths for fields: {', '.join(f'{k}: {v}' for k, v in lengths.items())} in {method}.")
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise DuplicateIDError(f"Expected IDs to be unique, found duplicates of: {', '.join(repeated)} in {method}.")
        if embeddings is None and documents is not None:
            embeddings = self._embed(documents, method)
        blank = [None] * len(ids)
        return list(zip(ids, documents or blank, metadatas or blank, embeddings or blank))

    def _write(self, method, ids, documents, metadatas, embeddings):
        rows = self._data()["rows"]
        place = {row[0]: index for index, row in enumerate(rows)}
        for doc_id, document, metadata, embedding in self._records(method, ids, documents, metadatas, embeddings):
            if doc_id not in place:
                if method != "update":
                    rows.append([doc_id, document, metadata, embedding])
            elif method != "add":           # add leaves an id it already holds exactly as it was
                old = rows[place[doc_id]]
                rows[place[doc_id]] = [doc_id, old[1] if document is None else document,
                                       old[2] if metadata is None else metadata, old[3] if embedding is None else embedding]
        self._client._save()

    def add(self, ids, embeddings=None, metadatas=None, documents=None):
        self._write("add", ids, documents, metadatas, embeddings)

    def upsert(self, ids, embeddings=None, metadatas=None, documents=None):
        self._write("upsert", ids, documents, metadatas, embeddings)

    def update(self, ids, embeddings=None, metadatas=None, documents=None):
        self._write("update", ids, documents, metadatas, embeddings)

    def count(self):
        return len(self._data()["rows"])

    def _result(self, rows, include, default, wrap=False, distances=None):
        include = list(default if include is None else include)
        for item in include:
            if item not in ("documents", "metadatas", "embeddings", "distances"):
                raise ValueError(_missing(f"include={item!r}"))
        columns = {"documents": [row[1] for row in rows], "metadatas": [row[2] for row in rows], "embeddings": [row[3] for row in rows]}
        result = {"ids": [row[0] for row in rows], "embeddings": None, "documents": None, "uris": None, "included": include, "data": None, "metadatas": None}
        if distances is not None:
            result["distances"] = distances if "distances" in include else None
        for key in ("documents", "metadatas", "embeddings"):
            if key in include:
                result[key] = columns[key]
        return result

    def get(self, ids=None, where=None, limit=None, offset=None, where_document=None, include=None):
        if where_document is not None:
            raise ValueError(_missing("where_document"))
        ids = _listed(ids, "ids", "get")
        rows = [row for row in self._data()["rows"] if (ids is None or row[0] in ids) and _matches(row[2], where, "get")]
        rows = rows[offset or 0:]
        return self._result(rows if limit is None else rows[:limit], include, ("metadatas", "documents"))

    def peek(self, limit=10):
        return self.get(limit=limit)

    def query(self, query_embeddings=None, query_texts=None, n_results=10, where=None, where_document=None, include=None, ids=None):
        if where_document is not None:
            raise ValueError(_missing("where_document"))
        if query_embeddings is None and query_texts is None:
            raise ValueError("At least one of one of embeddings, documents, images, uris must be provided in query.")
        if not isinstance(n_results, int) or isinstance(n_results, bool) or n_results <= 0:
            raise TypeError(f"Number of requested results {n_results}, cannot be negative, or zero. in query.")
        if query_embeddings is None:
            query_embeddings = self._embed(_listed(query_texts, "query_texts", "query"), "query")
        elif query_embeddings and not isinstance(query_embeddings[0], (list, tuple)):
            query_embeddings = [query_embeddings]
        data = self._data()
        ids = _listed(ids, "ids", "query")
        rows = [row for row in data["rows"] if row[3] is not None and (ids is None or row[0] in ids) and _matches(row[2], where, "query")]
        answers = []
        for vector in query_embeddings:
            scored = sorted(((_distance(data["space"], vector, row[3]), row) for row in rows), key=lambda pair: pair[0])[:n_results]
            answers.append(self._result([row for _, row in scored], include, ("metadatas", "documents", "distances"), distances=[d for d, _ in scored]))
        merged = dict(answers[0]) if answers else self._result([], include, ("metadatas", "documents", "distances"), distances=[])
        for key in ("ids", "documents", "metadatas", "embeddings", "distances"):   # one inner list for each question
            merged[key] = None if merged[key] is None else [answer[key] for answer in answers]
        return merged

    def delete(self, ids=None, where=None, where_document=None):
        if where_document is not None:
            raise ValueError(_missing("where_document"))
        if ids is None and where is None:
            raise ValueError("At least one of ids, where, or where_document must be provided in delete.")
        ids = _listed(ids, "ids", "delete")
        data = self._data()
        keep = [row for row in data["rows"] if not ((ids is None or row[0] in ids) and _matches(row[2], where, "delete"))]
        removed = len(data["rows"]) - len(keep)
        data["rows"][:] = keep
        self._client._save()
        return {"deleted": removed}


class _Client:
    _store: dict = _MEMORY
    _uids = 0

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotInPracticeVersion(_missing(f"client.{name}"))

    def _save(self):
        pass

    def _open(self, name, embedding_function):
        saved = self._store["collections"][name]["ef"]
        if embedding_function is not None and _ef_name(embedding_function) != saved:
            raise ValueError("An embedding function already exists in the collection configuration, and a new one is provided. "
                             "If this is intentional, please embed documents separately. "
                             f"Embedding function conflict: new: {_ef_name(embedding_function)} vs persisted: {saved}")
        return Collection(self, name, embedding_function)

    def create_collection(self, name, configuration=None, metadata=None, embedding_function=None, get_or_create=False):
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]", name):
            raise InvalidArgumentError("Validation error: name: Expected a name containing 3-512 characters from [a-zA-Z0-9._-], "
                                       f"starting and ending with a character in [a-zA-Z0-9]. Got: {name}")
        if name in self._store["collections"]:
            if get_or_create:
                return self._open(name, embedding_function)
            raise InternalError(f"Collection [{name}] already exists")
        space = ((configuration or {}).get("hnsw") or {}).get("space") or (metadata or {}).get("hnsw:space") or "l2"
        if space not in _SPACES:
            raise InvalidArgumentError(f"The space must be one of {', '.join(_SPACES)}. Got: {space}")
        _Client._uids += 1
        self._store["collections"][name] = {
            "uid": _Client._uids, "metadata": metadata, "space": space, "rows": [],
            "ef": "default" if embedding_function is None else _ef_name(embedding_function),
        }
        self._save()
        return Collection(self, name, embedding_function)

    def get_or_create_collection(self, name, configuration=None, metadata=None, embedding_function=None):
        return self.create_collection(name, configuration, metadata, embedding_function, get_or_create=True)

    def get_collection(self, name, embedding_function=None):
        if name not in self._store["collections"]:
            raise NotFoundError(f"Collection [{name}] does not exist")
        return self._open(name, embedding_function)

    def list_collections(self):
        return [Collection(self, name, None) for name in self._store["collections"]]

    def count_collections(self):
        return len(self._store["collections"])

    def delete_collection(self, name):
        if name not in self._store["collections"]:
            raise NotFoundError(f"Collection [{name}] does not exist")
        del self._store["collections"][name]
        self._save()

    def heartbeat(self):
        import time
        return time.time_ns()


class Client(_Client):
    """Keeps everything in memory. The data is gone when the program ends."""

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError(_missing("settings for chromadb.Client()"))


EphemeralClient = Client


class PersistentClient(_Client):
    """Saves everything in the folder at `path`, so a later program can open it again."""

    def __init__(self, path="./chroma", **kwargs):
        if kwargs:
            raise TypeError(_missing(f"the {next(iter(kwargs))} argument of PersistentClient"))
        folder = os.path.abspath(str(path))
        self._file = os.path.join(folder, _FILE)
        if os.path.exists(self._file):
            if folder not in _OPEN:
                with open(self._file, encoding="utf-8") as handle:
                    _OPEN[folder] = json.load(handle)
                for saved in _OPEN[folder]["collections"].values():
                    _Client._uids += 1
                    saved["uid"] = _Client._uids
        else:
            os.makedirs(folder, exist_ok=True)
            _OPEN[folder] = {"collections": {}}
        self._store = _OPEN[folder]
        self._save()

    def _save(self):
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        with open(self._file, "w", encoding="utf-8") as handle:
            json.dump(self._store, handle)
