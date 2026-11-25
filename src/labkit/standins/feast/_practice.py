"""The working parts of the practice version of Feast.

A feature store here is a folder (the "repo") that holds feature_store.yaml. `apply` saves the
definitions in the registry file named there. The offline read joins the source parquet file to
your entity rows the way Feast's file offline store does. `materialize` copies the latest value of
every entity into the online store file, and the online read looks one key up in it.
"""

import enum
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

GO_FURTHER = "If the spelling is right, it is in the real Feast: see Go further in this lab."


def not_included(what: str) -> str:
    return f"The practice version of Feast on the labs site does not include {what}. {GO_FURTHER}"


def module_getattr(module: str):
    """A module-level __getattr__ that explains, in a plain sentence, that a name is not in the practice version."""
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(f"module '{module}' has no attribute '{name}'")
        raise ImportError(not_included(f"`{module}.{name}`"))
    return __getattr__


# ---- errors ------------------------------------------------------------------

class FeastObjectNotFoundException(Exception):
    pass


class FeatureViewNotFoundException(FeastObjectNotFoundException):
    def __init__(self, name, project=None):
        super().__init__(f"Feature view {name} does not exist in project {project}" if project else f"Feature view {name} does not exist")


class EntityNotFoundException(FeastObjectNotFoundException):
    def __init__(self, name, project=None):
        super().__init__(f"Entity {name} does not exist in project {project}" if project else f"Entity {name} does not exist")


class FeastConfigError(Exception):
    pass


class FeastProviderNotImplementedError(Exception):
    pass


class FeastRegistryNotSetError(Exception):
    pass


class FeastOnlineStoreInvalidName(Exception):
    pass


class FeastOfflineStoreInvalidName(Exception):
    pass


# ---- types -------------------------------------------------------------------

class ValueType(enum.Enum):
    UNKNOWN = 0
    BYTES = 1
    STRING = 2
    INT32 = 3
    INT64 = 4
    DOUBLE = 5
    FLOAT = 6
    BOOL = 7
    UNIX_TIMESTAMP = 8


class PrimitiveFeastType(enum.Enum):
    INVALID = 0
    BYTES = 1
    STRING = 2
    INT32 = 3
    INT64 = 4
    FLOAT64 = 5
    FLOAT32 = 6
    BOOL = 7
    UNIX_TIMESTAMP = 8


def _a(value) -> str:
    name = type(value).__name__
    return f"{'an' if name[:1].lower() in 'aeiou' else 'a'} {name}: {value!r}"


def _need_text(value, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{what} should be text, for example \"district_id\". It is {_a(value)}")
    return value


# ---- definitions -------------------------------------------------------------

class Field:
    def __init__(self, *, name: str, dtype, description: str = "", tags=None, **other):
        for key in other:
            if key not in ("vector_index", "vector_length", "vector_search_metric"):
                raise TypeError(f"Field.__init__() got an unexpected keyword argument '{key}'")
        self.name = _need_text(name, "A Field's `name`")
        if not isinstance(dtype, PrimitiveFeastType):
            raise TypeError(f"The `dtype` of the field {name!r} should be a type from feast.types, for example Float64. It is {_a(dtype)}")
        self.dtype = dtype
        self.description = description
        self.tags = dict(tags or {})

    def __eq__(self, other):
        return isinstance(other, Field) and (self.name, self.dtype, self.description, self.tags) == (other.name, other.dtype, other.description, other.tags)

    def __hash__(self):
        return hash((self.name, self.dtype))

    def __repr__(self):
        return f"Field(name={self.name!r}, dtype={self.dtype!r})"


class Entity:
    def __init__(self, *, name: str, join_keys=None, value_type=None, description: str = "", tags=None, owner: str = ""):
        self.name = _need_text(name, "An Entity's `name`")
        if join_keys is not None and (isinstance(join_keys, str) or not all(isinstance(key, str) for key in join_keys)):
            raise TypeError(f"`join_keys` should be a list of column names, for example [\"district_id\"]. It is {_a(join_keys)}")
        if join_keys and len(join_keys) > 1:
            raise ValueError("An entity may only have a single join key. Multiple join keys will be supported in the future.")
        self.join_key = join_keys[0] if join_keys else name
        if value_type is not None and not isinstance(value_type, ValueType):
            raise TypeError(f"`value_type` should come from ValueType, for example ValueType.INT64. It is {_a(value_type)}")
        self.value_type = value_type or ValueType.UNKNOWN
        self.description = description
        self.tags = dict(tags or {})
        self.owner = owner

    def __eq__(self, other):
        return isinstance(other, Entity) and (self.name, self.join_key, self.value_type) == (other.name, other.join_key, other.value_type)

    def __hash__(self):
        return hash((self.name, self.join_key))

    def __repr__(self):
        return f"<Entity(name = {self.name}, join_key = {self.join_key}, value_type = {self.value_type})>"


class FileSource:
    def __init__(self, *, path: str, timestamp_field: str = "", created_timestamp_column: str = "", name: str = "", description: str = "",
                 tags=None, owner: str = "", **other):
        for key, value in other.items():
            if key == "event_timestamp_column":
                raise TypeError("FileSource.__init__() got an unexpected keyword argument 'event_timestamp_column'. Feast now calls it `timestamp_field`.")
            if key not in ("file_format", "field_mapping", "s3_endpoint_override"):
                raise TypeError(f"FileSource.__init__() got an unexpected keyword argument '{key}'")
            if value:
                raise NotImplementedError(not_included(f"the `{key}` setting of FileSource"))
        if isinstance(path, Path):
            path = str(path)
        self.path = _need_text(path, "A FileSource's `path`")
        if any(self.path.startswith(prefix) for prefix in ("s3://", "gs://", "http://", "https://", "abfs://")):
            raise NotImplementedError(not_included(f"a source file that is not on this machine ({self.path})"))
        if created_timestamp_column:
            raise NotImplementedError(not_included("the `created_timestamp_column` setting of FileSource"))
        if not isinstance(timestamp_field, str):
            raise TypeError(f"`timestamp_field` should be the name of a column, for example \"event_timestamp\". It is {_a(timestamp_field)}")
        self.timestamp_field = timestamp_field
        self.created_timestamp_column = ""
        self.name = name or self.path
        self.description = description
        self.tags = dict(tags or {})
        self.owner = owner

    def __repr__(self):
        return f"FileSource(path={self.path!r}, timestamp_field={self.timestamp_field!r})"


class FeatureView:
    def __init__(self, *, name: str, source, schema=None, entities=None, ttl=timedelta(days=0), online: bool = True, description: str = "",
                 tags=None, owner: str = ""):
        self.name = _need_text(name, "A FeatureView's `name`")
        if not isinstance(source, FileSource):
            raise TypeError(f"The `source` of the feature view {name!r} should be a FileSource. It is {_a(source)}")
        if entities is not None and (not isinstance(entities, (list, tuple)) or not all(isinstance(entity, Entity) for entity in entities)):
            raise TypeError(f"`entities` should be a list of Entity objects, for example [district]. Pass the objects, not their names. It is {_a(entities)}")
        if ttl is not None and not isinstance(ttl, timedelta):
            raise TypeError(f"`ttl` should be a timedelta, for example timedelta(days=30). It is {_a(ttl)}")
        if schema is not None and (not isinstance(schema, (list, tuple)) or not all(isinstance(field, Field) for field in schema)):
            raise TypeError(f"`schema` should be a list of Field objects, for example [Field(name=\"MedInc\", dtype=Float64)]. It is {_a(schema)}")
        entities = list(entities or [])
        if len(entities) > 1:
            raise NotImplementedError(not_included("a feature view with more than one entity"))
        self.batch_source = source
        self.stream_source = None
        self.entities = [entity.name for entity in entities]
        self._join_keys = [entity.join_key for entity in entities]
        self.ttl = ttl
        self.online = online
        self.description = description
        self.tags = dict(tags or {})
        self.owner = owner
        self.schema = list(schema or [])
        self.features = [field for field in self.schema if field.name not in self._join_keys]
        names = [field.name for field in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"The feature view {name!r} names the same feature twice in its schema: {names}")
        self.materialization_intervals: list = []

    @property
    def most_recent_end_time(self):
        return max((end for _, end in self.materialization_intervals), default=None)

    def __eq__(self, other):
        return isinstance(other, FeatureView) and (self.name, self.entities, self.ttl, self.features, self.batch_source.path) == (
            other.name, other.entities, other.ttl, other.features, other.batch_source.path)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"<FeatureView(name = {self.name}, entities = {self.entities}, ttl = {self.ttl}, features = {[field.name for field in self.features]})>"


# ---- the settings file -------------------------------------------------------

class _Section:
    def __init__(self, **values):
        self.__dict__.update(values)

    def __repr__(self):
        return f"{type(self).__name__}({self.__dict__})"


class RepoConfig(_Section):
    pass


_OTHER_ONLINE = ("redis", "dynamodb", "datastore", "bigtable", "postgres", "snowflake.online", "cassandra", "mysql", "hbase", "milvus", "remote")
_OTHER_OFFLINE = ("bigquery", "snowflake.offline", "redshift", "spark", "postgres", "trino", "athena", "duckdb", "remote")


def _read_config(repo: Path) -> RepoConfig:
    file = repo / "feature_store.yaml"
    if not file.exists():
        raise FileNotFoundError(f"[Errno 2] No such file or directory: '{file}'")
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise FeastConfigError(f"feature_store.yaml should hold settings such as `project: housing`, one per line. It holds {_a(raw)}\nat {file}")
    project = raw.get("project")
    if project is None:
        raise FeastConfigError(f"1 validation error for RepoConfig\nproject\n  Field required\nat {file}")
    if not isinstance(project, str) or not project.replace("_", "").isalnum() or project.startswith("_"):
        raise FeastConfigError("1 validation error for RepoConfig\nproject\n  Value error, Project name, "
                               f"{project}, should only have alphanumerical values and underscores but not start with an underscore.\nat {file}")
    provider = raw.get("provider", "local")
    if provider in ("gcp", "aws", "azure"):
        raise NotImplementedError(not_included(f"the {provider} provider. Use `provider: local`"))
    if provider != "local":
        raise FeastProviderNotImplementedError(f"Provider '{provider}' is not implemented")
    registry = raw.get("registry")
    if isinstance(registry, dict):
        registry = registry.get("path")
    if not registry:
        raise FeastRegistryNotSetError("Registry is not set, but is required")
    if not isinstance(registry, str) or "://" in registry:
        raise NotImplementedError(not_included(f"a registry that is not a file in the repo ({registry!r})"))

    def store(key: str, default: str, fine: tuple, others: tuple, error, kind: str) -> dict:
        section = raw.get(key)
        section = {"type": section} if isinstance(section, str) else dict(section or {})
        name = section.setdefault("type", default)
        if name in others:
            raise NotImplementedError(not_included(f"the {name} {kind} store. In feature_store.yaml, use `type: {default}` under `{key}`"))
        if name not in fine:
            raise error(f"{kind.capitalize()} Store Class '{name}' should end with the string `{kind.capitalize()}Store`.'")
        return section

    online = store("online_store", "sqlite", ("sqlite",), _OTHER_ONLINE, FeastOnlineStoreInvalidName, "online")
    offline = store("offline_store", "file", ("file", "dask"), _OTHER_OFFLINE, FeastOfflineStoreInvalidName, "offline")
    version = raw.get("entity_key_serialization_version", 1)
    return RepoConfig(project=project, provider=provider, registry=_Section(path=registry, registry_type="file"),
                      online_store=_Section(type="sqlite", path=online.get("path", "data/online.db")),
                      offline_store=_Section(type="dask" if offline["type"] == "dask" else "file"),
                      entity_key_serialization_version=version, repo_path=repo)


# ---- reading features --------------------------------------------------------

def _aware(value, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"`{name}` should be a datetime, for example datetime.now(timezone.utc). It is {_a(value)}")
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _utc_column(values: pd.Series) -> pd.Series:
    """A column of times, all in UTC. Times with no time zone are taken to be UTC, as Feast does."""
    return pd.to_datetime(values.apply(lambda x: x if getattr(x, "tzinfo", None) is not None else x.replace(tzinfo=timezone.utc)), utc=True)


class RetrievalJob:
    """What get_historical_features gives back. Nothing is read until to_df() is called."""

    def __init__(self, evaluate):
        self._evaluate = evaluate

    def to_df(self, **_ignored) -> pd.DataFrame:
        return self._evaluate()

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(not_included(f"`RetrievalJob.{name}`. Use `.to_df()`"))


class OnlineResponse:
    """What get_online_features gives back."""

    def __init__(self, columns: dict):
        self._columns = columns

    def to_dict(self, include_event_timestamps: bool = False) -> dict:
        if include_event_timestamps:
            raise NotImplementedError(not_included("`include_event_timestamps=True`"))
        return {name: list(values) for name, values in self._columns.items()}

    def to_df(self, include_event_timestamps: bool = False) -> pd.DataFrame:
        return pd.DataFrame(self.to_dict(include_event_timestamps))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(not_included(f"`OnlineResponse.{name}`. Use `.to_dict()` or `.to_df()`"))


class FeatureStore:
    def __init__(self, repo_path=None, config=None, fs_yaml_file=None):
        if config is not None or fs_yaml_file is not None:
            raise NotImplementedError(not_included("the `config` and `fs_yaml_file` settings of FeatureStore. Use `repo_path`"))
        self.repo_path = Path(repo_path) if repo_path is not None else Path.cwd()
        self.config = _read_config(self.repo_path)
        self._registry_file = self.repo_path / self.config.registry.path
        self._online_file = self.repo_path / self.config.online_store.path

    def __repr__(self):
        return f"<FeatureStore project={self.project} repo_path={self.repo_path}>"

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(not_included(f"`FeatureStore.{name}`"))

    @property
    def project(self) -> str:
        return self.config.project

    # The registry and the online store are files, so a second FeatureStore on the same repo sees the same things.
    def _load(self, file: Path, empty):
        if not file.exists():
            return empty
        with open(file, "rb") as handle:
            return pickle.load(handle)

    def _save(self, file: Path, value) -> None:
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, "wb") as handle:
            pickle.dump(value, handle)

    def _registry(self) -> dict:
        return self._load(self._registry_file, {"entities": {}, "views": {}})

    def _view(self, name: str, registry=None) -> FeatureView:
        views = (registry or self._registry())["views"]
        if name not in views:
            raise FeatureViewNotFoundException(name, self.project)
        return views[name]

    def _source(self, view: FeatureView) -> pd.DataFrame:
        path = Path(view.batch_source.path)
        path = path if path.is_absolute() else self.repo_path / path
        if not path.exists():
            raise FileNotFoundError(f"The feature view {view.name!r} reads its values from {path}, and there is no file there. "
                                    "A relative `path` in a FileSource starts from the repo folder.")
        if path.suffix.lower() not in (".parquet", ".pq"):
            raise NotImplementedError(not_included(f"source files that are not parquet ({path.name})"))
        return pd.read_parquet(path)

    def _timestamp_field(self, view: FeatureView, source: pd.DataFrame) -> str:
        field = view.batch_source.timestamp_field
        if not field:
            times = list(source.select_dtypes(include=["datetime", "datetimetz"]).columns)
            if len(times) != 1:
                raise ValueError(f"The source of the feature view {view.name!r} has no `timestamp_field`, and Feast could not work one out. "
                                 "Add timestamp_field=\"event_timestamp\" to the FileSource.")
            field = times[0]
        if field not in source.columns:
            raise KeyError(f"The source file of the feature view {view.name!r} has no column called {field!r}, which is its `timestamp_field`. "
                           f"Its columns are: {list(source.columns)}")
        return field

    def apply(self, objects) -> None:
        """Save definitions in the registry. Applying a definition again replaces the earlier one."""
        objects = list(objects) if isinstance(objects, (list, tuple)) else [objects]
        registry = self._registry()
        for item in objects:
            if isinstance(item, type):
                raise TypeError(f"apply() was given the class {item.__name__} itself. Pass the objects you created, for example store.apply([district, district_features]).")
            if isinstance(item, Entity):
                registry["entities"][item.name] = item
            elif isinstance(item, FeatureView):
                view = pickle.loads(pickle.dumps(item))
                if not view.features:   # no schema: every column that is not the key or the time is a feature
                    source = self._source(view)
                    skip = set(view._join_keys) | {self._timestamp_field(view, source)}
                    view.features = [Field(name=str(column), dtype=PrimitiveFeastType.FLOAT64) for column in source.columns if column not in skip]
                registry["views"][view.name] = view
            elif isinstance(item, FileSource):
                pass
            else:
                raise NotImplementedError(not_included(f"applying {_a(item)}. It applies Entity and FeatureView objects"))
        self._save(self._registry_file, registry)
        if not self._online_file.exists():
            self._save(self._online_file, {})

    def list_entities(self, **_ignored) -> list:
        return list(self._registry()["entities"].values())

    def list_feature_views(self, **_ignored) -> list:
        return list(self._registry()["views"].values())

    def get_feature_view(self, name: str, **_ignored) -> FeatureView:
        return self._view(name)

    def get_entity(self, name: str, **_ignored) -> Entity:
        entities = self._registry()["entities"]
        if name not in entities:
            raise EntityNotFoundException(name, self.project)
        return entities[name]

    def _requested(self, features) -> list:
        """Turn ["district_features:MedInc", ...] into [(view, [feature names])], keeping the order."""
        if isinstance(features, str) or not isinstance(features, (list, tuple)):
            raise TypeError(f"`features` should be a list of text such as [\"district_features:MedInc\"]. It is {_a(features)}")
        registry = self._registry()
        wanted: dict = {}
        for ref in features:
            if not isinstance(ref, str):
                raise TypeError(f"Every item in `features` should be text such as \"district_features:MedInc\". One is {_a(ref)}")
            view_name, _, feature = ref.partition(":")
            view = self._view(view_name, registry)
            if feature not in [field.name for field in view.features]:
                raise KeyError(f"Feature {feature} not found in projection {view_name}")
            wanted.setdefault(view_name, (view, []))[1].append(feature)
        return list(wanted.values())

    # -- offline: the path training uses --

    def get_historical_features(self, entity_df, features, full_feature_names: bool = False) -> RetrievalJob:
        """A point-in-time join. Each row of entity_df gets, for every feature, the latest value that was
        recorded at or before that row's event_timestamp, and no more than the view's ttl before it."""
        requested = self._requested(features)
        if not isinstance(entity_df, pd.DataFrame):
            raise ValueError(f"Please provide an entity_df of type {type(pd.DataFrame)} instead of type {type(entity_df)}")
        when = "event_timestamp"
        if when not in entity_df.columns:
            times = entity_df.select_dtypes(include=["datetime", "datetimetz"]).columns
            if len(times) != 1:
                raise ValueError("Please provide an entity_df with a column named event_timestamp representing the time of events.")
            print(f"Using {times[0]} as the event timestamp. To specify a column explicitly, please name it event_timestamp.")
            when = times[0]

        def evaluate() -> pd.DataFrame:
            result = entity_df.copy()
            result[when] = _utc_column(result[when])
            result = result.sort_values(when)
            all_keys: list = []
            for view, names in requested:
                source = self._source(view)
                time_field = self._timestamp_field(view, source)
                keys = list(view._join_keys)
                for key in keys:
                    if key not in result.columns:
                        raise KeyError(key)
                all_keys = list(set(all_keys + keys))
                shown = {name: f"{view.name}__{name}" if full_feature_names else name for name in names}
                source = source.rename(columns=shown)[[time_field, *keys, *shown.values()]]
                hidden = f"__{time_field}"
                source = source.rename(columns={time_field: hidden})
                # Every entity row is paired with every recorded value for the same key...
                joined = pd.merge(result, source, left_on=keys, right_on=keys, suffixes=("", "__"), how="left")
                recorded = joined[hidden]
                if recorded.notna().any():
                    joined[hidden] = recorded = _utc_column(recorded) if recorded.notna().all() else pd.to_datetime(recorded, utc=True)
                # ...then only the values from the allowed window are kept: not after the row's own time,
                # and not older than the ttl...
                keep = recorded.isna() | (recorded <= joined[when])
                if view.ttl and view.ttl.total_seconds() != 0:
                    keep &= recorded.isna() | (recorded >= joined[when] - view.ttl)
                joined = joined[keep]
                # ...and of those, the newest one wins.
                joined = joined.sort_values(by=hidden, na_position="first")
                joined = joined.drop_duplicates(all_keys + [when], keep="last", ignore_index=True)
                result = joined.drop(columns=[hidden])
            return result

        return RetrievalJob(evaluate)

    # -- online: the path inference uses --

    def materialize(self, start_date, end_date, feature_views=None) -> None:
        """Copy the latest value of every entity, recorded from start_date up to end_date, into the online store."""
        start, end = _aware(start_date, "start_date"), _aware(end_date, "end_date")
        if start > end:
            raise ValueError(f"The given start_date {start} is greater than the given end_date {end}.")
        registry = self._registry()
        views = self._views_to_materialize(registry, feature_views)
        print(f"Materializing {len(views)} feature views from {start.replace(microsecond=0)} to {end.replace(microsecond=0)} into the sqlite online store.\n")
        for view in views:
            print(f"{view.name}:")
            self._copy(view, start, end)
        self._save(self._registry_file, registry)

    def materialize_incremental(self, end_date, feature_views=None) -> None:
        """Like materialize, starting where the last one ended. The first time, it starts one ttl before now."""
        end = _aware(end_date, "end_date")
        registry = self._registry()
        views = self._views_to_materialize(registry, feature_views)
        print(f"Materializing {len(views)} feature views to {end.replace(microsecond=0)} into the sqlite online store.\n")
        for view in views:
            start = view.most_recent_end_time
            if start is None:
                if view.ttl is None:
                    raise Exception(f"No start time found for feature view {view.name}. materialize_incremental() requires"
                                    " either a ttl to be set or for materialize() to have been run at least once.")
                if view.ttl.total_seconds() > 0:
                    start = datetime.now(timezone.utc) - view.ttl
                else:
                    print(f"Since the ttl is 0 for feature view {view.name}, the start date will be set to 1 year before the current time.")
                    start = datetime.now(timezone.utc) - timedelta(weeks=52)
            print(f"{view.name} from {start.replace(microsecond=0)} to {end.replace(microsecond=0)}:")
            self._copy(view, start, end)
        self._save(self._registry_file, registry)

    def _views_to_materialize(self, registry: dict, names) -> list:
        if names is None:
            return [view for view in registry["views"].values() if view.online]
        if isinstance(names, str) or not all(isinstance(name, str) for name in names):
            raise TypeError(f"`feature_views` should be a list of feature view names, for example [\"district_features\"]. It is {_a(names)}")
        return [self._view(name, registry) for name in names]

    def _copy(self, view: FeatureView, start: datetime, end: datetime) -> None:
        source = self._source(view)
        time_field = self._timestamp_field(view, source)
        names = [field.name for field in view.features]
        missing = [column for column in view._join_keys + names if column not in source.columns]
        if missing:
            raise KeyError(f"{missing} not in index")
        source = source[[*view._join_keys, *names, time_field]].copy()
        source[time_field] = _utc_column(source[time_field])
        source = source.sort_values(by=time_field)
        source = source[(source[time_field] >= start) & (source[time_field] < end)]
        latest = source.drop_duplicates(view._join_keys, keep="last")
        online = self._load(self._online_file, {})
        for row in latest.to_dict("records"):
            key = (view.name, tuple(_plain(row[key]) for key in view._join_keys))
            online[key] = {name: _plain(row[name]) for name in names}
        self._save(self._online_file, online)
        view.materialization_intervals.append((start, end))
        print(f"  {len(latest)} rows written")

    def get_online_features(self, features, entity_rows, full_feature_names: bool = False) -> OnlineResponse:
        """A lookup by key: the value materialize last wrote for each entity. None when there is none."""
        requested = self._requested(features)
        if isinstance(entity_rows, dict):
            entity_rows = [dict(zip(entity_rows, values)) for values in zip(*entity_rows.values())] if all(
                isinstance(values, (list, tuple)) for values in entity_rows.values()) else None
        if not isinstance(entity_rows, (list, tuple)) or not all(isinstance(row, dict) for row in entity_rows):
            raise TypeError("`entity_rows` should be a list with one dictionary per entity, for example [{\"district_id\": 3}].")
        online = self._load(self._online_file, {})
        columns: dict = {}
        for view, _ in requested:
            for key in view._join_keys:
                provided = sorted({name for row in entity_rows for name in row})
                if any(key not in row for row in entity_rows):
                    raise KeyError(f"Missing join key values for keys: ['{key}']. No values provided for keys: ['{key}']. Provided join_key_values: {provided}")
                columns.setdefault(key, [row[key] for row in entity_rows])
        for view, names in requested:
            for name in names:
                shown = f"{view.name}__{name}" if full_feature_names else name
                columns[shown] = []
                for row in entity_rows:
                    stored = online.get((view.name, tuple(_plain(row[key]) for key in view._join_keys)))
                    columns[shown].append(None if stored is None else stored.get(name))
        return OnlineResponse(columns)


def _plain(value):
    """A numpy value as the plain Python value, so keys and values compare and print the ordinary way."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value != value:
        return None
    return value
