"""The working parts of the practice version of Airflow: a DAG, its tasks and the links between them."""

import re
from datetime import datetime, timedelta, timezone

GO_FURTHER = "If the spelling is right, it is in the real Airflow: see Go further in this lab."


def not_included(what: str) -> str:
    return f"The practice version of Airflow on the labs site does not include {what}. {GO_FURTHER}"


class AirflowException(Exception):
    pass


class AirflowSkipException(AirflowException):
    """Raise this inside a task to skip it. The tasks after it are skipped too."""


class AirflowFailException(AirflowException):
    """Raise this inside a task to fail it without any retries."""


class DuplicateTaskIdFound(AirflowException):
    pass


class TaskNotFound(AirflowException):
    pass


def _check_key(key) -> None:
    if not isinstance(key, str):
        raise TypeError(f"The key has to be a string and is {type(key)}:{key}")
    if not re.fullmatch(r"[\w.-]+", key):
        raise ValueError(f"The key {key!r} has to be made of alphanumeric characters, dashes, dots, and underscores exclusively")


def _utc(value, name: str):
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError(f"`{name}` should be a datetime, for example datetime(2026, 1, 1). It is a {type(value).__name__}: {value!r}")
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ---- schedules -------------------------------------------------------------

PRESETS = {"@hourly": "0 * * * *", "@daily": "0 0 * * *", "@weekly": "0 0 * * 0", "@monthly": "0 0 1 * *",
           "@quarterly": "0 0 1 */3 *", "@yearly": "0 0 1 1 *"}


class NullTimetable:
    """No schedule: the DAG only runs when someone triggers it."""

    def __repr__(self):
        return f"{type(self).__name__}()"


class OnceTimetable(NullTimetable):
    pass


class CronTriggerTimetable(NullTimetable):
    def __init__(self, text: str):
        cron = PRESETS.get(text, text)
        if text in PRESETS or len(cron.split()) in (5, 6):
            self.expression = cron   # the real Airflow also leaves this out when the text is not a cron expression

    def __repr__(self):
        return f"CronTriggerTimetable(expression={getattr(self, 'expression', None)!r})"


class DeltaTriggerTimetable(NullTimetable):
    def __init__(self, delta: timedelta):
        self.delta = delta

    def __repr__(self):
        return f"DeltaTriggerTimetable(delta={self.delta!r})"


def _timetable(schedule):
    if schedule is None:
        return NullTimetable()
    if schedule == "@once":
        return OnceTimetable()
    if isinstance(schedule, str):
        return CronTriggerTimetable(schedule)
    if isinstance(schedule, timedelta):
        return DeltaTriggerTimetable(schedule)
    raise NotImplementedError(not_included(f"a schedule of type {type(schedule).__name__}. Use a string such as \"@daily\", a timedelta, or None"))


# ---- the DAG ---------------------------------------------------------------

# Settings the real DAG accepts that change nothing in a lab. They are kept as attributes.
_QUIET_DAG_ARGS = ("max_active_tasks", "max_active_runs", "max_consecutive_failed_dag_runs", "dagrun_timeout", "doc_md", "params",
                   "on_success_callback", "on_failure_callback", "is_paused_upon_creation", "dag_display_name", "owner_links",
                   "fail_fast", "render_template_as_native_obj", "template_searchpath", "access_control")


class DAG:
    _open: list = []   # the DAGs whose `with` block is running, innermost last

    def __init__(self, dag_id, *, description=None, schedule=None, start_date=None, end_date=None,
                 default_args=None, catchup=False, tags=None, **other):
        for name in other:
            if name not in _QUIET_DAG_ARGS:
                tip = " Airflow 3 calls it `schedule`." if name == "schedule_interval" else ""
                raise TypeError(f"DAG.__init__() got an unexpected keyword argument '{name}'{tip}")
        _check_key(dag_id)
        if default_args is not None and not isinstance(default_args, dict):
            raise TypeError("default_args must be a dict")
        self.dag_id = dag_id
        self.description = description
        self.schedule = schedule
        self.timetable = _timetable(schedule)
        self.start_date = _utc(start_date, "start_date")
        self.end_date = _utc(end_date, "end_date")
        self.default_args = dict(default_args or {})
        self.catchup = catchup
        self.tags = set(tags or [])
        self.task_dict: dict = {}
        for name in _QUIET_DAG_ARGS:
            setattr(self, name, other.get(name, 16 if name.startswith("max_active") else None))

    def __enter__(self):
        DAG._open.append(self)
        return self

    def __exit__(self, *_):
        DAG._open.remove(self)

    def __repr__(self):
        return f"<DAG: {self.dag_id}>"

    @property
    def tasks(self) -> list:
        return list(self.task_dict.values())

    @property
    def task_ids(self) -> list:
        return list(self.task_dict)

    @property
    def roots(self) -> list:
        return [task for task in self.tasks if not task.upstream_task_ids]

    @property
    def leaves(self) -> list:
        return [task for task in self.tasks if not task.downstream_task_ids]

    def has_task(self, task_id: str) -> bool:
        return task_id in self.task_dict

    def get_task(self, task_id: str):
        if task_id not in self.task_dict:
            raise TaskNotFound(f"Task {task_id} not found")
        return self.task_dict[task_id]

    def add_task(self, task) -> None:
        if task.task_id in self.task_dict and self.task_dict[task.task_id] is not task:
            raise DuplicateTaskIdFound(f"Task id '{task.task_id}' has already been added to the Dag")
        self.task_dict[task.task_id] = task
        task._dag = self
        if task.start_date is None:
            task.start_date = self.start_date

    def topological_sort(self) -> tuple:
        """The tasks with every task after the ones it waits for."""
        done, left = [], self.tasks
        while left:
            ready = [task for task in left if task.upstream_task_ids <= {t.task_id for t in done}] or left   # `or left`: a loop, as in the real one
            done += ready
            left = [task for task in left if task not in ready]
        return tuple(done)

    def test(self):
        raise NotImplementedError("The practice version of Airflow on the labs site does not include `dag.test()`, which needs Airflow's database. "
                                  "The block at the bottom of the lab file runs your tasks once instead. "
                                  "It is in the real Airflow: see Go further in this lab.")


# ---- tasks -----------------------------------------------------------------

class BaseOperator:
    # Settings every task accepts, with their defaults. A DAG's default_args can supply any of them.
    _settings = {"retries": 0, "retry_delay": timedelta(seconds=300), "owner": "airflow", "start_date": None, "end_date": None,
                 "trigger_rule": "all_success", "depends_on_past": False, "execution_timeout": None, "doc_md": None, "params": None,
                 "email": None, "email_on_failure": True, "email_on_retry": True, "retry_exponential_backoff": False,
                 "max_retry_delay": None, "priority_weight": 1, "pool": "default_pool", "queue": "default",
                 "on_success_callback": None, "on_failure_callback": None, "on_retry_callback": None, "on_execute_callback": None}
    _required: tuple = ()

    def __init__(self, *args, **kwargs):
        if args:
            raise TypeError("Use keyword arguments when initializing operators")
        if "task_id" not in kwargs:
            raise TypeError("BaseOperator.__init__() missing 1 required keyword-only argument: 'task_id'")
        self.task_id = kwargs.pop("task_id")
        _check_key(self.task_id)
        dag = kwargs.pop("dag", None) or (DAG._open[-1] if DAG._open else None)
        if dag is not None and not isinstance(dag, DAG):
            raise TypeError(f"`dag` should be a DAG. It is a {type(dag).__name__}: {dag!r}")
        defaults = dag.default_args if dag is not None else {}
        settings = {}
        for cls in reversed(type(self).__mro__):
            settings.update(vars(cls).get("_settings", {}))
        for name in type(self)._required:
            if name not in kwargs and name not in defaults:
                raise TypeError(f"missing keyword argument '{name}'")
        for name, default in settings.items():
            setattr(self, name, kwargs.pop(name) if name in kwargs else defaults.get(name, default))
        if kwargs:
            raise TypeError(f"Invalid arguments were passed to {type(self).__name__} (task_id: {self.task_id}). Invalid arguments were:\n**kwargs: {kwargs}")
        self.start_date, self.end_date = _utc(self.start_date, "start_date"), _utc(self.end_date, "end_date")
        self.upstream_task_ids: set = set()
        self.downstream_task_ids: set = set()
        self._dag = None
        if dag is not None:
            dag.add_task(self)

    def __repr__(self):
        return f"<Task({type(self).__name__}): {self.task_id}>"

    @property
    def task_type(self) -> str:
        return type(self).__name__

    @property
    def dag(self):
        if self._dag is None:
            raise RuntimeError(f"Operator {self} has not been assigned to a Dag yet")
        return self._dag

    @dag.setter
    def dag(self, dag) -> None:
        dag.add_task(self)

    @property
    def dag_id(self) -> str:
        return self.dag.dag_id

    @property
    def upstream_list(self) -> list:
        return [self.dag.task_dict[task_id] for task_id in self.dag.task_dict if task_id in self.upstream_task_ids]

    @property
    def downstream_list(self) -> list:
        return [self.dag.task_dict[task_id] for task_id in self.dag.task_dict if task_id in self.downstream_task_ids]

    def execute(self, context=None):
        raise NotImplementedError()

    # The links: a >> b means "a runs before b".
    def _link(self, others, upstream: bool) -> None:
        others = list(others) if isinstance(others, (list, tuple)) else [others]
        for other in others:
            if not isinstance(other, BaseOperator):
                raise TypeError(f"A task can only be linked to another task, or to a list of tasks. {self.task_id} was linked to a {type(other).__name__}: {other!r}")
        everyone = [self, *others]
        dags = []
        for task in everyone:
            if task._dag is not None and task._dag not in dags:
                dags.append(task._dag)
        if len(dags) > 1:
            raise RuntimeError(f"Tried to set relationships between tasks in more than one Dag: {dags}")
        if not dags:
            raise ValueError(f"Tried to create relationships between tasks that don't have Dags yet. Set the Dag for at least one task and try again: {everyone}")
        for task in everyone:
            if task._dag is None:
                dags[0].add_task(task)
        for other in others:
            before, after = (other, self) if upstream else (self, other)
            before.downstream_task_ids.add(after.task_id)
            after.upstream_task_ids.add(before.task_id)

    def set_upstream(self, task_or_tasks) -> None:
        self._link(task_or_tasks, upstream=True)

    def set_downstream(self, task_or_tasks) -> None:
        self._link(task_or_tasks, upstream=False)

    def __rshift__(self, other):       # self >> other
        self.set_downstream(other)
        return other

    def __lshift__(self, other):       # self << other
        self.set_upstream(other)
        return other

    def __rrshift__(self, other):      # [a, b] >> self
        self.set_upstream(other)
        return self

    def __rlshift__(self, other):      # [a, b] << self
        self.set_downstream(other)
        return self


class PythonOperator(BaseOperator):
    _settings = {"python_callable": None, "op_args": None, "op_kwargs": None}
    _required = ("python_callable",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not callable(self.python_callable):
            raise AirflowException("`python_callable` param must be callable")
        self.op_args = tuple(self.op_args or ())
        self.op_kwargs = dict(self.op_kwargs or {})

    def execute(self, context=None):
        return self.python_callable(*self.op_args, **self.op_kwargs)


def chain(*tasks) -> None:
    """chain(a, b, c) is the same as a >> b >> c. An item can be a list of tasks."""
    for before, after in zip(tasks, tasks[1:]):
        if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
            if len(before) != len(after):
                raise AirflowException(f"Chain not supported for different length Iterable. Got {len(before)} and {len(after)}.")
            for one, two in zip(before, after):
                one.set_downstream(two)
        elif isinstance(before, (list, tuple)):
            after.set_upstream(before)
        else:
            before.set_downstream(after)


def module_getattr(module: str):
    """A module-level __getattr__ that explains, in a plain sentence, that a name is not in the practice version."""
    def __getattr__(name: str):
        if name.startswith("__"):
            raise AttributeError(f"module '{module}' has no attribute '{name}'")
        raise ImportError(not_included(f"`{module}.{name}`"))
    return __getattr__
