"""airflow.sdk.exceptions: where Airflow 3 keeps the exceptions a task can raise."""

from airflow._practice import AirflowException, AirflowFailException, AirflowSkipException, DuplicateTaskIdFound, TaskNotFound, module_getattr

__getattr__ = module_getattr("airflow.sdk.exceptions")
