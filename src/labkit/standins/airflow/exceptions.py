from airflow._practice import AirflowException, DuplicateTaskIdFound, TaskNotFound, module_getattr

__getattr__ = module_getattr("airflow.exceptions")
