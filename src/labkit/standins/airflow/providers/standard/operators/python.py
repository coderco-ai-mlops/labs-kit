from airflow._practice import PythonOperator, module_getattr

__getattr__ = module_getattr("airflow.providers.standard.operators.python")
