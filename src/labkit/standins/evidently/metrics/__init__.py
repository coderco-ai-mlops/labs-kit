from evidently._practice import DataDriftTable, DatasetDriftMetric, module_getattr

__getattr__ = module_getattr("evidently.metrics")
