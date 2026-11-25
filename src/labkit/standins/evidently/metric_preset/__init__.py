from evidently._practice import DataDriftPreset, module_getattr

__getattr__ = module_getattr("evidently.metric_preset")
