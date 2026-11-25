from evidently._practice import Report, module_getattr

__getattr__ = module_getattr("evidently.report")
