from feast._practice import FeastConfigError, RepoConfig, module_getattr

__getattr__ = module_getattr("feast.repo_config")
