from feast._practice import (EntityNotFoundException, FeastObjectNotFoundException, FeastOfflineStoreInvalidName, FeastOnlineStoreInvalidName,
                             FeastProviderNotImplementedError, FeastRegistryNotSetError, FeatureViewNotFoundException, module_getattr)

__getattr__ = module_getattr("feast.errors")
