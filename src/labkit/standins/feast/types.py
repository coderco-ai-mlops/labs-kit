"""feast.types: the types a Field can have."""

from feast._practice import PrimitiveFeastType, module_getattr

Invalid = PrimitiveFeastType.INVALID
Bytes = PrimitiveFeastType.BYTES
String = PrimitiveFeastType.STRING
Bool = PrimitiveFeastType.BOOL
Int32 = PrimitiveFeastType.INT32
Int64 = PrimitiveFeastType.INT64
Float32 = PrimitiveFeastType.FLOAT32
Float64 = PrimitiveFeastType.FLOAT64
UnixTimestamp = PrimitiveFeastType.UNIX_TIMESTAMP

__getattr__ = module_getattr("feast.types")
