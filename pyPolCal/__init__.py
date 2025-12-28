from .mm_registry import MM_FUNCTION_REGISTRY, _load_pyMuellerMat_functions

# Auto-load built-in pyMuellerMat functions on package import
_load_pyMuellerMat_functions()