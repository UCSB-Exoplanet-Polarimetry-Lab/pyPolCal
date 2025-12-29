from .mm_registry import MM_FUNCTION_REGISTRY, _load_pyMuellerMat_functions

# Auto-load built-in pyMuellerMat functions on package import
_load_pyMuellerMat_functions()

# Custom/user functions
try:
    from .custom_mms import custom_mms
except ImportError as e:
    print(f"Warning: could not load custom MM functions: {e}")