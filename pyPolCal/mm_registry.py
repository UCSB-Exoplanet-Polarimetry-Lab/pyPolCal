"""
mm_registry.py

Registry for Mueller matrix functions. This generates a database
from which "system dictionaries" can be built. 

"""

from typing import Callable, Dict
import inspect
import numpy as np

MM_FUNCTION_REGISTRY: Dict[str, Callable] = {}

def register_mm_function(name: str | None = None):
    """
    Decorator used to register a Mueller-matrix-generating function. IMPORTANT: all functions
    must only have keyword arguments! 

    Parameters
    ----------
    name : str, optional
        The string key under which the function will be registered.
        If omitted, the function's __name__ is used.

    Returns
    -------
    decorator
        A decorator that registers the function in MM_FUNCTION_REGISTRY.
    """
    # internal register helper
    def _register(func: Callable, key_name: str | None = None):
        key = key_name or func.__name__

        if key in MM_FUNCTION_REGISTRY:
            # allow overwriting but warn
            print(f"Mueller matrix function '{key}' is already registered. Overwriting.")

        # Validate signature: must be callable by keywords (parameters must have defaults)
        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError(f"Mueller matrix function '{key}' cannot have positional-only parameter '{param.name}'.")
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(f"Mueller matrix function '{key}' cannot accept *args (parameter '{param.name}').")
            if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and param.default is inspect._empty:
                raise TypeError(
                    f"Mueller matrix function '{key}' must only have keyword arguments or keyword-defaults; '{param.name}' is a required positional parameter."
                )

        # Store the original module-level function so pickle sees the module+name
        MM_FUNCTION_REGISTRY[key] = func

        # Return the original function object (important for identity/pickling)
        return func

    # Support both @register_mm_function and @register_mm_function(name='...')
    if callable(name):
        # Used as @register_mm_function without parentheses
        func = name
        return _register(func, None)

    def decorator(func: Callable):
        return _register(func, name)

    return decorator


def _load_pyMuellerMat_functions():
    """
    Load all callable Mueller-matrix functions from
    pyMuellerMat.common_mm_functions into the registry.

    This keeps pyMuellerMat completely unmodified while making
    its functions available through the registry.
    """
    import pyMuellerMat.common_mm_functions as cmf

    for name, obj in inspect.getmembers(cmf):
        # Only register public callables
        if name.startswith("_"):
            continue
        if not callable(obj):
            continue

        # Do not overwrite user-registered functions
        if name in MM_FUNCTION_REGISTRY:
            continue

        MM_FUNCTION_REGISTRY[name] = obj


