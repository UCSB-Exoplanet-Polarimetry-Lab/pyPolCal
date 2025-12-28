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
    def decorator(func: Callable):
        key = name or func.__name__

        if key in MM_FUNCTION_REGISTRY:
            print(
                f"Mueller matrix function '{key}' is already registered. Overwriting."
            )

        sig = inspect.signature(func)
        for param in sig.parameters.values():
            if param.default is param.empty:
                raise TypeError(
                    f"Mueller matrix function '{key}' must only have keyword arguments "
                    f"with defaults; '{param.name}' has no default."
                )

        def wrapped_func(**kwargs):
            mm = func(**kwargs)
            mm = np.asarray(mm)
            if mm.shape != (4, 4):
                raise ValueError(
                    f"Mueller matrix function '{key}' must return a 4x4 array, got {mm.shape}"
                )
            return mm
        MM_FUNCTION_REGISTRY[key] = func
        return wrapped_func
    
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


