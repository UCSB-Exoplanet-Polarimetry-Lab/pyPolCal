import copy

import numpy as np


TWOPI = 2 * np.pi
ANGLE_MIN_DEG = -45.0
ANGLE_MAX_DEG = 45.0
DICHROIC_THETA_MIN_DEG = -90.0
DICHROIC_THETA_MAX_DEG = 90.0
OPTICS_THETA_MIN_DEG = -90.0
OPTICS_THETA_MAX_DEG = 90.0


def build_shared_bounds():
    """Shared hard bounds for MBI runs with fixed IMR angle and PM90 dichroic/optics angles."""
    return {
        "dichroic": {
            "phi": (-TWOPI, TWOPI),
            "epsilon": (0.0, 1.0),
            "theta": (DICHROIC_THETA_MIN_DEG, DICHROIC_THETA_MAX_DEG),
        },
        "flc": {
            "phi": (0.0, TWOPI),
            "delta_theta": (ANGLE_MIN_DEG, ANGLE_MAX_DEG),
        },
        "optics": {
            "phi": (-TWOPI, TWOPI),
            "epsilon": (0.0, 1.0),
            "theta": (OPTICS_THETA_MIN_DEG, OPTICS_THETA_MAX_DEG),
        },
        "image_rotator": {
            "phi": (0.0, TWOPI),
        },
        "hwp": {
            "phi": (0.0, TWOPI),
            "delta_theta": (ANGLE_MIN_DEG, ANGLE_MAX_DEG),
        },
        "lp": {
            "theta": (ANGLE_MIN_DEG, ANGLE_MAX_DEG),
        },
    }


def _uniform_from(bounds, component, param):
    low, high = bounds[component][param]
    return {"type": "uniform", "kwargs": {"low": low, "high": high}}


def build_shared_prior_dict():
    """Shared prior family for all requested MBI wavelengths."""
    bounds = build_shared_bounds()
    return {
        "dichroic": {
            "phi": _uniform_from(bounds, "dichroic", "phi"),
            "epsilon": _uniform_from(bounds, "dichroic", "epsilon"),
            "theta": _uniform_from(bounds, "dichroic", "theta"),
        },
        "flc": {
            "phi": _uniform_from(bounds, "flc", "phi"),
            "delta_theta": {
                "type": "gaussian",
                "kwargs": {"mu": 0.0, "sigma": 5.0},
            },
        },
        "optics": {
            "phi": _uniform_from(bounds, "optics", "phi"),
            "epsilon": _uniform_from(bounds, "optics", "epsilon"),
            "theta": _uniform_from(bounds, "optics", "theta"),
        },
        "image_rotator": {
            "phi": _uniform_from(bounds, "image_rotator", "phi"),
        },
        "hwp": {
            "phi": _uniform_from(bounds, "hwp", "phi"),
            "delta_theta": {
                "type": "gaussian",
                "kwargs": {"mu": 0.0, "sigma": 5.0},
            },
        },
        "lp": {
            "theta": {
                "type": "gaussian",
                "kwargs": {"mu": 0.0, "sigma": 5.0},
            },
        },
    }


def _in_bounds(value, low, high):
    return low <= value <= high


def _wrap_to_bounds(value, low, high, period):
    wrapped = ((value - low) % period) + low
    if np.isclose(wrapped, low) and np.isclose(value, high):
        wrapped = high
    return wrapped


def _nudge_from_bound(value, low, high):
    width = high - low
    margin = max(abs(width) * 1e-6, 1e-8)
    if value - low < margin:
        return low + margin
    if high - value < margin:
        return high - margin
    return value


def prepare_p0_for_shared_priors(p0, bounds):
    """
    Return a p0 copy that satisfies the shared hard prior bounds.

    Phase-like ``phi`` values are wrapped by 2*pi. Dichroic and optics
    orientation angles use the requested [-90, 90] degree range. Values
    very close to a hard boundary are nudged just inside so the initial walker
    scatter starts with finite prior.
    """
    adjusted_p0 = copy.deepcopy(p0)
    adjustments = []
    nudges = []
    failures = []

    for component, param_bounds in bounds.items():
        if component not in adjusted_p0:
            continue
        for param, (low, high) in param_bounds.items():
            if param not in adjusted_p0[component]:
                continue

            value = float(adjusted_p0[component][param])
            adjusted_value = value

            if not _in_bounds(adjusted_value, low, high):
                if param == "phi":
                    wrapped = _wrap_to_bounds(adjusted_value, low, high, TWOPI)
                    if _in_bounds(wrapped, low, high):
                        adjusted_value = float(wrapped)
                        adjustments.append(
                            (f"{component}.{param}", value, adjusted_value, low, high)
                        )
                elif component in {"dichroic", "optics"} and param == "theta":
                    wrapped = _wrap_to_bounds(adjusted_value, low, high, 180.0)
                    if _in_bounds(wrapped, low, high):
                        adjusted_value = float(wrapped)
                        adjustments.append(
                            (f"{component}.{param}", value, adjusted_value, low, high)
                        )

            if not _in_bounds(adjusted_value, low, high):
                failures.append((f"{component}.{param}", value, low, high))
                continue

            nudged_value = _nudge_from_bound(adjusted_value, low, high)
            if nudged_value != adjusted_value:
                nudges.append(
                    (f"{component}.{param}", adjusted_value, nudged_value, low, high)
                )
                adjusted_value = nudged_value

            adjusted_p0[component][param] = float(adjusted_value)

    if failures:
        details = "\n".join(
            f"  {name}: {value} is outside [{low}, {high}]"
            for name, value, low, high in failures
        )
        raise ValueError(
            "Starting guesses are outside the shared hard prior bounds and "
            "cannot be fixed by phase/orientation wrapping:\n" + details
        )

    if adjustments:
        print("Wrapped starting guesses into shared prior bounds:")
        for name, old, new, low, high in adjustments:
            print(f"  {name}: {old:.8g} -> {new:.8g} within [{low:.8g}, {high:.8g}]")

    if nudges:
        print("Nudged boundary starting guesses inside shared prior bounds:")
        for name, old, new, low, high in nudges:
            print(f"  {name}: {old:.8g} -> {new:.8g} within [{low:.8g}, {high:.8g}]")

    return adjusted_p0
