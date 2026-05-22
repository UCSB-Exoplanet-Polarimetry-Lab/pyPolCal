import copy

import numpy as np


TWOPI = 2 * np.pi
DICHROIC_THETA_CENTER_DEG = 45.0


def build_shared_bounds():
    """Shared bounds used by the MBI adjusted-EM-gain strict-prior runs."""
    return {
        "dichroic": {
            "phi": (-0.5 * TWOPI, 0.5 * TWOPI),
            "epsilon": (0.0, 1),
            "theta": (-90.0, 90.0),
        },
        "flc": {
            "phi": (0 , TWOPI),
            "delta_theta": (-5.0, 5.0),
        },
        "optics": {
            "phi": (0, TWOPI),
            "epsilon": (0.0, 1),
            "theta": (-90.0, 90.0),
        },
        "image_rotator": {
            "phi": (0.0, TWOPI),
        },
        "hwp": {
            "phi": (0, TWOPI),
            "delta_theta": (-5.0, 5.0),
        },
        "lp": {
            "theta": (-5.0, 5.0),
        },
    }


def build_shared_prior_dict():
    """Shared prior family used for all wavelengths in the copied MBI runs."""
    bounds = build_shared_bounds()
    return {
        "dichroic": {
            "phi": {
                "type": "uniform",
                "kwargs": {
                    "low": bounds["dichroic"]["phi"][0],
                    "high": bounds["dichroic"]["phi"][1],
                },
            },
            "epsilon": {
                "type": "uniform",
                "kwargs": {
                    "low": bounds["dichroic"]["epsilon"][0],
                    "high": bounds["dichroic"]["epsilon"][1],
                },
            },
            "theta": {
                "type": "gaussian",
                "kwargs": {"mu": DICHROIC_THETA_CENTER_DEG, "sigma": 10.0},
            },
        },
        "flc": {
            "phi": {
                "type": "gaussian",
                "kwargs": {"mu": 0.50 * TWOPI, "sigma": 0.1 * TWOPI},
            },
            "delta_theta": {
                "type": "gaussian",
                "kwargs": {"mu": 0.0, "sigma": 1.0},
            },
        },
        "optics": {
            "phi": {
                "type": "uniform",
                "kwargs": {
                    "low": bounds["optics"]["phi"][0],
                    "high": bounds["optics"]["phi"][1],
                },
            },
            "epsilon": {
                "type": "uniform",
                "kwargs": {
                    "low": bounds["optics"]["epsilon"][0],
                    "high": bounds["optics"]["epsilon"][1],
                },
            },
            "theta": {
                "type": "gaussian",
                "kwargs": {"mu": 0.0, "sigma": 10.0},
            },
        },
        "image_rotator": {
            "phi": {
                "type": "uniform",
                "kwargs": {
                    "low": bounds["image_rotator"]["phi"][0],
                    "high": bounds["image_rotator"]["phi"][1],
                },
            },
        },
        "hwp": {
            "phi": {
                "type": "gaussian",
                "kwargs": {"mu": 0.50 * TWOPI, "sigma": 0.08 * TWOPI},
            },
            "delta_theta": {
                "type": "gaussian",
                "kwargs": {"mu": 0.0, "sigma": 0.75},
            },
        },
        "lp": {
            "theta": {
                "type": "gaussian",
                "kwargs": {"mu": 0.0, "sigma": 0.75},
            },
        },
    }


def _in_bounds(value, low, high):
    return low <= value <= high


def _wrap_to_bounds(value, low, high, period=TWOPI):
    wrapped = ((value - low) % period) + low
    if np.isclose(wrapped, low) and np.isclose(value, high):
        wrapped = high
    return wrapped


def prepare_p0_for_shared_priors(p0, bounds):
    """
    Return a p0 copy that satisfies hard prior bounds.

    Phase-like ``phi`` values are wrapped by 2*pi when that puts them inside the
    selected interval. Non-phase parameters are not silently changed.
    """
    adjusted_p0 = copy.deepcopy(p0)
    adjustments = []
    failures = []

    for component, param_bounds in bounds.items():
        if component not in adjusted_p0:
            continue
        for param, (low, high) in param_bounds.items():
            if param not in adjusted_p0[component]:
                continue

            value = float(adjusted_p0[component][param])
            if _in_bounds(value, low, high):
                continue

            if param == "phi":
                wrapped = _wrap_to_bounds(value, low, high)
                if _in_bounds(wrapped, low, high):
                    adjusted_p0[component][param] = float(wrapped)
                    adjustments.append(
                        (f"{component}.{param}", value, float(wrapped), low, high)
                    )
                    continue

            failures.append((f"{component}.{param}", value, low, high))

    if failures:
        details = "\n".join(
            f"  {name}: {value} is outside [{low}, {high}]"
            for name, value, low, high in failures
        )
        raise ValueError(
            "Starting guesses are outside the shared hard prior bounds and "
            "cannot be fixed by 2*pi phase wrapping:\n" + details
        )

    if adjustments:
        print("Wrapped phase starting guesses into shared prior bounds:")
        for name, old, new, low, high in adjustments:
            print(f"  {name}: {old:.8g} -> {new:.8g} within [{low:.8g}, {high:.8g}]")

    return adjusted_p0
