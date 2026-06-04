"""
mcmc_helper_funcs.py

Helper functions for MCMC sampling using JAX.

"""




import numpy as np

def unflatten_p(params, keys):
    out = {}
    for (comp, param), val in zip(keys, params):
        out.setdefault(comp, {})[param] = val
    return out

def log_prior(theta, keys, prior_dict, bounds_dict):
    logp = 0.0
    for (comp, param), val in zip(keys, theta):
        # Special-case: hard-code a uniform prior/bounds for the synthetic
        # log_f parameter (component '__log_f', parameter 'log_f'). This
        # ensures log_f is constrained to [-5, 2] even if the caller did not
        # include it in prior_dict or bounds_dict.
        if comp == "__log_f" and param == "log_f":
            low, high = -5.0, 2.0
            if not (low <= val <= high):
                return -np.inf
            # uniform log-prior inside bounds contributes 0.0 to logp
            logp += 0.0
            continue

        # Regular handling for other parameters. If bounds/prior entries are
        # missing, allow the normal exceptions to surface so the user is made
        # aware of mis-specified prior/bounds.
        if not (bounds_dict[comp][param][0] <= val <= bounds_dict[comp][param][1]):
            return -np.inf

        prior_info = prior_dict[comp][param]
        prior_type = prior_info["type"]
        kwargs = prior_info.get("kwargs", {})

        if prior_type == "uniform":
            lp = uniform_prior(val, **kwargs)
        elif prior_type == "gaussian":
            lp = gaussian_prior(val, **kwargs)
        else:
            raise ValueError(f"Unsupported prior type: {prior_type}")
        logp += lp
    return logp

def log_prob(theta, system_mm, dataset, errors, configuration_list, 
             p_keys, s_in, process_model, process_dataset, process_errors,
             prior_dict, bounds_dict, logl_function, mode):
    lp = log_prior(theta, p_keys, prior_dict, bounds_dict)
    log_l = logl_function(
        theta, system_mm, dataset, errors, configuration_list,
        p_keys, s_in, process_model, process_dataset, process_errors, mode
    )
    if np.isfinite(lp):
        return lp + log_l
    else:
        return -np.inf

def uniform_prior(x, low, high):
    return 0.0 if (x >= low and x <= high) else -np.inf

def gaussian_prior(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2