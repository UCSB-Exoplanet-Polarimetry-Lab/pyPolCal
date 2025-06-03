import numpy as np

def unflatten_p(params, keys):
    out = {}
    for (comp, param), val in zip(keys, params):
        out.setdefault(comp, {})[param] = val
    return out

def log_prob(theta, system_mm, dataset, errors, configuration_list, 
             p_keys, s_in, process_model, process_dataset, process_errors,
             prior_dict, bounds_dict, logl_function):
    lp = log_prior(theta, p_keys, prior_dict, bounds_dict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + logl_function(
        theta, p_keys, system_mm, dataset, errors, configuration_list,
        s_in=s_in,
        process_model=process_model,
        process_dataset=process_dataset,
        process_errors=process_errors
    )

def log_prior(theta, keys, prior_dict, bounds_dict):
    p_dict = unflatten_p(theta, keys)
    logp = 0
    for (comp, param), val in zip(keys, theta):
        if not (bounds_dict[comp][param][0] <= val <= bounds_dict[comp][param][1]):
            return -np.inf
        if comp in prior_dict and param in prior_dict[comp]:
            logp += prior_dict[comp][param](val)
    return logp

def gaussian_prior(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2

def uniform_prior(x, low, high):
    if low <= x <= high:
        return 0.0  # log(1) = 0
    return -np.inf
