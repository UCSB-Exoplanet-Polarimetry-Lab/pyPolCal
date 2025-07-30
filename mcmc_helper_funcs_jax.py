import jax.numpy as jnp
from jax import jit

def unflatten_p(params, keys):
    out = {}
    for (comp, param), val in zip(keys, params):
        out.setdefault(comp, {})[param] = val
    return out

def log_prior(theta, keys, prior_dict, bounds_dict):
    logp = 0.0
    for (comp, param), val in zip(keys, theta):
        if not (bounds_dict[comp][param][0] <= val <= bounds_dict[comp][param][1]):
            return -jnp.inf

        prior_info = prior_dict[comp][param]
        prior_type = prior_info["type"]
        kwargs = prior_info["kwargs"]

        if prior_type == "uniform":
            logp += uniform_prior(val, **kwargs)
        elif prior_type == "gaussian":
            logp += gaussian_prior(val, **kwargs)
        else:
            raise ValueError(f"Unsupported prior type: {prior_type}")
    return logp

def log_prob(theta, system_mm, dataset, errors, configuration_list, 
             p_keys, s_in, process_model, process_dataset, process_errors,
             prior_dict, bounds_dict, logl_function, mode):
    lp = log_prior(theta, p_keys, prior_dict, bounds_dict)
    log_l = logl_function(
        theta, system_mm, dataset, errors, configuration_list,
        p_keys, s_in, process_model, process_dataset, process_errors,mode
    )
    return jnp.where(jnp.isfinite(lp), lp + log_l, -jnp.inf)

@jit
def uniform_prior(x, low, high):
    return jnp.where((x >= low) & (x <= high), 0.0, -jnp.inf)

@jit
def gaussian_prior(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2