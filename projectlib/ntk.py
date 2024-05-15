import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrng
import optax
from flax.core import FrozenDict
from typing import NamedTuple
from functools import partial

from projectlib.models import Array

def perturb_params(rng, params, noise_fn = jrng.normal):
    leaves, structure = jtu.tree_flatten(params)
    keys = jrng.split(rng, len(leaves))
    leaves = (p + noise_fn(k, p.shape, p.dtype) for p, k in zip(leaves, keys))

    return jtu.tree_unflatten(structure, leaves)

def log_ntk_likelihood(sign_delta, grad, inv_temperature, eta):
    log_likelihood = eta * jnp.log1p( - inv_temperature * sign_delta * grad)
    return log_likelihood

class NTKEnsembleState(NamedTuple):
    sign_deltas: FrozenDict[str, Array]
    log_deltas: FrozenDict[str, Array]
    log_Z: float
    current_step: int

def switch_logic(val1, val2, switch):
    return switch * val1 + (1 - switch) * val2

def ntk_ensemble(inv_temperature, scale_param, seed, max_delta = None, eta = 1, reset_taylor_steps = 1):
    def init_fn(params):
        leaves, structure = jtu.tree_flatten(params)
        rng = jrng.PRNGKey(seed)
        keys = jrng.split(rng, len(leaves))

        leaves = [scale_param * jnp.sign(jrng.normal(k, p.shape, p.dtype))
                      for p, k in zip(leaves, keys)]
        leaves = [jnp.full_like(p, -scale_param) for p in leaves] # make all deltas negative; make the net smaller
        log_Z = jnp.log(sum(jnp.sum(jnp.abs(d)) for d in leaves))

        deltas = jtu.tree_unflatten(structure, leaves)
        sign_deltas = jtu.tree_map(lambda d: jnp.asarray(jnp.sign(d), dtype=jax.numpy.int8), deltas) # int8 but in theory could be a single bit per param, e.g. signbit plus later logic
        log_deltas = jtu.tree_map(jnp.log, jtu.tree_map(jnp.abs, deltas))
        
        return NTKEnsembleState(sign_deltas, log_deltas, log_Z, 0)

    def update_fn(grads, state: NTKEnsembleState, _ = None):
        # compute ntk likelihood
        log_likelihood_fn = partial(log_ntk_likelihood,
                                inv_temperature=inv_temperature,
                                eta=eta)
        log_likelihood = jtu.tree_map(log_likelihood_fn, state.sign_deltas, grads)
        # jax.debug.print("0 log_deltas max {x} min {y}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x), state.log_deltas))
        #                                                 , y=jtu.tree_reduce(jnp.minimum, jtu.tree_map(lambda x: jnp.min(x), state.log_deltas)))
        # vals = jtu.tree_map(lambda s,g: - s * g * inv_temperature, state.sign_deltas, grads)
        # jax.debug.print("0.1 (- sign_delta * grad) max {x} min {y}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x), vals))
        #                                                 , y=jtu.tree_reduce(jnp.minimum, jtu.tree_map(lambda x: jnp.min(x), vals)))
        # jax.debug.print("1 log_likelihood max {x} min {y}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x), log_likelihood))
        #                                                 , y=jtu.tree_reduce(jnp.minimum, jtu.tree_map(lambda x: jnp.min(x), log_likelihood)))
        # jax.debug.print("2 grad max sq {x}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x**2), grads)))
        # sum over batch
        log_likelihood = jtu.tree_map(lambda l: jnp.sum(l, axis=0), log_likelihood)
        # jax.debug.print("3 summed log_likelihood max {x} min {y}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x), log_likelihood))
        #                                                 , y=jtu.tree_reduce(jnp.minimum, jtu.tree_map(lambda x: jnp.min(x), log_likelihood)))       
        # compute update 
        log_deltas = jtu.tree_map(lambda ld, ll: ld + ll,
                                  state.log_deltas, log_likelihood)
        # jax.debug.print("4 log_deltas max {x} min {y}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x), log_deltas))
        #                                                 , y=jtu.tree_reduce(jnp.minimum, jtu.tree_map(lambda x: jnp.min(x), log_deltas)))
        # renormalize and clip
        log_current_norm = jax.nn.logsumexp(jnp.array([
            jax.nn.logsumexp(ll)
            for ll in jtu.tree_leaves(log_deltas)
        ]))
        # jax.debug.print("5 log_current_Z {x}", x=log_current_norm)
        # jax.debug.print("5.1 log_Z {x} max_norm {y}", x=state.log_Z, y=jnp.exp(state.log_Z))
        log_max_delta = jnp.log(max_delta)
        log_deltas = jtu.tree_map(
            lambda ld: jnp.clip(ld + state.log_Z - log_current_norm,
                                           a_max=log_max_delta),
            log_deltas,
        )
        # jax.debug.print("6 log_deltas max {x} min {y}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x), log_deltas))
        #                                                 , y=jtu.tree_reduce(jnp.minimum, jtu.tree_map(lambda x: jnp.min(x), log_deltas)))   
        # jax.debug.print("6.1 log_current_norm again {x}", x=jax.nn.logsumexp(jnp.array([
        #     jax.nn.logsumexp(ll)
        #     for ll in jtu.tree_leaves(log_deltas)
        # ])))                         
        deltas = jtu.tree_map(lambda ld, s: jnp.exp(ld) * s,
                                  log_deltas, state.sign_deltas)
        # jax.debug.print("7 deltas max {x} min {y}", x=jtu.tree_reduce(jnp.maximum, jtu.tree_map(lambda x: jnp.max(x), deltas))
        #                                                 , y=jtu.tree_reduce(jnp.minimum, jtu.tree_map(lambda x: jnp.min(x), deltas)))
        # reset the deltas to be equal?
        to_reset = state.current_step % reset_taylor_steps == 0
        new_state = init_fn(jtu.tree_map(jnp.zeros_like, deltas))
        switch_fn = partial(switch_logic, switch=to_reset)

        new_state = new_state._replace(sign_deltas = jtu.tree_map(switch_fn, new_state.sign_deltas, state.sign_deltas),
                                        log_deltas = jtu.tree_map(switch_fn, new_state.log_deltas, log_deltas),
                                        current_step = state.current_step + 1)

        return deltas, new_state

    return optax.GradientTransformation(init_fn, update_fn)
