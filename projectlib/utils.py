import jax
import jax.numpy as jnp
import jax.random as jrng
import hydra
import optax
from omegaconf import DictConfig

def noop(x):
    return x

def maybe(this, that):
    return that if this is None else this

def unzip(iter):
    if isinstance(iter, list):
        return tuple(list(x) for x in zip(*iter))
    else:
        return tuple(tuple(x) for x in zip(*iter))

def flatten(x):
    return jnp.reshape(x, (x.shape[0], -1))

def psplit(x, devices = jax.local_devices()):
    ndevices = len(devices)

    return jnp.reshape(x, (ndevices, x.shape[0] // ndevices, *x.shape[1:]))

def mse(predictions, targets):
    error = optax.squared_error(flatten(predictions), flatten(targets))

    return jnp.mean(error, axis=-1)

def setup_rngs(seed, keys = ["model", "train"]):
    root_rng = jrng.PRNGKey(seed) if not isinstance(seed, jax.Array) else seed
    rngs = {k: rng for k, rng in zip(keys, jrng.split(root_rng, len(keys)))}

    return {"root": root_rng, **rngs}

def _wrap_schedule(s):
    if isinstance(s, int) or isinstance(s, float):
        return optax.constant_schedule(s)
    else:
        return s

def instantiate_schedule(cfg: DictConfig, steps_per_epoch):
    if (cfg._target_ == "optax.exponential_decay") and cfg.staircase:
        transition_steps = cfg.transition_steps * steps_per_epoch
        return hydra.utils.instantiate(cfg, transition_steps=transition_steps)
    elif cfg._target_ == "optax.piecewise_constant_schedule":
        boundaries = {int(k * steps_per_epoch): v
                      for k, v in cfg.boundaries_and_scales.items()}
        return hydra.utils.instantiate(cfg, boundaries_and_scales=boundaries)
    elif cfg._target_ == "optax.join_schedules":
        schedules = [_wrap_schedule(s) for s in cfg.schedules]
        boundaries = [b * steps_per_epoch for b in cfg.boundaries]
        return optax.join_schedules(schedules, boundaries)
    else:
        return hydra.utils.instantiate(cfg)

def instantiate_optimizer(cfg: DictConfig, steps_per_epoch):
    opt_base = hydra.utils.get_method(cfg._target_)
    opt_factory = optax.inject_hyperparams(opt_base)
    opt_kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
    if isinstance(cfg.learning_rate, DictConfig):
        lr = instantiate_schedule(cfg.learning_rate, steps_per_epoch)
        opt_kwargs["learning_rate"] = lr

    return optax.chain(optax.clip(1), opt_factory(**opt_kwargs))
