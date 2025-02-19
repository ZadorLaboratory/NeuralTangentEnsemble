import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
import flax.jax_utils
import orbax.checkpoint as ocp
from clu import metrics
from flax import struct
from flax.training import train_state
from typing import Dict, Any, Optional
import wandb
import optax

from projectlib.utils import maybe
from projectlib.logging import PrintLogger

Array = Any

def batch_values(batch):
    if isinstance(batch, Dict):
        return tuple(batch.values())
    else:
        return batch

def get_hparams(opt_state):
    if isinstance(opt_state, tuple):
        return opt_state[-1].hyperparams
    else:
        return opt_state.hyperparams

@struct.dataclass
class MetricCollection(metrics.Collection):
    @classmethod
    def create(cls, **metrics: type[metrics.Metric]):
        return flax.struct.dataclass(
            type("_InlineCollection", (cls,), {"__annotations__": metrics})
        )

    @classmethod
    def _from_model_output(cls: type[metrics.C], **kwargs) -> metrics.C:
        """Creates a `Collection` from model outputs."""
        return cls(
            _reduction_counter=metrics._ReductionCounter(jnp.array(1, dtype=jnp.int32)),
            **{
                name: metric.from_model_output(**kwargs) if name in kwargs
                      else metric.empty()
                for name, metric in cls.__annotations__.items()
            })

    def init_history(self):
        return {
            "train": {k: [] for k in self.__annotations__.keys()},
            "test": {k: [] for k in self.__annotations__.keys()}
        }

    def _get_internal_array(self):
        arr = getattr(self, next(iter(self.__annotations__.keys())))

        return arr.count

    def compute(self):
        arr = self._get_internal_array()
        if len(arr.devices()) > 1:
            return super(MetricCollection, self.unreplicate()).compute()
        elif arr.ndim > 0:
            return jax.vmap(lambda m: super(MetricCollection, m).compute())(self)
        else:
            return super(MetricCollection, self).compute()

    def reempty(self):
        arr = self._get_internal_array()
        devices = list(arr.devices())
        if len(devices) > 1:
            empty_metric = super(MetricCollection, self).empty()
            return flax.jax_utils.replicate(empty_metric, devices)
        elif arr.ndim > 0:
            return jax.vmap(lambda m: super(MetricCollection, m).empty())(self)
        else:
            return super(MetricCollection, self).empty()

@struct.dataclass
class Metrics(MetricCollection):
    accuracy: metrics.Average.from_output('accuracy') # type: ignore
    loss: metrics.Average.from_output('loss') # type: ignore

@struct.dataclass
class MultitaskMetrics(MetricCollection):
    @classmethod
    def create(cls, n = 1):
        return MetricCollection.create(
            **{f"accuracy_{i}": metrics.Average.from_output(f"accuracy_{i}")
               for i in range(n)},
            **{f"loss_{i}": metrics.Average.from_output(f"loss_{i}")
               for i in range(n)}
        )

def replace_zeros_with_noise(param, noise_level = 1e-3):
    # Create a random key
    key = jax.random.PRNGKey(0)
    # Create a mask of the same shape as the parameter, which is True where the parameter is zero
    mask = jnp.equal(param, 0)
    # Generate random noise of the same shape as the parameter
    noise = jax.random.normal(key, param.shape) * noise_level
    # Replace zeros in the parameter with the corresponding values from the noise
    return jnp.where(mask, noise, param)

class TrainState(train_state.TrainState):
    metrics: Optional[MetricCollection] = None
    rngs: Dict[str, Array] = struct.field(default_factory=dict)
    current_step: int = 0

    @classmethod
    def from_model(cls, model, dummy_input, opt, rngs, metrics = Metrics, param_init = None):
        _init = model.init if param_init is None else param_init
        if isinstance(dummy_input, tuple):
            params = _init(rngs, *dummy_input)
        else:
            params = _init(rngs, dummy_input)

        params = jtu.tree_map(replace_zeros_with_noise, params)

        return cls.create(apply_fn=model.apply,
                          params=params,
                          tx=opt,
                          rngs=rngs,
                          metrics=metrics.empty())

    def split_rngs(self):
        def _split(key):
            if len(list(key.devices())) > 1:
                return jax.pmap(jrng.split,
                                static_broadcasted_argnums=1)(key, 1)[:, 0, :]
            elif key.ndim > 1:
                return jax.vmap(jrng.split)(key)[:, 0, :]
            else:
                return jrng.split(key, 1)[0]
        rngs = {k: _split(v) for k, v in self.rngs.items()}

        return self.replace(rngs=rngs)

    def reset(self, tx = None):
        devices = list(jtu.tree_leaves(self.params)[0].devices())
        if len(devices) > 1:
            state = flax.jax_utils.unreplicate(self)
        else:
            state = self
        _tx = state.tx if tx is None else tx
        opt_state = _tx.init(state.params)
        state = state.replace(step=0, tx=_tx, opt_state=opt_state)
        if len(devices) > 1:
            state = flax.jax_utils.replicate(state, devices)

        return state

def create_train_step(loss_fn, batch_stats = False):
    # if batch_stats are calculated, then we need to augment the apply_fn
    if batch_stats:
        @jax.jit
        def train_step(state: TrainState, batch, softmax_mask, _ = None):
            *xs, ys = batch
            def compute_loss(params, mask):
                yhats, aux = state.apply_fn(params, *xs,
                                            rngs=state.rngs,
                                            train=True,
                                            mutable=["batch_stats"])

                return jnp.mean(loss_fn(yhats, ys, mask)), aux

            grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
            (loss, aux), grads = grad_fn(state.params, softmax_mask)
            state = state.apply_gradients(grads=grads)
            state = state.replace(params=state.params.copy(aux))

            return loss, state, state.params
    else:
        @jax.jit
        def train_step(state: TrainState, batch, softmax_mask, _ = None):
            *xs, ys = batch
            def compute_loss(params, mask):
                yhats = state.apply_fn(params, *xs, rngs=state.rngs)

                return jnp.mean(loss_fn(yhats, ys, mask))

            grad_fn = jax.value_and_grad(compute_loss)
            loss, grads = grad_fn(state.params, softmax_mask)
            state = state.apply_gradients(grads=grads)

            return loss, state

    return train_step


def create_ntk_ensemble_train_step(loss_fn, use_current_params = True, batch_stats = False):
    # pull this out to avoid branching inside core train step
    if use_current_params:
        def ntk_diff_center(current_params, deltas):
            # recompute inital parameters
            init_params = jtu.tree_map(lambda p, d: p - d, current_params, deltas)

            return current_params, init_params
    else:
        def ntk_diff_center(current_params, deltas):
            # recompute inital parameters
            init_params = jtu.tree_map(lambda p, d: p - d, current_params, deltas)

            return init_params, init_params
    # if batch_stats are calculated, then we need to augment the apply_fn
    if batch_stats:
        @jax.jit
        def train_step(state: TrainState, batch, softmax_mask, _ = None):
            xs, ys = batch
            def compute_loss(params, xs, ys, mask):
                xs = jnp.expand_dims(xs, axis=0) # add dummy batch dim
                yhats, aux = state.apply_fn(params, xs,
                                            rngs=state.rngs,
                                            train=True,
                                            mutable=["batch_stats"])
                yhats = yhats[0]

                return loss_fn(yhats, ys, mask), aux

            # recompute inital parameters + choose ntk diff center
            ntk_params, init_params = ntk_diff_center(state.params, state.opt_state.deltas)
            # broadcast mask to match batch size
            softmax_mask = jnp.broadcast_to(softmax_mask, (xs.shape[0], *softmax_mask.shape))
            # compute per example gradients around initial parameters
            grad_fn = jax.vmap(jax.value_and_grad(compute_loss, has_aux=True),
                               in_axes=(None, 0, 0, 0))
            (loss, aux), grads = grad_fn(ntk_params, xs, ys, softmax_mask)
            # average loss over samples
            loss = jnp.mean(loss)
            # compute ntk ensemble updates
            state = state.replace(params=init_params)
            state = state.apply_gradients(grads=grads)
            # update batch_stats
            state = state.replace(params=state.params.copy(aux))

            return loss, state
    else:
        @jax.jit
        def train_step(state: TrainState, batch, softmax_mask, _ = None):
            xs, ys = batch
            def compute_loss(params, xs, ys, mask):
                xs = jnp.expand_dims(xs, axis=0) # add dummy batch dim
                yhats = state.apply_fn(params, xs, rngs=state.rngs)[0]

                return loss_fn(yhats, ys, mask)

            # recompute inital parameters + choose ntk diff center
            ntk_params, init_params = ntk_diff_center(state.params, state.opt_state.deltas)
            # broadcast mask to match batch size
            softmax_mask = jnp.broadcast_to(softmax_mask, (xs.shape[0], *softmax_mask.shape))
            # compute per example gradients around initial parameters
            grad_fn = jax.vmap(jax.value_and_grad(compute_loss),
                               in_axes=(None, 0, 0, 0))
            loss, grads = grad_fn(ntk_params, xs, ys, softmax_mask)
            # average loss over samples
            loss = jnp.mean(loss)
            # compute ntk ensemble updates
            state = state.replace(params=init_params)
            state = state.apply_gradients(grads=grads)

            return loss, state

    return train_step

def evaluate_metrics(state: TrainState, loaders, metrics_fn, rng, softmax_masks, rng_split = jrng.split):
    loaders = loaders if isinstance(loaders, list) else [loaders]
    for i, loader in enumerate(loaders):
        for batch in loader.as_numpy_iterator():
            batch = batch_values(batch)
            rng, rng_metric = rng_split(rng)
            state = state.split_rngs()
            state = metrics_fn(state, batch, softmax_masks[i], rng_metric, suffix=f"_{i}")

    return state

def fit(data, state: TrainState, step_fn, metrics_fn,
        rng = None,
        save_fn = None,
        nepochs = 1,
        nsteps = None,
        start_epoch = 0,
        epoch_logging = True,
        step_log_interval = 100,
        logger = PrintLogger(),
        metric_history = None):
    metric_history = maybe(metric_history, state.metrics.init_history())
    rng = maybe(rng, jrng.PRNGKey(0))

    # vmap helpers for multiple parallel models
    if rng.ndim > 1:
        def rng_split(key):
            splits = jax.vmap(jrng.split)(key)
            return splits[:, 0, :], splits[:, 1, :]
    else:
        def rng_split(key):
            return jrng.split(key)

    # evaluate initial train metrics
    test_state = evaluate_metrics(state, data["train"], metrics_fn, rng, data["test_masks"], rng_split)
    # average metrics
    for metric, value in test_state.metrics.compute().items():
        metric_history["train"][metric].append(value)
    # evaluate initial test metrics
    if "test" in data.keys():
        test_state = evaluate_metrics(state, data["train"], metrics_fn, rng, data["test_masks"], rng_split)
        # average metrics
        for metric, value in test_state.metrics.compute().items():
            metric_history["test"][metric].append(value)
    # save initial metrics
    ckpt = {"train_state": state, "metrics_history": metric_history}
    save_fn(start_epoch, args=ocp.args.StandardSave(ckpt), force=True)

    if nsteps is not None:
        nepochs = int(jnp.ceil(nsteps / len(data["train"])))


    epoch_len = len(data["train"])
    current_step = 0
    for epoch in range(start_epoch, start_epoch + nepochs):
        # run epoch
        for i, batch in enumerate(data["train"].as_numpy_iterator()):
            batch = batch_values(batch)
            rng, rng_step = rng_split(rng)
            rng_step, rng_metric = rng_split(rng_step)
            state = state.split_rngs()
            loss, state,  = step_fn(state, batch, data['train_mask'], rng_step)
            state = metrics_fn(state, batch, data['train_mask'], rng_metric)
            state = state.replace(current_step=(state.current_step + 1))
            current_step += 1
            if (step_log_interval is not None) and (i % step_log_interval == 0):
                logger.log({"epoch": epoch, "step": i, "loss": loss},
                           commit=(i < epoch_len - 1))
            if nsteps is not None and current_step >= nsteps:
                break

        # average metrics
        for metric, value in state.metrics.compute().items():
            metric_history["train"][metric].append(value)
        state = state.replace(metrics=state.metrics.reempty())

        # run test validation
        if "test" in data.keys():
            test_state = evaluate_metrics(state, data["test"], metrics_fn, rng, data["test_masks"],rng_split)
            # average metrics
            for metric, value in test_state.metrics.compute().items():
                metric_history["test"][metric].append(value)

        # run save function
        ckpt = {"train_state": state, "metrics_history": metric_history}
        save_fn(epoch, args=ocp.args.StandardSave(ckpt))

        # log outputs
        if epoch_logging == "summary":
            train_logs = {k: jnp.mean(v[-1])
                          for k, v in metric_history["train"].items()
                          if len(v) > 0}
            if "test" in data.keys():
                test_logs = {k: jnp.mean(v[-1])
                             for k, v in metric_history["test"].items()
                             if len(v) > 0}
            else:
                test_logs = {}
        elif epoch_logging:
            train_logs = {k: v[-1] for k, v in metric_history["train"].items()
                                   if len(v) > 0}
            if "test" in data.keys():
                test_logs = {k: v[-1] for k, v in metric_history["test"].items()
                                      if len(v) > 0}
            else:
                test_logs = {}
        else:
            train_logs = {}
            test_logs = {}

        if start_epoch == 0:
            logger.log({"epoch": epoch,
                    "step": epoch_len - 1,
                    # "hparams": get_hparams(state.opt_state),
                    "train metrics": train_logs,
                    "test metrics": test_logs,
                    "single_task": test_logs})
        else:
            logger.log({"epoch": epoch,
                    "step": epoch_len - 1,
                    # "hparams": get_hparams(state.opt_state),
                    "train metrics": train_logs,
                    "test metrics": test_logs})

    return state, metric_history
