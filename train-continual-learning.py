import os

# this stops jax from stealing all the memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import jax.random as jrng
import jax.tree_util as jtu
from jax.nn import log_softmax

import tensorflow as tf
import tensorflow_datasets as tfds
import optax
import hydra
import orbax.checkpoint as ocp
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from functools import partial, reduce
from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions
from clu.preprocess_spec import PreprocessFn

from projectlib.utils import setup_rngs
from projectlib.data import (build_dataloader,
                             select_class_subset_and_make_contiguous,
                             select_class_subset,
                             default_data_transforms,
                             force_dataset_length,
                             StaticShuffle)
from projectlib.training import (MultitaskMetrics,
                                 TrainState,
                                 batch_values,
                                 create_train_step,
                                 create_ntk_ensemble_train_step,
                                 fit)

def generate_tasks(cfg, rng):
    data = tfds.load(cfg.data.dataset)
    perm_size = reduce(lambda x, y: x * y, cfg.data.shape)

    if cfg.task_style == "shuffle":
        base_preprocess = default_data_transforms(cfg.data.dataset)

        shuffle_preprocess = [PreprocessFn([StaticShuffle(perm_size, i)], only_jax_types=True)
                              for i in range(cfg.ntasks)]
        train_loaders = [build_dataloader(
            data["train"],
            batch_transform=base_preprocess + shuffle_preprocess[i],
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]
        test_loaders = [build_dataloader(
            data["test"],
            batch_transform=base_preprocess + shuffle_preprocess[i],
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]
        class_subsets = [jnp.arange(len(cfg.data.classes)) for _ in range(cfg.ntasks)]

    elif cfg.task_style == "domain-incremental": # Just len(cfg.data.classes) // cfg.ntasks logits
        base_preprocess = default_data_transforms(cfg.data.dataset, len(cfg.data.classes) // cfg.ntasks)
        assert len(cfg.data.classes) % cfg.ntasks == 0, "Number of classes must be divisible by number of tasks"

        class_subsets = jrng.permutation(rng, jnp.asarray(cfg.data.classes),
                                         independent=True)
        class_subsets = jnp.array_split(class_subsets, cfg.ntasks)
        train_loaders = [build_dataloader(
            force_dataset_length(select_class_subset_and_make_contiguous(data["train"],
                                                     class_subsets[i])),
            batch_transform=base_preprocess,
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]
        test_loaders = [build_dataloader(
            force_dataset_length(select_class_subset_and_make_contiguous(data["test"],
                                                     class_subsets[i])),
            batch_transform=base_preprocess,
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]

    elif cfg.task_style == "task-incremental": # All logits, but we mask out the ones we don't want
        base_preprocess = default_data_transforms(cfg.data.dataset, len(cfg.data.classes))
        class_subsets = jrng.permutation(rng, jnp.asarray(cfg.data.classes),
                                         independent=True)
        class_subsets = jnp.array_split(class_subsets, cfg.ntasks)    
        train_loaders = [build_dataloader(
            force_dataset_length(select_class_subset(data["train"],
                                                     class_subsets[i])),
            batch_transform=base_preprocess,
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]
        test_loaders = [build_dataloader(
            force_dataset_length(select_class_subset(data["test"],
                                                     class_subsets[i])),
            batch_transform=base_preprocess,
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]

    elif cfg.task_style == "class-incremental": # All logits, no masking.
        base_preprocess = default_data_transforms(cfg.data.dataset, len(cfg.data.classes))
        class_subsets = jrng.permutation(rng, jnp.asarray(cfg.data.classes),
                                         independent=True)
        class_subsets = jnp.array_split(class_subsets, cfg.ntasks)    
        train_loaders = [build_dataloader(
            force_dataset_length(select_class_subset(data["train"],
                                                     class_subsets[i])),
            batch_transform=base_preprocess,
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]
        test_loaders = [build_dataloader(
            force_dataset_length(select_class_subset(data["test"],
                                                     class_subsets[i])),
            batch_transform=base_preprocess,
            batch_size=cfg.data.batchsize
        ) for i in range(cfg.ntasks)]

        class_subsets = [jnp.arange(len(cfg.data.classes)) for _ in range(cfg.ntasks)]


    return train_loaders, test_loaders, class_subsets

@jax.jit
def masked_cross_entropy(logits, labels, mask):
    # Compute softmax only over unmasked elements
    log_probs = log_softmax(logits, where=mask)
    return -jnp.sum(labels * log_probs, axis=-1)

def grad_alignment(loss_fn, batch, initial_params, state, softmax_mask):
    xs, ys = batch
    def compute_loss(params, mask):
        yhats = state.apply_fn(params, xs, rngs=state.rngs)

        return jnp.mean(loss_fn(yhats, ys, mask))
    # compute initial gradient
    grad_fn = jax.grad(compute_loss)
    init_grad = grad_fn(initial_params, softmax_mask)
    # compute current gradient
    current_grad = grad_fn(state.params, softmax_mask)
    # compute alignment
    current_grad = jnp.concatenate([jnp.reshape(p, -1)
                                    for p in jtu.tree_leaves(current_grad)], axis=0)
    init_grad = jnp.concatenate([jnp.reshape(p, -1)
                                    for p in jtu.tree_leaves(init_grad)], axis=0)
    align = optax.cosine_distance(current_grad, init_grad, epsilon=1e-8)

    return align

@hydra.main(config_path="./configs", config_name="train-continual-learning", version_base=None)
def main(cfg: DictConfig):
    if isinstance(cfg.gpu, ListConfig):
        # select device(s) for parallel training
        devices = [jax.devices()[i] for i in cfg.gpu]
    else:
        # use specific GPU of machine
        devices = [jax.devices()[cfg.gpu]]
    jax.config.update("jax_default_device", devices[0])

    # setup rngs
    if cfg.nmodels > 1:
        seeds = jrng.split(jrng.PRNGKey(cfg.seed), cfg.nmodels + 1)
        rngs = jax.vmap(setup_rngs)(seeds[:-1])
        rngs["tasks"] = seeds[-1] # task rng is same across models
    else:
        rngs = setup_rngs(cfg.seed, keys=["model", "train", "tasks"])
    # initialize randomness
    tf.random.set_seed(cfg.seed) # deterministic data iteration

    # setup dataloaders
    train_loaders, test_loaders, class_subsets = generate_tasks(cfg, rngs["tasks"])

    # setup model
    model = hydra.utils.instantiate(cfg.model)
    init_keys = {"params": rngs["model"]}

    # create optimizer
    opt = hydra.utils.instantiate(cfg.optimizer)
    # create training state (initialize parameters)
    dummy_input = jnp.ones((1, *cfg.data.shape))
    init_state = partial(TrainState.from_model, model, dummy_input, opt,
                         metrics=MultitaskMetrics.create(n=cfg.ntasks))

    if cfg.nmodels > 1:
        train_state = jax.vmap(init_state)(init_keys)
    else:
        train_state = init_state(init_keys)
    # make sure no parameters are zero
    assert jnp.all(train_state.params != 0)
    # create training step
    loss_fn = masked_cross_entropy
    if "projectlib.ntk" in cfg.optimizer._target_:
        train_step = create_ntk_ensemble_train_step(loss_fn, cfg.ntk_use_current_params)
        def compute_init_params(state):
            return jtu.tree_map(lambda p, d: p - d, state.params, state.opt_state.deltas)
    else:
        train_step = create_train_step(loss_fn)
        def compute_init_params(state):
            return state.params
    # create metric step
    @partial(jax.jit, static_argnums=4)
    def metric_step(state: TrainState, batch, softmax_mask, _ = None, suffix = ""):
        xs, ys = batch
        ypreds = state.apply_fn(state.params, xs, rngs=state.rngs)
        loss = jnp.mean(loss_fn(ypreds, ys, softmax_mask))
        acc = jnp.mean(jnp.argmax(ypreds, axis=-1) == jnp.argmax(ys, axis=-1))
        metrics_updates = state.metrics.single_from_model_output(
            **{f"loss{suffix}": loss, f"accuracy{suffix}": acc}
        )
        metrics = state.metrics.merge(metrics_updates)
        state = state.replace(metrics=metrics)

        return state
    if cfg.nmodels > 1:
        train_step = jax.vmap(train_step, in_axes=(0, None, None, 0))
        metric_step = jax.vmap(metric_step, in_axes=(0, None, None, 0))

    # create checkpointing utility
    ckpt_opts = CheckpointManagerOptions(
        create=True,
        save_interval_steps=cfg.checkpointing.rate,
        max_to_keep=2
    )
    ckpt_path = os.sep.join([os.getcwd(), cfg.checkpointing.path])
    ckpt_mgr = CheckpointManager(ckpt_path,
                                 options=ckpt_opts,
                                 metadata=OmegaConf.to_container(cfg))

    # create logger
    logger = hydra.utils.instantiate(cfg.logger)
    logger.log_config(cfg)

    # log num params
    num_params = sum(jnp.prod(jnp.array(p.shape)) for p in jax.tree.leaves(train_state.params))
    logger.log({"num_params": num_params})

    task_masks = [jnp.zeros(len(cfg.data.classes), dtype=jnp.float32).at[active_classes].set(1.0)
                  for active_classes in class_subsets]

    # run training
    trace = train_state.metrics.init_history()
    for i, train_loader in enumerate(train_loaders):
        print(f"\nRUNNING Task {i}...")

        train_state, trace = fit({"train": train_loader, "test": test_loaders,
                                  "train_mask": task_masks[i], "test_masks": task_masks},
                                 train_state,
                                 train_step,
                                 partial(metric_step, suffix=f"_{i}"),
                                 save_fn=ckpt_mgr.save,
                                 rng=rngs["train"],
                                 nepochs=cfg.training.nepochs,
                                 nsteps=cfg.training.nsteps,
                                 start_epoch=(i * cfg.training.nepochs),
                                 metric_history=trace,
                                 logger=logger,
                                 epoch_logging=cfg.training.log_epochs,
                                 step_log_interval=cfg.training.log_interval)

    # compute alignment with initial params
    if "projectlib.ntk" in cfg.optimizer._target_:
        avg_alignment = 0
        for batch in test_loaders[-1].as_numpy_iterator():
            batch = batch_values(batch)
            init_params = compute_init_params(train_state)
            align = grad_alignment(loss_fn, batch, init_params, train_state, task_masks[-1])
            avg_alignment += align
        avg_alignment = avg_alignment / len(test_loaders[-1])
        print("Average NTK alignment =", avg_alignment)

    # save final state
    if "projectlib.ntk" in cfg.optimizer._target_:
        ckpt = {"train_state": train_state,
                "metrics_history": trace,
                "ntk_alignment": avg_alignment}
    else:
        ckpt = {"train_state": train_state,
                "metrics_history": trace}
    ckpt_mgr.save(cfg.ntasks * cfg.training.nepochs,
                  args=ocp.args.StandardSave(ckpt), force=True)
    ckpt_mgr.wait_until_finished()

    # close logger
    logger.finish()

if __name__ == "__main__":
    # prevent TF from using the GPU
    tf.config.experimental.set_visible_devices([], "GPU")
    # run training
    main()
