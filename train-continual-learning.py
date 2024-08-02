import os

# this stops jax from stealing all the memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax
import jax.numpy as jnp
import jax.random as jrng
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

    elif cfg.task_style == "domain-incremental":
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

    elif cfg.task_style == "task-incremental":
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

    return train_loaders, test_loaders


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
    train_loaders, test_loaders = generate_tasks(cfg, rngs["tasks"])

    # setup model
    model = hydra.utils.instantiate(cfg.model)
    init_keys = {"params": rngs["model"]}

    # create optimizer
    opt = hydra.utils.instantiate(cfg.optimizer)
    # create training state (initialize parameters)
    dummy_input = jnp.ones((1, *cfg.data.shape))
    init_state = partial(TrainState.from_model, model, dummy_input, opt,
                         metrics=MultitaskMetrics.create(n=cfg.ntasks))
    # make sure no parameters are zero
    assert jnp.all(init_state(init_keys).params != 0)

    if cfg.nmodels > 1:
        train_state = jax.vmap(init_state)(init_keys)
    else:
        train_state = init_state(init_keys)
    # create training step
    loss_fn = optax.softmax_cross_entropy
    if "projectlib.ntk" in cfg.optimizer._target_:
        train_step = create_ntk_ensemble_train_step(loss_fn, cfg.ntk_use_current_params)
    else:
        train_step = create_train_step(loss_fn)
    @partial(jax.jit, static_argnums=3)
    def metric_step(state: TrainState, batch, _ = None, suffix = ""):
        xs, ys = batch
        ypreds = state.apply_fn(state.params, xs, rngs=state.rngs)
        loss = jnp.mean(loss_fn(ypreds, ys))
        acc = jnp.mean(jnp.argmax(ypreds, axis=-1) == jnp.argmax(ys, axis=-1))
        metrics_updates = state.metrics.single_from_model_output(
            **{f"loss{suffix}": loss, f"accuracy{suffix}": acc}
        )
        metrics = state.metrics.merge(metrics_updates)
        state = state.replace(metrics=metrics)

        return state
    if cfg.nmodels > 1:
        train_step = jax.vmap(train_step, in_axes=(0, None, 0))
        metric_step = jax.vmap(metric_step, in_axes=(0, None, 0))

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
    logger.log({"num_params":num_params})
    
    # run training
    trace = train_state.metrics.init_history()
    for i, train_loader in enumerate(train_loaders):
        print(f"\nRUNNING Task {i}...")

        train_state, trace = fit({"train": train_loader, "test": test_loaders,},
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

    # save final state
    ckpt = {"train_state": train_state, "metrics_history": trace}
    ckpt_mgr.save(cfg.training.nepochs, args=ocp.args.StandardSave(ckpt), force=True)
    ckpt_mgr.wait_until_finished()

    # close logger
    logger.finish()

if __name__ == "__main__":
    # prevent TF from using the GPU
    tf.config.experimental.set_visible_devices([], "GPU")
    # run training
    main()
