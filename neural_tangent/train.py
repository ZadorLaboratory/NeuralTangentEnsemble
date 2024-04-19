# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import logging


from flax.training import train_state
import jax 
import jax.numpy as jp
import numpy as np
import ml_collections
import optax

from models import MLP
from tasks import shuffled_MNIST

@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jp.mean(jp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

def train_epoch(state, train_ds):
  epoch_loss = []
  epoch_accuracy = []

  for batch_images, batch_labels in train_ds:
    grads, loss, accuracy = apply_model(state, batch_images.numpy(), batch_labels.numpy())
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy



def create_train_state(rng, config):
  model = MLP(
      input_dim=28*28,
      num_features=10000,
      num_hidden = 2,        
      num_classes=10
  )
  params = model.init(rng, jp.ones([1, 28, 28, 1]))['params']
  tx = optax.sgd(config.learning_rate, config.momentum)
  return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_and_evaluate(
    config: ml_collections.ConfigDict,
    state: train_state.TrainState,
    train_ds,
    test_ds,
    rng
) -> train_state.TrainState:


  for epoch in range(1, config.num_epochs+1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
       state, train_ds
    )

    test_batch_loss = {i: [] for i in range(len(test_ds))}
    test_batch_accuracy = {i: [] for i in range(len(test_ds))}
    test_loss = {i: [] for i in range(len(test_ds))}
    test_accuracy = {i: [] for i in range(len(test_ds))}


    for i, tds in enumerate(test_ds):
        for test_img, test_label in tds:
          _, loss, accuracy = apply_model(
             state, test_img.numpy(), test_label.numpy()
          )
          test_batch_loss[i].append(loss)
          test_batch_accuracy[i].append(accuracy)
    

    for i in range(len(test_ds)):
        test_loss[i] = np.mean(test_batch_loss[i])
        test_accuracy[i] = np.mean(test_batch_accuracy[i])

    logging.info(f'epoch: {epoch}, train_loss: {train_loss}, train_accuracy: {train_accuracy}, test_loss: {test_loss}, test_accuracy: {test_accuracy}')


  return state, {'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_loss': test_loss, 'test_accuracy': test_accuracy}

def train_multi_data(
    config: ml_collections.ConfigDict,
    workdir: str
) -> train_state.TrainState:
  rng = jax.random.key(0)
  rng, init_rng = jax.random.split(rng)
  all_accuracies = []
  
  state = create_train_state(init_rng, config)

  for round in range(config.num_rounds):
      rng, round_rng = jax.random.split(rng)
      train_loader, test_loader = shuffled_MNIST(config.batch_size)  # Create new shuffled dataset
  
      # Update all test loaders to include the new test set
      if round == 0:
          all_test_loaders = [test_loader]
          # start on a good foot
      else:
          all_test_loaders.append(test_loader)
  
      # Train the model and evaluate on all datasets
      state, accuracies = train_and_evaluate(config, state, train_loader, all_test_loaders, round_rng)
      all_accuracies.append(accuracies)