import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple, Any

from projectlib.utils import flatten

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = jnp.dtype
Array = jax.Array

class BaseModel(nn.Module):
    def has_batchnorm(self):
        return False

class SimpleMLP(BaseModel):
    """A simple MLP model."""
    features_per_layer: int
    nlayers: int
    nclasses: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = flatten(x)
        for _ in range(self.nlayers):
            x = nn.Dense(features=self.features_per_layer, dtype=self.dtype)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.nclasses)(x)

        return x

class SimpleCNN(BaseModel):
    """A simple CNN model."""
    features: Sequence[int]
    nclasses: int
    pooling_factor: int = 2
    use_bn: bool = False
    dtype: Dtype = jnp.float32

    def has_batchnorm(self):
        return self.use_bn

    @nn.compact
    def __call__(self, x, train = False):
        for feature in self.features:
            x = nn.Conv(features=feature,
                        kernel_size=(3, 3),
                        padding=1,
                        dtype=self.dtype)(x)
            if self.use_bn:
                x = nn.BatchNorm(use_running_average=not train,
                                 dtype=self.dtype)(x)
            else:
                x = nn.LayerNorm(dtype=self.dtype)(x)
            x = nn.relu(x)
            pool_size = (self.pooling_factor, self.pooling_factor)
            x = nn.avg_pool(x, window_shape=pool_size, strides=pool_size)
        x = flatten(x)
        x = nn.Dense(features=x.shape[-1] // 2, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.nclasses)(x)

        return x
