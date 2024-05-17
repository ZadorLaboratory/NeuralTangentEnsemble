import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple, Any, Callable
from functools import partial

from projectlib.utils import flatten

ModuleDef = Any
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
    pool_final: bool = True
    dtype: Dtype = jnp.float32

    def has_batchnorm(self):
        return self.use_bn

    @nn.compact
    def __call__(self, x, train = False):
        pool_size = (self.pooling_factor, self.pooling_factor)
        for feature in self.features[:-1]:
            x = nn.Conv(features=feature,
                        kernel_size=(3, 3),
                        padding=1,
                        dtype=self.dtype)(x)
            # if self.use_bn:
            #     x = nn.BatchNorm(use_running_average=not train,
            #                      dtype=self.dtype)(x)
            # else:
            #     x = nn.LayerNorm(dtype=self.dtype)(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=pool_size, strides=pool_size)
        x = nn.Conv(features=self.features[-1],
                    kernel_size=(3, 3),
                    padding=1,
                    dtype=self.dtype)(x)
        x = nn.relu(x)
        if self.pool_final:
            x = nn.avg_pool(x, window_shape=pool_size, strides=pool_size)
        x = flatten(x)
        x = nn.Dense(features=x.shape[-1] // 2, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.nclasses)(x)

        return x

class ResNetBlock(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters, (1, 1), self.strides, name='conv_proj'
            )(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name='conv_proj'
            )(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNet(BaseModel):
    """ResNetV1.5."""

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    def has_batchnorm(self):
        return True

    @nn.compact
    def __call__(self, x, train: bool = False):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype
        )

        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name='conv_init',
        )(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                )(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlock)
ResNet50 = partial(
    ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
)
ResNet101 = partial(
    ResNet, stage_sizes=[3, 4, 23, 3], block_cls=BottleneckResNetBlock
)
ResNet152 = partial(
    ResNet, stage_sizes=[3, 8, 36, 3], block_cls=BottleneckResNetBlock
)
ResNet200 = partial(
    ResNet, stage_sizes=[3, 24, 36, 3], block_cls=BottleneckResNetBlock
)
