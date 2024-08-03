import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple, Any, Callable, Optional, Union, Iterable
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
    width_multiplier: int
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
            x = nn.Conv(features=feature * self.width_multiplier,
                        kernel_size=(3, 3),
                        padding=1,
                        dtype=self.dtype)(x)
            if self.use_bn:
                x = nn.BatchNorm(use_running_average=not train,
                                 dtype=self.dtype)(x)

            x = nn.avg_pool(x, window_shape=pool_size, strides=pool_size)
        x = nn.Conv(features=self.features[-1] * self.width_multiplier,
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
    nclasses: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv

    def has_batchnorm(self):
        return False  # Changed to False since we're using LayerNorm now

    @nn.compact
    def __call__(self, x, train: bool = False):
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            nn.LayerNorm,
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
        x = norm(name='ln_init')(x)
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
        x = nn.Dense(self.nclasses, dtype=self.dtype)(x)
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


class DepthwiseConv2D(nn.Module):
    kernel_shape: Union[int, Sequence[int]] = (1, 1)
    stride: Union[int, Sequence[int]] = (1, 1)
    padding: Any = "SAME"
    channel_multiplier: int = 1
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input):
        w = self.param(
            "kernel",
            self.weights_init,
            self.kernel_shape + (1, self.channel_multiplier * input.shape[-1]),
        )
        if self.use_bias:
            b = self.param(
                "bias", self.bias_init, (self.channel_multiplier * input.shape[-1],)
            )

        conv = jax.lax.conv_general_dilated(
            input,
            w,
            self.stride,
            self.padding,
            (1,) * len(self.kernel_shape),
            (1,) * len(self.kernel_shape),
            ("NHWC", "HWIO", "NHWC"),
            input.shape[-1],
        )
        if self.use_bias:
            bias = jnp.broadcast_to(b, conv.shape)
            return conv + bias
        else:
            return conv

initializer = nn.initializers.variance_scaling(
    0.2, "fan_in", distribution="truncated_normal"
)

import jax.random as random

class DropPath(nn.Module):
    """
    Implementation referred from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    dropout_prob: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        if deterministic:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = keep_prob + random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor

class ConvNeXtBlock(nn.Module):
    dim: int = 256
    layer_scale_init_value: float = 1e-6
    drop_path: float = 0.1
    deterministic: Optional[bool] = None

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = DepthwiseConv2D((7, 7), weights_init=initializer, name="dwconv")(inputs)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(4 * self.dim, kernel_init=initializer, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, kernel_init=initializer, name="pwconv2")(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", self.init_fn, (self.dim,), self.layer_scale_init_value
            )
            x = gamma * x

        x = inputs + DropPath(self.drop_path)(x, deterministic)
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt Module

    Attributes:

        depths (list or tuple): Depths for every block
        dims (list or tuple): Embedding dimension for every stage.
        drop_path (float): Dropout value for DropPath. Default is 0.1
        layer_scale_init_value (float): Initialization value for scale. Default is 1e-6.
        head_init_scale (float): Initialization value for head. Default is 1.0.
        attach_head (bool): Whether to attach classification head. Default is False.
        nclasses (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.

    """

    depths: Iterable = (3, 3, 9, 3)
    dims: Iterable = ( 12, 24, 48, 96)
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_head: bool = True
    nclasses: int = 10
    deterministic: Optional[bool] = None
    width_multiplier: int = 1

    @nn.compact
    def __call__(self, inputs, deterministic=None):

        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        dp_rates = jnp.linspace(0, self.drop_path, sum(self.depths))
        curr = 0

        # Stem
        x = nn.Conv(
            self.dims[0] * self.width_multiplier, (4, 4), 4, kernel_init=initializer, name="downsample_layers00"
        )(inputs)
        x = nn.LayerNorm(name="downsample_layers01")(x)

        for j in range(self.depths[0]):
            x = ConvNeXtBlock(
                self.dims[0] * self.width_multiplier,
                drop_path=dp_rates[curr + j],
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"stages0{j}",
            )(x, deterministic)
        curr += self.depths[0]

        # Downsample layers
        for i in range(3):
            x = nn.LayerNorm(name=f"downsample_layers{i+1}0")(x)
            x = nn.Conv(
                self.dims[i + 1] * self.width_multiplier,
                (2, 2),
                2,
                kernel_init=initializer,
                name=f"downsample_layers{i+1}1",
            )(x)

            for j in range(self.depths[i + 1]):
                x = ConvNeXtBlock(
                    self.dims[i + 1] * self.width_multiplier,
                    drop_path=dp_rates[curr + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    name=f"stages{i+1}{j}",
                )(x, deterministic)

            curr += self.depths[i + 1]

        if self.attach_head:
            x = nn.LayerNorm(name="norm")(jnp.mean(x, [1, 2]))
            x = nn.Dense(self.nclasses, kernel_init=initializer, name="head")(x)
        return x

ConvNeXtTiny = partial(
    ConvNeXt,
    depths=(3, 3, 9, 3),
    dims= ( 12, 24, 48, 96),
    drop_path=0.0,
    attach_head=True,
    deterministic=False,
    layer_scale_init_value = 1e-6,
    head_init_scale = 1.0
)

ConvNeXtSmall = partial(
    ConvNeXt,
    depths=(3, 3, 27, 3),
    dims=(96, 192, 384, 768),
    drop_path=0.0,
    attach_head=True,
    deterministic=False,
    layer_scale_init_value = 1e-6,
    head_init_scale = 1.0
)

ConvNeXtBase = partial(
    ConvNeXt,
    depths=(3, 3, 27, 3),
    dims=(96, 192, 384, 768),
    drop_path=0.1,
    attach_head=True,
    deterministic=False,
    layer_scale_init_value = 1e-6,
    head_init_scale = 1.0
)

