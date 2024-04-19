from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jp

import models


class MLPTest(parameterized.TestCase):
    def test_mlp_model(self):
        rng = jax.random.key(0)
        model_def = models.MLP(
            input_dim=28*28,
            num_features=10000,
            num_hidden = 2,        
            num_classes=10
        )
        variables = model_def.init(rng, jp.ones((16,28,28,1), jp.float32))

if __name__ == '__main__':
    absltest.main()