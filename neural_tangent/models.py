from flax import linen as nn


class MLP(nn.Module):
    input_dim: int
    num_features: int
    num_classes: int
    num_hidden: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape(-1, self.input_dim)
        x = nn.Dense(self.num_features)(x)
        x = nn.relu(x)
        for i in range(self.num_hidden):
            x = nn.Dense(self.num_features)(x)
            x = nn.relu(x)
        return nn.Dense(self.num_classes)(x)