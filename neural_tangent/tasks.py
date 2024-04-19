import torch
from torchvision import transforms, datasets
import jax.numpy as jp
import numpy as np
from jax.tree_util import tree_map
from torch.utils import data

def numpy_collate(batch):
  return tree_map(np.asarray, data.default_collate(batch))

class NumpyLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)

def shuffled_MNIST(batch_size, dataset_folder='./data'):
    """Create a MNIST dataset in which pixels are shuffled, and return the test and train dataloaders.
    Uses a new random seed every time it's called.

    Arguments:    batch_size: tha batch size
                  dataself_folder: path to the mnist dataset, or where it should go once downloaded

    """

    kwargs = {'num_workers': 0, 'pin_memory': True}

    permute_mask = torch.randperm(784)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(784)[permute_mask]),
        transforms.Lambda(lambda x: np.array(x, dtype=jp.float32))
        ])

    train_dataset = datasets.MNIST(dataset_folder, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    test_dataset = datasets.MNIST(dataset_folder, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)

    return (train_loader, test_loader)