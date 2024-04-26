import tensorflow as tf
import tensorflow_datasets as tfds
from clu.preprocess_spec import PreprocessFn
from dataclasses import dataclass
from typing import Tuple, Sequence, Optional

@dataclass
class ToFloat:
    name: str = "image"

    def __call__(self, features):
        return {
            k: tf.cast(v, tf.float32) / 255.0 if k == self.name else v
            for k, v in features.items()
        }

@dataclass
class RandomCrop:
    size: Tuple[int, int]
    pad: Tuple[int, int] = (0, 0)
    name: str = "image"

    def crop(self, img):
        s = tf.shape(img)
        bs, h, w, c = s[0], s[1], s[2], s[3]
        img = tf.image.resize_with_crop_or_pad(img, h + self.pad[0], w + self.pad[1])

        return tf.image.random_crop(img, [bs, self.size[0], self.size[1], c])

    def __call__(self, features):
        return {
            k: self.crop(v) if k == self.name else v
            for k, v in features.items()
        }

@dataclass
class RandomFlipLR:
    name: str = "image"

    def __call__(self, features):
        return {
            k: tf.image.random_flip_left_right(v) if k == self.name else v
            for k, v in features.items()
        }

@dataclass
class Standardize:
    mean: Sequence[float]
    std: Sequence[float]
    name: str = "image"

    def standardize(self, img):
        m = tf.reshape(self.mean, (1, 1, 1, -1))
        s = tf.reshape(self.std, (1, 1, 1, -1))

        return (img - m) / s

    def __call__(self, features):
        return {
            k: self.standardize(v) if k == self.name else v
            for k, v, in features.items()
        }

@dataclass
class OneHot:
    num_classes: int
    name: str = "label"

    def __call__(self, features):
        return {
            k: tf.one_hot(v, self.num_classes) if k == self.name else v
            for k, v in features.items()
        }

@dataclass
class StaticShuffle:
    size: int
    seed: int = 0
    permutation: Optional[tf.Tensor] = None
    name: str = "image"

    def __post_init__(self):
        if self.permutation is None:
            self.permutation = tf.random.shuffle(tf.range(self.size),
                                                 seed=self.seed)

    def _shuffle(self, feature):
        s = feature.shape
        feature = tf.reshape(feature, [-1, tf.math.reduce_prod(s[1:])])
        feature = tf.gather(feature, self.permutation, axis=1)

        return tf.reshape(feature, [-1, *s[1:]])

    def __call__(self, features):
        return {
            k: self._shuffle(v) if k == self.name else v
            for k, v in features.items()
        }

def default_data_transforms(dataset):
    if dataset == "mnist":
        return PreprocessFn([ToFloat(),
                             Standardize((0.1307,), (0.3081,)),
                             OneHot(10)],
                            only_jax_types=True)
    elif dataset == "cifar10":
        return PreprocessFn([RandomCrop((32, 32), (2, 2)),
                             RandomFlipLR(),
                             ToFloat(),
                             Standardize((0.4914, 0.4822, 0.4465),
                                         (0.247, 0.243, 0.261)),
                             OneHot(10)], only_jax_types=True)
    else:
        return None

def select_class_subset(data, classes, name = "label"):
    _data = data if isinstance(data, dict) else {"__data": data}
    classes = tf.constant(classes, dtype=tf.int64)
    _data = {k: v.filter(lambda x: tf.reduce_any(x[name] == classes))
             for k, v in _data.items()}

    return _data if isinstance(data, dict) else _data["__data"]

def build_dataloader(data,
                     batch_size = 1,
                     shuffle = True,
                     reshuffle = True,
                     batch_transform = None,
                     shuffle_buffer_size = None):
    # convert to a tf.data.Dataset
    # we know how to convert tfds DatasetBuilder
    # everything else is treated like tensor slices
    # can always pass in a tf.data.Dataset directly
    if not isinstance(data, tf.data.Dataset):
        if isinstance(data, tfds.core.DatasetBuilder):
            data = data.as_dataset()
        else:
            data = tf.data.Dataset.from_tensor_slices(data)
    # build data loader
    if shuffle:
        buffer_size = len(data) if shuffle_buffer_size is None else shuffle_buffer_size
        data = data.shuffle(buffer_size,
                            reshuffle_each_iteration=reshuffle)
    if batch_size > 1:
        data = data.batch(batch_size)

    # possible transform the data
    if batch_transform is not None:
        data = data.map(batch_transform)

    # prefetch the next batch as the current one is being used
    return data.prefetch(2)