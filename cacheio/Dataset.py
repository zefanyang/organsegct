#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 3/8/2021 4:19 PM
# @Author: yzf
import hashlib
import pickle
import collections
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Sequence, Union, Dict, Tuple, Hashable, Mapping

import torch
import numpy as np
import nibabel as nib
from scipy.ndimage import rotate
from torch.utils.data import Dataset as _TorchDataset

MAX_SEED = np.iinfo(np.uint).max + 1  # 2**32

class Dataset(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, transform: Optional[Callable] = None, progress: bool = True) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
            progress: whether to display a progress bar.
        """
        self.data = data
        self.transform = transform
        self.progress = progress

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data_ = self.data[index]
        if self.transform is not None:
            data_ = apply_transform(self.transform, data_)
        return data

class Randomizable(ABC):
    """
    An interface for handling random state locally, currently based on a class variable `R`,
    which is an instance of `np.random.RandomState`.
    """

    R: np.random.RandomState = np.random.RandomState()

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Randomizable":
        if seed is not None:
            _seed = id(seed) if not isinstance(seed, (int, np.integer)) else seed
            _seed = _seed % MAX_SEED  # _seed must be in [0, MAX_SEED - 1] for uint32
            self.R = np.random.RandomState(_seed)
            return self  # for method cascading

        if state is not None:
            if not isinstance(state, np.random.RandomState):
                raise TypeError(f"state must be None or a np.random.RandomState but is {type(state).__name__}.")
            self.R = state
            return self

        self.R = np.random.RandomState()
        return self

    @abstractmethod
    def randomize(self, data: Any) -> None:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

class Transform(ABC):
    @abstractmethod
    def __call__(self, data: Any):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method")

class RandomizableTransform(Randomizable, Transform):
    def __init__(self, prob=1.0, do_transform=False):
        self._do_transform = do_transform
        self.prob = min(max(prob, 0.0), 1.0)

    def randomize(self, data: Any) -> None:
        self._do_transform = self.R.rand() < self.prob

class Compose:
    def __init__(self, transforms: Union[Sequence[Callable], Callable]) -> None:
        if transforms is None:
            transforms = ()
        self.transforms = ensure_tuple(transforms)

    def __call__(self, input_):
        for _transform in self.transforms:
            input_ = _transform(input_)  # avoid naming conflicts
        return input_

def apply_transform(transform: Callable, data, map_items: bool = True):
    try:
        if isinstance(data, (list, tuple)) and map_items:
            return [transform(item) for item in data]
        return transform(data)
    except Exception as e:
        raise RuntimeError(f"applying transform {transform}") from e

def pickle_hashing(item, protocol=pickle.HIGHEST_PROTOCOL) -> bytes:
    # NOTE: Sort the item using the same key function so that dicts that
    #       have the same key-value pairs can produce a consistent hash value.
    cache_key = hashlib.md5(pickle.dumps(sorted_dict(item), protocol=protocol)).hexdigest()  # encode a hash value using the hash algorithm (message digest 5, MD5)
    return f"{cache_key}".encode("utf-8")

def sorted_dict(item, key=None, reverse=False):
    """Return a new sorted dictionary from the `item`."""
    if not isinstance(item, dict):
        return item
    return {k: sorted_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items(), key=key, reverse=reverse)}  # item may be a list of dicts

def ensure_tuple(vals: Any) -> Tuple[Any, ...]:
    """Return a tuple of `vals`"""
    if not isinstance(vals, collections.abc.Iterable) or isinstance(vals, str):  # not an iterable instance or is an iterable instance but is an instance of `str`
        vals = (vals, )
    return tuple(vals)

def normalize_foreground(img, label):
    nonzero_label = np.where(label>0, 1, 0)
    mean = np.sum(img * nonzero_label) / np.prod(nonzero_label.shape)
    std = np.sqrt(np.sum(np.square(img - mean) * nonzero_label) / np.prod(nonzero_label.shape))
    img = (img - mean) / std
    return img

class PersistentDataset(Dataset):
    """
    Persistent storage of pre-computed values to efficiently manage larger than memory dictionary format data,
    it can operate transforms for specific fields.  Results from the non-random transform components are computed
    when first used, and stored in the `cache_dir` for rapid retrieval on subsequent uses.

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]

    For a composite transform like

    .. code-block:: python

        [ LoadImaged(keys=['image', 'label']),
          Orientationd(keys=['image', 'label'], axcodes='RAS'),
          ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
          RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96),
                                 pos=1, neg=1, num_samples=4, image_key='image', image_threshold=0),
          ToTensord(keys=['image', 'label'])]

    Upon first use a filename based dataset will be processed by the transform for the
    [LoadImaged, Orientationd, ScaleIntensityRanged] and the resulting tensor written to
    the `cache_dir` before applying the remaining random dependant transforms
    [RandCropByPosNegLabeld, ToTensord] elements for use in the analysis.

    Subsequent uses of a dataset directly read pre-processed results from `cache_dir`
    followed by applying the random dependant parts of transform processing.

    Note:
        The input data must be a list of file paths and will hash them as cache keys.

    """

    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable],
        cache_dir: Optional[Union[Path, str]] = None,
        hash_func: Callable[..., bytes] = pickle_hashing,
        progress: bool = True,
    ) -> None:
        """
        Args:
            data: input data file paths to load and transform to generate dataset for model.
                `PersistentDataset` expects input data to be a list of serializable
                and hashes them as cache keys using `hash_func`.
            transform: transforms to execute operations on input data.
            cache_dir: If specified, this is the location for persistent storage
                of pre-computed transformed data tensors. The cache_dir is computed once, and
                persists on disk until explicitly removed.  Different runs, programs, experiments
                may share a common cache dir provided that the transforms pre-processing is consistent.
                If the cache_dir doesn't exist, will automatically create it.
            hash_func: a callable to compute hash from data items to be cached.
                defaults to `monai.data.utils.pickle_hashing`.
            progress: whether to display a progress bar.
        """
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data=data, transform=transform, progress=progress)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.hash_func = hash_func
        if self.cache_dir is not None:
            if not self.cache_dir.exists():  # is_file and is_dir
                self.cache_dir.mkdir(parents=True)
            if not self.cache_dir.is_dir():  # is_dir
                raise ValueError("cache_dir must be a directory.")

    def _pre_transform(self, item_transformed):
        """
        Process the data from original state up to the first random element.

        Args:
            item_transformed: The data to be transformed

        Returns:
            the transformed element up to the first identified
            random transform object

        """
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")
        for _transform in self.transform.transforms:  # self.transform is an instance of Compose
            # execute all the deterministic transforms
            if isinstance(_transform, RandomizableTransform) or not isinstance(_transform, Transform):
                break
            item_transformed = apply_transform(_transform, item_transformed)
        return item_transformed

    def _post_transform(self, item_transformed):
        """
        Process the data from before the first random transform to the final state ready for evaluation.

        Args:
            item_transformed: The data to be transformed (already processed up to the first random transform)

        Returns:
            the transformed element through the random transforms

        """
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of Compose.")
        start_post_randomize_run = False
        for _transform in self.transform.transforms:
            if (
                start_post_randomize_run
                or isinstance(_transform, RandomizableTransform)
                or not isinstance(_transform, Transform)
            ):
                start_post_randomize_run = True  # indicate that all transforms will be processed
                item_transformed = apply_transform(_transform, item_transformed)
        return item_transformed

    def _cachecheck(self, item_transformed: Mapping[Hashable, str]):
        """
        A function to cache the expensive input data transform operations
        so that huge data sets (larger than computer memory) can be processed
        on the fly as needed, and intermediate results written to disk for
        future use.

        Args:
            item_transformed: The current data element to be mutated into transformed representation

        Returns:
            The transformed data_element, either from cache, or explicitly computing it.

        Warning:
            The current implementation does not encode transform information as part of the
            hashing mechanism used for generating cache names.  If the transforms applied are
            changed in any way, the objects in the cache dir will be invalid.  The hash for the
            cache is ONLY dependant on the input filename paths.

        """
        hashfile = None
        if self.cache_dir is not None:
            data_item_md5 = self.hash_func(item_transformed).decode("utf-8")
            hashfile = self.cache_dir / f"{data_item_md5}.pt"  # save the pre-transformed data using .pt file extension

        if hashfile is not None and hashfile.is_file():  # cache hit
            return torch.load(hashfile)

        _item_transformed = self._pre_transform(deepcopy(item_transformed))  # keep the original hashed, because it is reused in every epoch.
        if hashfile is not None:
            # NOTE: Writing to ".temp_write_cache" and then using a nearly atomic rename operation
            #       to make the cache more robust to manual killing of parent process
            #       which may leave partially written cache files in an incomplete state
            temp_hash_file = hashfile.with_suffix(".temp_write_cache")
            torch.save(_item_transformed, temp_hash_file)
            temp_hash_file.rename(hashfile)
        return _item_transformed

    def __getitem__(self, index: int):
        pre_random_item = self._cachecheck(self.data[index])  # make sure data is pre-transformed and cached
        return self._post_transform(pre_random_item)

class RegularDataset(Dataset):
    def __init__(
        self,
        data: Sequence,
        transform: Union[Sequence[Callable], Callable]):
        if not isinstance(transform, Compose):
            transform = Compose(transform)
        super().__init__(data, transform)

    def __getitem__(self, ind: int):
        d = self.data[ind]
        for _transform in self.transform.transforms:  # self.transform is an instance of `Compose`
            d = apply_transform(_transform, d)  # process all the rest of transforms
        return d

class LoadImage(Transform):
    """
    Dictionary-based loading image
    """
    def __init__(
        self,
        keys: Sequence[str],
        dtype: np.dtype = np.float32) -> None:
        self.keys = keys
        self.dtype = dtype

    def __call__(
        self,
        data: Mapping[Hashable, str]) -> np.ndarray:
        d = dict(data)
        for key in self.keys:
            d_tmp = nib.load(d[key]).get_data()

            # flip to align orientation. See great performance gain.
            # Key design of the second group of experiments.
            if 'tcia' in d[key]:
                d_tmp = d_tmp[:, :, ::-1]

            d[key] = d_tmp.astype(self.dtype)
        return d

class Clip(Transform):
    """
    Dictionary-base intensity clip
    """
    def __init__(
        self,
        keys: Sequence[str],
        min: float,
        max: float,) -> None:
        self.keys = keys
        self.min = min
        self.max = max

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = np.clip(data[key], self.min, self.max)  # functionality
        return d

class ForeNormalize(Transform):
    """
    Dictionary-based intensity normalization
    """
    def __init__(
        self,
        keys: Sequence[str],
        mask_key: str) -> None:
        self.keys = keys
        self.mask_key = mask_key
        self.norm = normalize_foreground  # functionality

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            d[key] = self.norm(data[key], data[self.mask_key])
        return d

class RandFlip(RandomizableTransform):
    """
    Dictionary-based random flip
    """
    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.5,
        spatial_axis: Optional[Sequence[int]] = (0, 1),
        ) -> None:
        super().__init__(prob)
        self.keys = keys
        self.spatial_axis = spatial_axis

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        super().randomize(None)
        d = dict(data)
        if self._do_transform:
            flip_axis = np.random.choice(self.spatial_axis)
            for key in self.keys:
                d[key] = np.flip(data[key], axis=flip_axis)  # functionality
        return d

class RandRotate(RandomizableTransform):
    """
    Dictionary-based random rotation
    """
    def __init__(
        self,
        keys: Sequence[str],
        interp_order: Sequence[int],
        angle: Optional[float] = 15.0,
        prob: Optional[float] = 0.5,
        ) -> None:
        super().__init__(prob)
        self.keys = keys
        self.interp_order = interp_order
        self.angle = angle

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        super().randomize(None)
        the_angle = self.R.uniform(low=-self.angle, high=self.angle)
        d = dict(data)
        if self._do_transform:
            for i, key in enumerate(self.keys):
                d[key] = rotate(data[key], angle=the_angle, axes=(0, 1), reshape=False, order=self.interp_order[i]) # functionality
        return d

class ToTensor(Transform):
    """
    Dictionary-based ToTensor
    """
    def __init__(self, keys: Sequence[str]) -> None:
        self.keys = keys

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
        d = dict(data)
        for key in self.keys:
            if isinstance(data[key], torch.Tensor):
                d[key] = data[key].continguous()
            d[key] = torch.as_tensor(np.ascontiguousarray(data[key][None]))  # functionality: add a channel dimension to the input image before ToTensor
        return d





