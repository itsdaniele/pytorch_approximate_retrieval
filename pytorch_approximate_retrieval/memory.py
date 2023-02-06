import torch
import torch.nn.functional as F

import abc
from typing import Tuple, Sequence

import numpy as np


Dtype = torch.dtype
Shape = Sequence[int]
Tensor = torch.Tensor


class MemoryLayer(torch.nn.Module, metaclass=abc.ABCMeta):
    """Internal interface for memory layers without batch dim."""

    @abc.abstractmethod
    def update(self, key: Tensor, value: Tensor) -> int:
        """Adds key/value pairs to memory.

        Args:
            key: of shape (num_kv, num_datasets, k_features)
            value: of shape (num_kv, num_datasets, v_features)

        Returns:
            None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def topk_retrieval(
        self, query: Tensor, num_neighbors: int
    ) -> Tuple[Tensor, Tensor]:
        """Retrieves the nearest neighbors for each query.
        Args:
            query: of shape (num_queries, num_datasets, k_features)
            num_neighbors: int indicating the number of neighbors to retrieve

        Returns:
            Tuple of selected keys and selected values of shapes
            (num_queries, num_datasets, num_neighbors, k_features), and
            (num_queries, num_datasets, num_neighbors, v_features)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, datasets: Tensor) -> int:
        """Reset some or all of the datasets in the memory.
        Args:
            datasets: A vector of shape (num_datasets) of type bool. Each position
                indicates whether the dataset with the same index should be reset.
        Returns:
            None
        """
        raise NotImplementedError()

    def forward(self, query, num_neighbors):
        return self.topk_retrieval(query, num_neighbors)


def _target_dimensions(shape: Shape, source_dimensions: Sequence[int]) -> Sequence[int]:
    target_dimensions = range(-2, -2 - len(source_dimensions), -1)
    assert len(source_dimensions) == len(target_dimensions)
    return sorted(d % len(shape) for d in target_dimensions)


def _rearrange_dimensions_shapes(
    shape: Shape, split_dimensions: Sequence[int]
) -> Tuple[Shape, Shape]:
    split_shape = tuple(shape[d] for d in split_dimensions)
    remaining_shape = tuple(
        shape[d] for d in range(len(shape)) if d not in split_dimensions
    )
    batch_shape = remaining_shape[:-1]
    return split_shape, batch_shape


def _rearrange_dimensions(x: Tensor, split_dimensions: Sequence[int]) -> Tensor:
    """Rearrange array so that we can split by a single dimension.
    Turns an array of shape [d1, ..., dn, features] and a list of dimensions to
    split by into [prod(remaining_dimensions), prod(split_dimensions),
    features]
    Args:
      x: array of shape [d1, ..., dn, features]
      split_dimensions: list of dimensions that should end up in dimension -2.
    Returns:
      Rearranged array as described above.
    """
    split_dimensions = [d % len(x.shape) for d in split_dimensions]
    split_dimensions = sorted(split_dimensions)
    split_shape, batch_shape = _rearrange_dimensions_shapes(x.shape, split_dimensions)

    target_dimensions = _target_dimensions(x.shape, split_dimensions)
    x = torch.moveaxis(x, split_dimensions, target_dimensions)
    assert len(x.shape) > len(split_dimensions)
    assert all(isinstance(d, int) and d >= 0 for d in batch_shape)
    assert all(isinstance(d, int) and d >= 0 for d in split_shape)
    new_shape = [
        # The use of numpy is okay here, since shapes are concrete at jit time.
        np.prod(batch_shape),
        np.prod(split_shape),
        x.shape[-1],  # features dimension
    ]
    res = x.reshape(new_shape)
    assert res.ndim == 3
    return res


def _restore_dimensions(
    x: Tensor, original_shape: Shape, split_dimensions: Sequence[int]
) -> Tensor:
    """Restores arrays encoded with _rearrange_dimensions.
    Args:
      x: Array of shape [prod(batch_shape), prod(split_shape), feature...]
      original_shape: Shape of the array to restore to.
      split_dimensions: Dimensions that were multiplied into dimension 2.
    Returns:
      Array of the original shape and axis order for all dimensions in batch_shape
      and split_shape. Feature dimensions may have changed (can include additional
      dimensions for neighbors, for example).
    """
    split_dimensions = [d % len(original_shape) for d in split_dimensions]
    split_dimensions = sorted(split_dimensions)
    split_shape, batch_shape = _rearrange_dimensions_shapes(
        original_shape, split_dimensions
    )

    features_shape = x.shape[2:]
    x = x.reshape((*batch_shape, *split_shape, *features_shape))

    # rearrange
    target_dimensions = _target_dimensions(original_shape, split_dimensions)
    x = torch.moveaxis(x, target_dimensions, split_dimensions)
    return x


class Memory(torch.nn.Module):
    def __init__(
        self, mem_layer: MemoryLayer = None, split_dimensions: Tuple[int, ...] = (-2,)
    ):

        super().__init__()
        self.wrapped = mem_layer
        # `split_dimensions` indicates the dimensions of the query and update tensors
        # that will go to separate databases. By default, we use a separate database
        # for each head.
        # Note that some implementations of the memory share memory across all hosts
        # and devices (memory_on_borg, unless configured otherwise) or just across
        # devices of each host (memory_on_host).
        # Default is (-2,) to split by head only; use (0, -2) to also slit by batch
        # dimensions.
        self.split_dimensions = split_dimensions

    def update(self, key: Tensor, value: Tensor):
        """Adds key/value pairs to memory.

        Args:
            key: typically of shape (kv_len, num_heads, k_features). This
                tensor is split up into datasets according to `split_dimensions`.
            value: typically of shape (kv_len, num_heads, v_features). This
                tensor is split up into datasets according to `split_dimensions`.
        Returns:
            None
        """
        if key.ndim != 3 or value.ndim != 3:
            raise ValueError(
                "Expected non-batched inputs; got shapes: %s and %s."
                % (key.shape, value.shape)
            )

        key = _rearrange_dimensions(key, self.split_dimensions)
        value = _rearrange_dimensions(value, self.split_dimensions)
        return self.wrapped.update(key, value)

    def topk_retrieval(self, query: Tensor, num_neighbors: int):
        """Retrieves the nearest neighbors for each query.
        Args:
            query: typically of shape (batch, q_len, num_heads, k_features). This
                tensor is split up into datasets according to `split_dimensions`.
            num_neighbors: number of neighbors to retrieve

        Returns:
            Tuple of tensors with the retrieved keys and value of the same shape as
            query, but with an extra dimension of length num_neighbors - typically:
            (q_len, num_heads, num_neighbors, k_features)
        """

        original_shape = query.shape
        query = _rearrange_dimensions(query, self.split_dimensions)
        key, value = self.wrapped.topk_retrieval(query, num_neighbors)
        key = _restore_dimensions(key, original_shape, self.split_dimensions)
        # Note that `original_shape` here may have the wrong feature dimension (if
        # k_features != v_features. But `_restore_dimensions` does not depend on
        # that dimension and the tests cover this case.
        value = _restore_dimensions(value, original_shape, self.split_dimensions)
        assert key.ndim == len(original_shape) + 1
        return key, value

    def reset(self, datasets: Tensor) -> int:
        """Resets the memory.

        Args:
            datasets: of shape (num_datasets,), typically the same as (num_heads,).

        Returns:
            None
        """
        return self.wrapped.reset(datasets)


def _chunking_sparsify(
    query: Tensor, key: Tensor, num_buckets: int, bucket_size: int
) -> Tuple[Tensor, Tensor, Tensor]:
    """Approximate top k operation for a single head."""
    # q = q_length, f = qk features, d = database_size
    scores = torch.einsum("qf,df->qd", query, key)
    mask = (key.sum(-1) == 0).type(torch.float32) * -1e6
    scores += mask

    num_queries, _ = scores.shape
    reshaped_scores = torch.reshape(scores, (num_queries, bucket_size, num_buckets))

    sparse_scores = F.softmax(reshaped_scores * 1e6, dim=1)

    # topk_scores and topk_indices will only be computed if we depend on their
    # results.
    topk_scores = torch.max(reshaped_scores, dim=1)[0]
    local_indices = torch.argmax(reshaped_scores, dim=1)
    topk_indices = local_indices * num_buckets + torch.arange(
        num_buckets
    ).cuda().reshape((1, num_buckets))
    return sparse_scores, topk_scores, topk_indices


def _retrieve_topk_gatherless(
    query: Tensor, key: Tensor, value: Tensor, num_neighbors: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Retrieves for a single head - used to simplify array accesses."""
    num_kv, query_features = query.shape
    database_size, key_features = key.shape
    _, value_features = value.shape
    assert query_features == key_features
    num_buckets = num_neighbors
    if num_buckets > database_size:
        raise ValueError(
            "More buckets than items in database. %s > %s"
            % (num_buckets, database_size)
        )
    if database_size % num_buckets:
        raise ValueError(
            "Buckets must divide database: %s %% %s." % (database_size, num_buckets)
        )
    bucket_size = database_size // num_buckets

    sparse_scores, topk_scores, topk_indices = _chunking_sparsify(
        query, key, num_buckets, bucket_size
    )
    key = key.reshape(bucket_size, num_buckets, key_features)
    value = value.reshape(bucket_size, num_buckets, value_features)
    selected_keys = torch.einsum("qbn,bnd->qnd", sparse_scores, key)
    selected_values = torch.einsum("qbn,bnd->qnd", sparse_scores, value)

    assert selected_keys.shape == (num_kv, num_neighbors, key_features)
    assert selected_values.shape == (num_kv, num_neighbors, value_features)
    return selected_keys, selected_values, topk_scores, topk_indices


class MemoryOnGpu(MemoryLayer):
    def __init__(
        self,
        database_size: int,
        dtype: Dtype = torch.float32,
        key_features: int = 64,
        value_features: int = 64,
        report_scores_and_indices: bool = False,
        num_datasets: int = None,
    ):
        super().__init__()

        self.num_datasets = num_datasets
        self.database_size = database_size
        self.dtype = dtype
        self.key_features = key_features
        self.value_features = value_features
        self.report_scores_and_indices = report_scores_and_indices

        self.setup()

    def setup(self):
        self.db_index = torch.zeros((self.num_datasets,), dtype=torch.int32).cuda()
        self.key_db = torch.zeros(
            (self.num_datasets, self.database_size, self.key_features), dtype=self.dtype
        ).cuda()
        self.value_db = torch.zeros(
            (self.num_datasets, self.database_size, self.value_features),
            dtype=self.dtype,
        ).cuda()

        self.retrieved_indices = torch.zeros((0, 0, 0), dtype=torch.int32)
        self.retrieved_indices_scores = torch.zeros((0, 0, 0), dtype=torch.float32)

    def _update_kv_database(self, database, new_values, start_index):
        num_datasets, database_size, _ = database.shape
        assert (
            database_size == self.database_size
        ), f"{database_size} vs {self.database_size}"
        assert num_datasets == self.num_datasets
        assert new_values.ndim == 3
        assert start_index.shape == (self.num_datasets,)

        for i in range(database.shape[0]):  # TODO parallelize
            database[
                i, start_index[i] : start_index[i] + new_values[i].shape[0]
            ] = new_values[i]
        return database

    def update(self, key: Tensor, value: Tensor) -> int:
        """Add keys and values to the memory; overwrite oldest if memory is full."""

        assert len(key.shape) == len(value.shape)
        assert key.shape[:-1] == value.shape[:-1]
        num_kv, num_datasets, key_features = key.shape
        assert num_datasets == self.num_datasets
        assert key_features == self.key_features
        assert value.shape[-1] == self.value_features
        assert (
            self.database_size % num_kv == 0
        ), "Database size must be integer multiple of num_kv."
        key = torch.moveaxis(key, source=1, destination=0)  # split by dataset
        value = torch.moveaxis(value, source=1, destination=0)  # split by dataset

        # start_index can be larger than DB - we use that to detect which entries
        # are not written to yet
        start_index = self.db_index % self.database_size
        self.key_db = self._update_kv_database(self.key_db, key, start_index)
        self.value_db = self._update_kv_database(self.value_db, value, start_index)
        self.db_index = self.db_index + num_kv

    def topk_retrieval(
        self, query: Tensor, num_neighbors: int
    ) -> Tuple[Tensor, Tensor]:
        """Nearest neighbors by full multiplication and approximate top k on GPU."""

        unused_num_kv, num_datasets, query_features = query.shape
        assert num_datasets == self.num_datasets
        assert query_features == self.key_features
        query = torch.moveaxis(query, source=1, destination=0)

        selected_keys, selected_values, topk_scores, topk_indices = [], [], [], []

        for i in range(num_datasets):

            (
                selected_keys_,
                selected_values_,
                topk_scores_,
                topk_indices_,
            ) = _retrieve_topk_gatherless(
                query[i], self.key_db[i], self.value_db[i], num_neighbors
            )

            selected_keys.append(selected_keys_)
            selected_values.append(selected_values_)
            topk_scores.append(topk_scores_)
            topk_indices.append(topk_indices_)

        selected_keys = torch.stack(selected_keys)
        selected_values = torch.stack(selected_values)
        topk_scores = torch.stack(topk_scores)
        topk_indices = torch.stack(topk_indices)

        if self.report_scores_and_indices:
            self.retrieved_indices.value = topk_indices
            self.retrieved_indices_scores.value = topk_scores

        assert selected_keys.ndim == selected_values.ndim == 4
        selected_keys = torch.moveaxis(selected_keys, source=0, destination=1)
        selected_values = torch.moveaxis(selected_values, source=0, destination=1)
        return selected_keys, selected_values

    def reset(self, datasets: Tensor) -> int:
        """Resets specified datasets."""
        datasets = datasets.detach().requires_grad_(False)
        assert datasets.shape == (self.num_datasets,)
        assert datasets.dtype == torch.bool

        def _reset_single_dataset(input_tuple):
            """Resets a single head; reset is a single bool."""
            database, reset = input_tuple
            assert reset.shape == tuple(), reset.shape
            assert reset.dtype == torch.bool
            return database * (1 - reset)

        self.db_index = self.db_index * (1 - datasets.int())

        for i in range(self.num_datasets):  # TODO broadcast?
            self.key_db[i, :, :] = self.key_db[i, :, :] * (1 - datasets[i].int())
            self.value_db[i, :, :] = self.value_db[i, :, :] * (1 - datasets[i].int())
