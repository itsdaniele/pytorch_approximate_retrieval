from typing import Any, Optional, Tuple
import torch
from pytorch_approximate_retrieval.memory import MemoryOnGpu, Memory

Shape = Tuple[int]
Dtype = Any
Array = Any
MemoryResource = Any


class MemoryManager:
    """Manages any external resources that may be required by external memory.
  """

    def __init__(
        self,
        num_heads: int,
        key_size: int,
        value_size: int,
        database_size: Optional[int] = None,
        dtype: Dtype = torch.float32,
    ):
        """Create a MemoryManager object.

        A MemoryManager configures external memory.

        Args:
            mode:       e.g. ("train", or "test")
            num_heads:  The number of transformer heads.
            key_size:   The length of the key vectors.
            value_size: The length of the value vectors.
            database_size:  The total number of tokens in the database.
            dtype:      The datatype used for keys and values.
        """

        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size
        self.database_size = database_size
        self.dtype = dtype

    def create_memory_layer(self):
        """Create a module that implements external memory."""

        num_datasets = self.num_heads

        assert self.database_size is not None
        mem_layer = MemoryOnGpu(
            num_datasets=num_datasets,
            key_features=self.key_size,
            value_features=self.value_size,
            database_size=self.database_size,
            dtype=self.dtype,
        )
        # Handle queries of shape [seq_len, num_heads, kv_features]
        return Memory(mem_layer, split_dimensions=(-2,))

