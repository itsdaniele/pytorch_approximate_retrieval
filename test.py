import torch
from memory_factory import MemoryManager


mm = MemoryManager(num_heads=1, key_size=8, value_size=8, database_size=8)
layer = mm.create_memory_layer()

key = torch.ones((4, 1, 8))
value = torch.ones((4, 1, 8))

layer.update(key, value)


key = torch.ones((4, 1, 8)) + 1
value = torch.ones((4, 1, 8)) + 1
layer.update(key, value)
key = torch.ones((4, 1, 8)) + 2
value = torch.ones((4, 1, 8)) + 2
layer.update(key, value)

query = torch.ones((1, 1, 8)) + 2

out = layer.topk_retrieval(query, 2)

print(out)

