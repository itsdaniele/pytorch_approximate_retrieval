import torch
from memory_factory import MemoryManager

DB_SIZE = 1000000
KEY_SIZE = VALUE_SIZE = QUERY_SIZE = 256
NUM_HEADS = 4
NUM_NEIGHBORS = 8

mm = MemoryManager(
    num_heads=NUM_HEADS, key_size=KEY_SIZE, value_size=VALUE_SIZE, database_size=DB_SIZE
)
layer = mm.create_memory_layer()

key = torch.ones((DB_SIZE, NUM_HEADS, KEY_SIZE))
value = torch.ones((DB_SIZE, NUM_HEADS, VALUE_SIZE))

layer.update(key, value)

query = torch.ones((256, NUM_HEADS, QUERY_SIZE))

out = layer.topk_retrieval(query, 2)

print(out)

# measure time between events
import time

start = time.time()
out = layer.topk_retrieval(query, NUM_NEIGHBORS)
end = time.time()
print(end - start)

print(out.shape)

