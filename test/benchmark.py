import torch
from pytorch_approximate_retrieval.memory_factory import MemoryManager

DB_SIZE = 32000
KEY_SIZE = VALUE_SIZE = QUERY_SIZE = 32
NUM_HEADS = 4
NUM_NEIGHBORS = 8

mm = MemoryManager(
    num_heads=NUM_HEADS, key_size=KEY_SIZE, value_size=VALUE_SIZE, database_size=DB_SIZE
)
layer = mm.create_memory_layer()

key = torch.ones((DB_SIZE, NUM_HEADS, KEY_SIZE)).cuda()
value = torch.ones((DB_SIZE, NUM_HEADS, VALUE_SIZE)).cuda()

layer.update(key, value)

query = torch.ones((256, NUM_HEADS, QUERY_SIZE)).cuda()

out = layer.topk_retrieval(query, 2)

# measure time between events
import time

start = time.time()
out = layer.topk_retrieval(query, NUM_NEIGHBORS)
end = time.time()
print(end - start)

to_reset = [True] * NUM_HEADS
layer.reset(torch.tensor(to_reset).cuda())
