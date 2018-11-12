import numpy
import random
from collections import deque

class Memory(object):
    def __init__(self, memory_size, batch_size, _):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def add(self, data, priority):
        self.memory.append(data)

    def select(self, _):
        batch = random.sample(self.memory, k=self.batch_size)
        return batch, None, None

    def priority_update(self, _, __):
        pass

    def reset_alpha(self, _):
        pass

    def __len__(self):
        return len(self.memory)
