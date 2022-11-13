#

import timeit
import numpy as np

n = 512

A = np.random.rand(n,n)
B = np.random.rand(n,n)

reps = 3
num = 30000
times = timeit.repeat("C = A @ B", globals=globals(), number=num, repeat=reps)
forward_time = min(times) / num
print(f"Forward pass (best of {reps}): {forward_time} sec per loop")
