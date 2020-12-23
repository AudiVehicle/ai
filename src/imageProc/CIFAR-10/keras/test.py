import platform
import random
import string
import numpy as np

print(platform.system())
print(random.choice('abcdefghijklmnopqrstuvwxyz!@#$%^&*()'))
## 输出随机字符串
print(''.join(random.sample(string.ascii_letters + string.digits, 8)))

for b in range(1, 5):
    print(b)

import time

millis = int(round(time.time() * 1000))
print(millis)

a = (1, 2, 3)
print(a)

n = 1000
a = np.arange(n)
print(a)
b = np.lib.stride_tricks.as_strided(a, (n, n), (0, 8))
print(b)
print(b.size, b.shape, b.nbytes)
