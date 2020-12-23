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

## https://zhuanlan.zhihu.com/p/56684539
n = 1000
a = np.arange(n)
print(a)
b = np.lib.stride_tricks.as_strided(a, (n, n), (0, 8))
print(b)
print(b.size, b.shape, b.nbytes)

## https://www.cnblogs.com/gl1573/archive/2019/04/01/10634857.html
## 定义三维数组来测试strides
ls = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
      [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]
a = np.array(ls, dtype=int)
print(a)
## 数据类型（dtype）：描述了每个元素所占字节数。
## 维度（shape）：一个表示数组形状的元组。
## 跨度（strides）：一个表示从当前维度前进道下一维度的当前位置所需要“跨过”的字节数。
print(a.strides)
