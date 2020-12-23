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

## https://blog.csdn.net/shwan_ma/article/details/78244044

# 数独是个 9x9 的二维数组
# 包含 9 个 3x3 的九宫格
sudoku = np.array([
    [2, 8, 7, 1, 6, 5, 9, 4, 3],
    [9, 5, 4, 7, 3, 2, 1, 6, 8],
    [6, 1, 3, 8, 4, 9, 7, 5, 2],
    [8, 7, 9, 6, 5, 1, 2, 3, 4],
    [4, 2, 1, 3, 9, 8, 6, 7, 5],
    [3, 6, 5, 4, 2, 7, 8, 9, 1],
    [1, 9, 8, 5, 7, 3, 4, 2, 6],
    [5, 4, 2, 9, 1, 6, 3, 8, 7],
    [7, 3, 6, 2, 8, 4, 5, 1, 9]
])

# 要将其变成 3x3x3x3 的四维数组
# 但不能直接 reshape，因为这样会把一行变成一个九宫格
shape = (3, 3, 3, 3)

# 大行之间隔 27 个元素，大列之间隔 3 个元素
# 小行之间隔 9 个元素，小列之间隔 1 个元素
strides = sudoku.itemsize * np.array([27, 3, 9, 1])

squares = np.lib.stride_tricks.as_strided(sudoku, shape=shape, strides=strides)
print(squares)
print(squares.shape)

## 就是按shape的形状对一个矩阵进行切块
'''
[[[[2 8 7]    [9 5 4]    [6 1 3]]
  [[1 6 5]    [7 3 2]    [8 4 9]]
  [[9 4 3]    [1 6 8]    [7 5 2]]]

 [[[8 7 9]    [4 2 1]    [3 6 5]]
  [[6 5 1]    [3 9 8]    [4 2 7]]
  [[2 3 4]    [6 7 5]    [8 9 1]]]

 [[[1 9 8]    [5 4 2]    [7 3 6]]
  [[5 7 3]    [9 1 6]    [2 8 4]]
  [[4 2 6]    [3 8 7]    [5 1 9]]]]
'''
