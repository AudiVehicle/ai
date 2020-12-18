import platform
import random
import string

print(platform.system())
print(random.choice('abcdefghijklmnopqrstuvwxyz!@#$%^&*()'))
## 输出随机字符串
print(''.join(random.sample(string.ascii_letters + string.digits, 8)))

for b in range(1, 5):
    print(b)
