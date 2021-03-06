import matplotlib.pyplot as plt
from cnn.cnn_net import *
from common.data_utils import get_CIFAR10_data
from common.solver import Solver
import time
import string
import random

## https://github.com/yunjey/cs231n
## https://github.com/martinkersner/cs231n

## 介绍了几种学习优化算法
##http://www.xyu.ink/1817.html

data = get_CIFAR10_data()
model = ThreeLayerConvNet(reg=0.5)
solver = Solver(model, data, lr_decay=0.95, print_every=100, num_epochs=20, batch_size=200,
                update_rule='adam', optim_config={'learning_rate': 5e-4, 'momentum': 0.9})
solver.train()

plt.subplot(2, 1, 1)
plt.title('Training Loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.plot([0.5] * len(solver.val_acc_history), 'k--')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.gcf().set_size_inches(15, 12)
# plt.show()
plt.savefig(str(round(time.time() * 1000)) + '_loss&train.png')
plt.close()
