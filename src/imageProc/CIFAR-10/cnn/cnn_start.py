import matplotlib.pyplot as plt
from cnn.cnn_net import *
from common.data_utils import get_CIFAR10_data
from common.solver import Solver
import string
import random

## https://github.com/yunjey/cs231n
## https://github.com/martinkersner/cs231n

data = get_CIFAR10_data()
model = ThreeLayerConvNet(reg=0.5)
solver = Solver(model, data, lr_decay=0.95, print_every=10, num_epochs=5, batch_size=2,
                update_rule='sgd_momentum', optim_config={'learning_rate': 5e-4, 'momentum': 0.9})
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
plt.savefig(''.join(random.sample(string.ascii_letters + string.digits, 4)) + '_loss&train.png')
plt.close()
