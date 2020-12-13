import matplotlib.pyplot as plt
from fc_net import *
from data_utils import get_CIFAR10_data
from solver import Solver


data = get_CIFAR10_data()
model = TwoLayerNet(reg=0.5)
solver=Solver(model,data,lr_decay=0.95,print_every=100,num_epocjs=40,batch_size=400,
              update_rule='sgd_momentum',optim_config=('learning_rate':5e-4,'momentum':0.9))
solver.train()

plt.subplot(2,1,1)
plt.title('Training Loss')
plt.plot(solver.loss_history,'o')
plt.xlabel('Iteration')