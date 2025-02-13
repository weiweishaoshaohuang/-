import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型參數
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) # 用Parameter指定其為參數
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

# 激勵涵數
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# Model
def net(X):
    X = X.reshape((-1, num_inputs)) # 忽略空間結構，將一張圖的二維結構展平，前面-1應為批次
    H = relu(X@W1 + b1) # 這裡@代表矩陣乘法
    return (H@W2 + b2)

# Loss
loss = nn.CrossEntropyLoss(reduction='none') # redunction none使損失為batch*類數矩陣

# Train
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.predict_ch3(net, test_iter)