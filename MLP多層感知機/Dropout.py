import torch
from torch import nn
from d2l import torch as d2l

# 從零開始實現
def dropout_layer(X, dropout):
    assert 1 >= dropout >= 0
    if dropout == 1: # 本情況X全部丟棄
        return torch.zeros_like(X)
    if dropout == 1: # 本情況X全部保留
        return X
    mask = (torch.rand(X.shape) > dropout).float() # rand生成0~1的均勻隨機分布，大於dp 1否則0
    return mask * X / (1.0 - dropout)

# test
# X = torch.arange(16, dtype=float).reshape((2, 8))
# print(X)
# print(dropout_layer(X, 0.))
# print(dropout_layer(X, 0.5))
# print(dropout_layer(X, 1.))

# 定義模型參數
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

# 定義模型
dropout1, dropout2 = 0.2, 0.5 # 常規技巧靠近輸入層概率較低

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden1, num_hidden2, 
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.lin3 = nn.Linear(num_hidden2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在訓練過程才用dropout 推理不用
        if self.training == True:
            # 在第一個全聯階層之後添加dropout層
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二個全聯階層之後添加dropout層
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out
    
# net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 訓練和測試
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# trainer = torch.optim.SGD(net.parameters(), lr=lr)
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

# 簡潔實現
net = nn.Sequential(nn.Flatten(), 
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def _init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(_init_weights)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

