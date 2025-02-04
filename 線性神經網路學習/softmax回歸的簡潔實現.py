import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 1.初始化模型參數
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10)) # 因為Sequential不會隱式將圖片展平，所以前面加Flatten將28x28圖片變成784輸入10輸出

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)


# 2.重新審視Softmax的實現 因為指數可能造成數overflow所以使用log(y_hat)來評估
loss = nn.CrossEntropyLoss(reduction='none')


# 3.優化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.01)


# 4.訓練

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# d2l.plt.show()
d2l.predict_ch3(net, test_iter)