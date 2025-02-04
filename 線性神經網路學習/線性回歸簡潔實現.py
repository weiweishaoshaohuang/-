# 在前面我們自己設計了數據迭代器、損失涵式、優化器、神經網路層，但其實這些都有現成組件

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn #nn是神經網路的縮寫

# 1.生成資料集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 2.讀取資料集
# 構造一個PyTorch數據迭代器
def load_array(data_arrays, batch_size, is_train = True): # 將feature&label合起來作為參數data_array傳入
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle= is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)
# print(next(iter(data_iter))) # 與前者不同這邊以Python內建iter代替for next獲取迭代起第一項

# 3.定義模型
net = nn.Sequential(nn.Linear(2, 1)) # 輸入features形狀為2 輸出特徵形狀為1

# 4.初始化模型參數
net[0].weight.data.normal_(0, 0.01) # 先通過net[0]選擇網路第一個圖層，再用weight.data訪問參數
net[0].bias.data.fill_(0) #先通過net[0]選擇網路第一個圖層，再用bias.data訪問參數

# 5.定義損失涵式
loss = nn.MSELoss() # Mean Square Error

# 6.定義優化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03) # net.parameters可以得到需要優化的參數們，SGD只需要設置lr值

# 7.訓練
num_epochs = 3
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad() # 因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前
        l.backward() # 如果没有进行tensor.backward()的话，梯度值将会是None，因此loss.backward()要写在optimizer.step()之前。
        trainer.step() #進行下一步優化，前提是要知道梯度，所以必須在backward之後
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print(f"w的估計誤差: {true_w - w.reshape(true_w.shape)}")
b = net[0].bias.data
print('b的估計誤差: ',b - true_b) 