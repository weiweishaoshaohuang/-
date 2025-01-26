import matplotlib # 注意这个也要import一次
import matplotlib.pyplot as plt
import random
import torch
from d2l import torch as d2l

# "生成資料集"
# 生成1000個樣本的資料集，每個樣本從標準正態分布抽樣兩個特徵。合成一個數據集X(大小1000x2)
def synthetic_data(w, b, num_examples):
    #"生成y = Wx + b噪聲"
    X = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(X,w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1)) # 這邊如果不reshape的話y是橫的
true_w = torch.tensor([2, -3.4])
true_b = 4.2
print(f"true_w {true_w}, true_b {true_b}\n")
features, labels = synthetic_data(true_w, true_b, 1000)
# print('features:',features[0],'\nlabels:',labels[0])
# plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(),1)
# plt.title('Scatter Plot of w1 from X and y')
# plt.grid(True)
# plt.show()

# "讀取資料集"
def data_iter(batch_size, features, labels):
    num_examples = len(features) 
    indices = list(range(num_examples)) # 生成從0到num_example-1的索引
    random.shuffle(indices) # 隨機打亂索引順序
    for i in range(0, num_examples, batch_size): # for-loop的range用法第三個參數代表遞增量
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices] # yeild類似return只是他可以放在迴圈中使用到在回傳

batch_size = 10
# for X,y in data_iter(batch_size,features,labels):
#     print(X,'\n',y)
#     break

# "初始化模型參數w以及b" 注意這邊的w、b是預測值
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# "定義模型"
def linreg(X, w, b): # 線性回歸模型獲得預測y_hat
    return torch.matmul(X, w) + b

# "定義損失函數"
def square_loss(y_hat, y): # 均方損失
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# "定義優化算法"
def sgd(params, lr, batch_size): #傳入分別為參數集合、學習率、批量大小
    with torch.no_grad(): # torch.no_grad() wrap的語句將不被track到梯度計算過程中
        for param in params:
            param -= lr * param.grad / batch_size # 為了使batch_size不影響步長所以有除法
            param.grad.zero_()

# "開始訓練!"
# 設置超參數
lr = 0.03
num_epochs = 3 # 代表迭代週期，一個迭代週期用data_iter走過整個數據集
net = linreg
loss = square_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # 因為l現在是batch_size*1的向量，所以要先加總，在算梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用參數的梯度更新參數
    with torch.no_grad():
        train_l = loss(net(features,w,b), labels)
        print(w ,b)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")




