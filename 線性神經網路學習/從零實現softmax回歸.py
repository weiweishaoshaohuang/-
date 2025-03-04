import torch
from IPython import display
from d2l import torch as d2l


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)



# 1.初始化模型參數
num_inputs = 784 # 將28x28的圖像矩陣展平成784的向量
num_outputs = 10 #共有10個圖像類別

W = torch.normal(0, 0.01, size=(num_inputs,num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)


# 2.定義softmax操作
# X = torch.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# print(X)
# print(X.sum(0, keepdim=True),'\n' ,X.sum(1, keepdim=True)) # 複習沿張量維度求和

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 這裡應用了broadcasting

# X = torch.normal(0, 1, (2,5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(1)) # 測試


# 3.定義模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 4.定義損失涵式
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1],y]) # y_hat[[0, 1]是指y_hat行0、行1 # 測試

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
# print(cross_entropy(y_hat, y)) # 測試


# 5.定義分類精度計算
def accuracy(y_hat, y):
    # "計算預測成功的數量"
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: # y_hat是否為二維張量
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y # 產生一列值為0或1的張量
    return float(cmp.type(y.dtype).sum())

# print(accuracy(y_hat,y) / len(y)) # 測試:計算成功率

class Accumulator:
    # "在n個變量上累加"
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def evaluate_accuracy(net, data_iter):
    # "計算在指定數據集模型的精度"
    if isinstance(net, torch.nn.Module):
        net.eval() # 將模型設置成評估模式
    metric = Accumulator(2) # 正確預測數 預測總數
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # 正確預測數 預測總數
    return metric[0] / metric[1]

# print(evaluate_accuracy(net, test_iter))
# 6.訓練
def train_epoch_ch3(net, train_iter, loss, updater): # updater是更新模型参数的常用函数，它接受批量大小作为参数。 它可以是d2l.sgd函数，也可以是框架的内置优化函数。
    # 訓練模型的一個迭代週期
    if isinstance(net, torch.nn.Module):    # net是否為後者或後者子累的實例
        net.train() # 將模型設置成訓練模式
    # 訓練損失總和，訓練準確度總和，樣本數
    metric = Accumulator(3)
    for X,y in train_iter:
        # 計算梯度並更新參數
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch內置的優化器和損失涵數
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用訂製的優化器和損失涵數
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回訓練損失和訓練精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)  

def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)