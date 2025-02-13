import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
# d2l.plt.show() # relu圖
y.backward(torch.ones_like(x), retain_graph=True) # retrain_graph使得計算圖中間過程不會直接釋放掉
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu(x)', figsize=(5, 2.5))
# d2l.plt.show() # relu的梯度圖

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# d2l.plt.show() # sigmoid圖
x.grad.data.zero_() # 清除梯度
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid(x)', figsize=(5, 2.5))
# d2l.plt.show() # sigmoid梯度圖

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)' , figsize=(5, 2.5))
# d2l.plt.show() # tanh圖
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.show() #tanh梯度圖