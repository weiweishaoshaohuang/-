import torch
from torch import nn
from d2l import torch as d2l

# 生成數據y = 0.05 + 0.01 * x + eps
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 初始化模型參數
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定義L2 penalty
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

# 定義訓練代碼實現
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X,y in train_iter:
            # 增加了L2 penalty
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print("w的L2范數是 : ", torch.norm(w).item()) # norm沒有平方

# 忽略regularization直接train
# train(lambd=0) # 從訓練誤差減少 測試誤差不變知道發生overfitting

# Weight decay
# train(lambd=30) # 測試誤差也減少了 沒有overfitting了

# 簡潔實現
def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for params in net.parameters():
        params.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置參數沒有衰減
    trainer = torch.optim.SGD([
        {"params" : net[0].weight,'weight_decay' : wd},
        {"params" : net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())

# train_concise(0) # 不regulize
train_concise(3) # regulize
