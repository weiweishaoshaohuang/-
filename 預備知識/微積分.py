import torch
#1.自動微分
x = torch.arange(4.0,requires_grad=True) # 或多寫一句x.requires_grad_(True)
# print(x) # 分別是x1=0, x2=1, x3=2, x4=3
# y = 2 * torch.dot(x,x) # y = 2x^2
# y.backward() # y相對於每個輸入邊量求梯度，這邊只有x再計算每個xi的梯度
# # print(x.grad) # x.grad = 4x
# #因為Pytorch會自動累積梯度，所以需要先清除之前的值
# x.grad.zero_()
# y = x.sum()
# y.backward()
# print(x.grad)

# x.grad.zero_()
# y = x * x
# y.sum().backward() # 等價於y.backward(torch.ones(Len(x)))
# # print(x.grad)

# x.grad.zero_()
# y = x * x
# u = y.detach()
# z = u * x
# z.sum().backward()
# print(x.grad == u) # 此時微分結果不是以x * x * x的結果去算而是以ux去算

# x.grad.zero_()
# y.sum().backward()
# #print(x.grad == 2*x)

#2.Python控制流的梯度計算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(),requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a) # 因為f(a)的結果就是某個k使得f(a)=ka