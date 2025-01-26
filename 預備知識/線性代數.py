import torch

#1標量
# x = torch.tensor(3.0)
# y = torch.tensor(2.0)
# print(x + y,x * y,x / y,x ** y)

#2向量
# x = torch.arange(4)
# print(x[3],len(x),x.shape)# 輸出張量第三、長度、軸長度

#3矩陣
# A = torch.arange(20).reshape(5,4)
# # print(A)
# # print(A.T)
# B = torch.tensor([[1,2,3],[2,0,4],[3,4,5]]) #symmetric matrix
# print(B == B.T)

#4張量
# X = torch.arange(24).reshape(2, 3, 4)
# print(X)

#5張量算法的基本性質
# A = torch.arange(20,dtype=torch.float32).reshape(5,4)
# B = A.clone() # 通過分配新記憶體，將A的一個副本分配給B
# print(A,A+B)
# print(A*B) # Hadamard product逐元素相乘
# a = 2
# X = torch.arange(24).reshape(2, 3, 4)
# print(a + X,a * X) # 逐元素對標量運算

#6降維
# x = torch.arange(4,dtype=torch.float32)
# # print(x,x.sum())
# A = torch.arange(20,dtype=torch.float32).reshape(5,4)
# # print(A.shape,A.sum()) # 逐元素求和
# A_sum_axis0 = A.sum(axis = 0) # 通過行元素求和(軸0)降維成一行
# A_sum_axis1 = A.sum(axis = 1) # 通過將列求和(軸1)的元素降維成一列
# print(A)
# # print(A_sum_axis0.shape,A_sum_axis0)
# # print(A_sum_axis1.shape,A_sum_axis1)
# # print(A.sum(axis=[0,1])) # 等同於A.sum()
# # print(A.mean(), A.sum()/A.numel()) # A的平均 = A的和除以A的總元素數
# # print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])
# #非降維求和
# sum_A = A.sum(axis=1,keepdim=True)
# # print(sum_A)
# # print(A / sum_A) # 此時可以通過廣播將A除以sum_A
# # print(A.cumsum(axis=0)) # 沿著每一行計算A元素的累積和，此時不降維

#7 Dot product
# x = torch.arange(4,dtype=torch.float32)
# y = torch.ones(4, dtype=torch.float32)
# print(x, y, torch.dot(x,y))

#8矩陣-向量積
# A = torch.arange(20,dtype=torch.float32).reshape(5,4)
# x = torch.arange(4,dtype=torch.float32)
# print(A,x,torch.mv(A,x)) #　注意到這邊向量默認都是列向量（直）

#9矩陣乘法
# A = torch.arange(20,dtype=torch.float32).reshape(5,4)
# B = torch.ones(4,3)
# print(torch.mm(A,B))

#10.Norm范數(其中向量的范數就是size，這個size是分量大小)
# u = torch.tensor([3.0,-4.0])
# print(torch.norm(u)) # L_2范數 像是向量歐幾里德距離 可寫成u.norm()
# print(torch.abs(u).sum()) # L_1范數 絕對值的和
# #補充說明:L_p范數為sigma(絕對值(xi)^p)^(1/p)
# print(torch.norm(torch.ones(4,9))) # 矩陣也有類似范數