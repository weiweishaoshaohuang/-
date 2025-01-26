#學習如何利用torch中計算張量的便利
import torch

#1.基本
# x = torch.arange(12) # 用arrang創造行向量 包含從0開始的12個整數
#print(x) # 輸出張量所有元素
# print(x.shape) # 張量沿每個軸的形狀
# print(x.numel()) # 得到張量元素的總數=形狀所有元素的乘積(長*寬*高...)
# X = x.reshape((3,4)) # 僅改變x形狀3x4矩陣，不改變元素，張量大小
# X = x.reshape((-1,4)) # 用-1讓他自動判斷寬是多少
# X = x.reshape((3,-1)) # 用-1讓他自動判斷高是多少
# print(X)
# print(torch.zeros((2,3,4))) # 創造形狀(2,3,4)的張量
# print(torch.ones(2,3,4)) # 創造形狀(2,3,4)的張量
# print(torch.randn((3,4))) # 創建形狀(3,4)張量，元素值是均0標準差1的正態分布隨機採樣
# print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])) # python基礎賦值列表


#2.張量運算
x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
# print(x + y,x - y,x * y,x ** y) # **是求冪運算
# print(torch.exp(x)) # 按元素求自然指數冪e^x
X = torch.arange(12,dtype = torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# print(torch.cat((X,Y),dim=0),torch.cat((X,Y),dim=1)) # 張量連接(concatenate)
# print(X==Y) # 通過邏輯符構建新張量(裡面只有True False)
# print(X.sum()) # 通過對張量所有元素求和得到單元素張量


#3.廣播機制(broadcasting mechanism)
a = torch.arange(3).reshape((3,1))#3x1向量
b = torch.arange(2).reshape((1,2))#1x2向量
# print(a+b) # 結果為先進行廣播再計算(將向量3x1 -> 3x2 | 1x2 -> 3x2)


#4.索引&切片
# print(X[-1],X[1:3])# 與python陣列中0代表第一項，-1代表最後一項。[1:3]是指1到2不算右邊界
# X[1,2] = 9# 透過索引改變陣列值
# # print(X)
# X[0:2, :] = 12# 透過索引改變第0行第1行，列那邊代表所有元素
# # print(X)


#5.節省內存
# before = id(Y) # id是python內建函數提供內存物件的確切地址
# Y = Y + X
# print(id(Y) == before) # 因為Y在運算時會將新值儲存在新分配的內存，釋放舊的Y空間內存，這對ML並不好

# Z = torch.zeros_like(Y) # 構造一個形狀跟Y一樣的全0陣列
# before = id(Z)
# Z[:] = X + Y # 方法:使用切片表示法，將操作結果直接給早就分配好的陣列
# print(id(Z) == before)

# before = id(X)
# X += Y # 另外對X，也可替換成'X[:] = X + Y'
# print(id(X) == before)


#6.轉換成其他python物件
A = X.numpy()# X: tensor -> numpy
B = torch.tensor(A)# A: numpy -> tensor
# print(type(A),type(B))
a = torch.tensor([3.5])
# print(a,a.item(),float(a),int(a)) # 將 大小為1的張量轉換成標量可以用item或直接強轉



