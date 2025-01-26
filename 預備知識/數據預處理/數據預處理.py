#這邊使用pandas預處理原始數據，將原始數據轉換成與張量兼容結構的數據
#處理缺失可以使用插值法or刪除法(這邊使用插值法)
import torch
import os
import pandas as pd

#1.讀取數據集
os.makedirs(os.path.join('..','data'),exist_ok=True) # '..'代表當前文件夾的上一層(提升可移植性)，'..'與data合而為一，存在不報錯
data_file = os.path.join('..','data','house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# print(os.path.abspath(data_file))  # 打印出檔案的絕對路徑

data = pd.read_csv(data_file) # 讀取.csv文件
# print(data)


#2.處理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.select_dtypes(include='number').mean()) # 確認轉換的列是數字類型，然後用平均填上NaN
# print(inputs)
inputs = pd.get_dummies(inputs,dummy_na=True) # 將Alley區分成Alley_Pave與Alley_nan兩個不同的類別值為0 or 1 也將原本的str轉換成了數值
# print(inputs)


#3.轉換為張量值
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X,y)










