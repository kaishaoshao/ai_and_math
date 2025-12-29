# 数据预处理
# 我们首先创建一个人工数据集，并存储再CSV文件../data/house_tiny.csv中，
# 以其他格式存储的数据可以通过类似的方式进行处理

# 读取数据集
import os

os.makedirs(os.path.join('./', 'data'), exist_ok=True)
data_file = os.path.join('./', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write("NumRooms,Alley,Price\n")  # 列名
    f.write("NA,Pave,127500\n")  # 每行表示一个数据样本
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

# 要从创建的CSV文件中加载原始数据集，导入pandas包
# 并调用read_csv函数
import pandas as pd
data = pd.read_csv(data_file)
print(data)

# 处理缺失值
# "NaN"项代表缺失值，为了处理缺失的数据，经典的方法是插值法和删除法
# 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值
# 通过位置索引iloc,我们将data分成inputs和outputs，其中前者为data的
# 第一和第二个列，后者为data的最后一列，对inputs中缺少的数值，我们
# 使用同一列的均值替换"NaN"
# inputs.mean()是取均值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print("\n",inputs)
# 对于inputs中类别值或离散值，我们将"NaN"视为一个类别，由于"巷子类型"
# (Alley)列只接受两种类型的类别值"Pave"和"NaN"，pandas可以自动将此
# 列转换为两列"Alley_Pave"和"Alley_nan",有此类型，"Alley_Pave"的
# 值是1，而"Alley_nan"的值是0，缺少巷子类型的行会将"Alley_Pave"的 
# 值设置为0 ，"Alley_nan"都设置为1
inputs = pd.get_dummies(inputs, dummy_na=True)
print("\n",inputs)

# 转换为张量格式
import torch
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print("\n", X, y)

# 练习
# 创建更多数据集
os.makedirs(os.path.join('./', 'data'), exist_ok=True)
data_file2 = os.path.join('./', 'data', 'house.csv')
with open(data_file2, 'w') as f:
    f.write("NumRooms,Bedroom,Bathroom,Alley,Price\n")
    f.write("NA,2,2,Pave,127500\n")
    f.write("2,3,1,NA,106000\n")
    f.write("4,3,2,NA,178100\n")
    f.write("1,3,2,NA,140000\n")
    f.write("NA,2,2,Pave,178000\n")
    f.write("3,3,1,NA,124000\n")
    f.write("5,2,1,Grvl,129000\n")
    f.write("NA,2,2,Grvl,118000\n")
    f.write("3,2,2,NA,123000\n")
    f.write("NA,2,2,Pave,114000\n")
    f.write("NA,1,1,NA,112000\n")
data2 = pd.read_csv(data_file2)
print("\n原始数据:\n",data2)
# 计算每列的缺失值数量
missing_values = data2.isnull().sum()
print("\n缺失值数量:\n",missing_values)
# 缺失值最多的列
max_missing_col = missing_values.idxmax()
print("\n缺失值最多的列:\n",max_missing_col)
# 删除缺失值最多的列
data2_del = data2.drop(columns=[max_missing_col])
print("\n删除缺失值最多的列:\n",data2_del)
# 处理缺失值
inputs2, outputs2 = data2_del.iloc[:, :-1], data2_del.iloc[:, -1]
# 数值型数据均值插值
inputs2 = inputs2.fillna(inputs2.mean(numeric_only=True))
print("\n数值型数据均值插值后的数据:\n",inputs2)
# 类别型数据处理
inputs2 = pd.get_dummies(inputs2, dummy_na=True)
print("\n类别型数据处理后的数据:\n",inputs2)
# 转换为张量格式
X1 = torch.tensor(inputs2.to_numpy(dtype=float))
y1 = torch.tensor(outputs2.to_numpy(dtype=float))
print("\n张量格式的数据:\n", X1, "\n", y1)
