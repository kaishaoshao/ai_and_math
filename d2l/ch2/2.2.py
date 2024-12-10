# 我们首先创建一个人工数据集，并存储再CSV文件../data/house_tiny.csv中，
# 以其他格式存储的数据可以通过类似的方式进行处理
import os

os.makedirs(os.path.join('./', 'data'), exist_ok=True)
data_file = os.path.join('./', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write("NumRooms,Alley,Price\n")  # 列名
    f.write("NA,Pave,127500\n")  # 每行表示一个数据样本
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

