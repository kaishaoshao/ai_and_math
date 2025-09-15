# 验证torch是否匹配

import torch
import torch.nn as nn

# 测试torch版本，检测是否安装
def print_torch_version():
    print("PyTorch Version:",torch.__version__)

# 创建并打印一个随机初始化的张量
def print_random_tensor():
    x = torch.rand(5, 3)
    print("Random tensor:",x)
    print("Type of the tensor:",type(x))

# 检测CUDA的可用性
def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU")
        print("Current CUDA device:",torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available, Using CPU")

# 定义并运行一个简单的神经网络
def run_simple_net():
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet,self).__init__()
            self.fc1 = nn.Linear(5, 3) # 输入层到隐藏层

        def forward(self, x):
            x = self.fc1(x)
            return x
        
    # 创建网络实例
    net = SimpleNet()

    # 创建一个随机数据
    input = torch.randn(1,5)
    output = net(input)

    # 打印网络输出
    print("Network output:",output)

# main函数
def main():
    print_torch_version()
    print_random_tensor()
    print_random_tensor()
    run_simple_net()

# 程序入口
if __name__ == "__main__":
    main()