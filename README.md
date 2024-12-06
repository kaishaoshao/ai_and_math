# d2l_ai

动手学习深度学习学习笔记[https://zh-v2.d2l.ai/]

Triton 学习笔记

## 安装并使用miniconda

miniconda常见语法

```bash
# 激活conda环境
source ~/.bashrc
conda

# 换源
# 产生.condarc文件
conda config --set show_channel_urls yes

# 如果不想进入终端默认激活base环境
conda config --set auto_activate_base false
# 换回默认源（清除所有用户添加的镜像源路径，只保留默认的路径）
conda config --remove-key channels
# 换其他源
vim ~/.condarc
```

```bash
# 清华源

channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

创建环境

```bash
# 查看conda信息
conda info 

# 创建环境
conda create -n d2l python=3.12
# 查看
conda env list
# 进入环境
conda activate d2l
# 退出
conda deactivate

# 删除环境
conda env remove -n d2l

# 安装pytorch
conda install pytorch torchvision cpuonly -c pytorch
```
