import numpy as np
import torch
from torchvision import datasets, transforms
import os

# 可选：若想同时生成真正的 MATLAB .mat 文件，设置 True
SAVE_AS_MATLAB = False

def ds_to_numpy(dataset):
    """
    dataset: torchvision.datasets (with transform=ToTensor())
    返回 (images, labels)：
      images: numpy array, dtype=float32, shape (N, 28, 28, 1), values in [0,1]
      labels: numpy array, dtype=float32, shape (N,)
    """
    imgs = []
    labs = []
    for img, lbl in dataset:  # img: torch.Tensor (1,28,28), lbl: int
        # ensure float32 and detach CPU
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        imgs.append(img.numpy())        # shape (1,28,28)
        labs.append(np.float32(lbl))
    imgs = np.stack(imgs, axis=0)       # shape (N,1,28,28)
    # transpose to (N,28,28,1) to be consistent with TFDS channel-last
    imgs = np.transpose(imgs, (0, 2, 3, 1)).astype(np.float32)  # (N,28,28,1)
    labs = np.asarray(labs, dtype=np.float32)
    return imgs, labs

def main(root='./data'):
    transform = transforms.ToTensor()  # yields float32 in [0,1], shape (1,28,28)
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    train_images, train_labels = ds_to_numpy(train_ds)
    test_images,  test_labels  = ds_to_numpy(test_ds)

    # 与你原脚本行为一致：保存为原始二进制（.mat 扩展名只是文件名，非 MATLAB 格式）
    os.makedirs('out', exist_ok=True)
    train_images.tofile('out/train_images.mat')
    train_labels.tofile('out/train_labels.mat')
    test_images.tofile('out/test_images.mat')
    test_labels.tofile('out/test_labels.mat')

    print('train_images', train_images.shape, train_images.dtype)
    print('train_labels', train_labels.shape, train_labels.dtype)
    print('test_images', test_images.shape, test_images.dtype)
    print('test_labels', test_labels.shape, test_labels.dtype)

    if SAVE_AS_MATLAB:
        try:
            from scipy import io
            io.savemat('out/mnist_matlab.mat', {
                'train_images': train_images,
                'train_labels': train_labels,
                'test_images': test_images,
                'test_labels': test_labels,
            })
            print('Also saved MATLAB .mat: out/mnist_matlab.mat')
        except Exception as e:
            print('scipy.io.savemat failed:', e)
            print('Install scipy if you want real .mat files: pip install scipy')

if __name__ == '__main__':
    main()
