# 使用MNIST數據集

import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms # torchvision是PYtorch中專門用來處理圖像的庫
from d2l import torch as d2l

d2l.use_svg_display()

# 1.讀取數據集
# 通過ToTensor實例將圖像數據從PIL轉成32位float
# 將此float除以255，使得數據介於0 ~ 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST( # Fashin-MNIST由十個類別圖像組成，其中每個類別有6000train datasets，1000test datasets
    root="../data", train=True, transform=trans, download=True,) # Fashion-MNIST数据集下载并读取到内存中
mnist_test = torchvision.datasets.FashionMNIST(
    root= "../data", train=False, transform=trans, download=True) # 藉由train = False使之自動將其視為testsets

# print(len(mnist_train),len(mnist_test))
print(mnist_test[0][0].shape) # 每個輸入圖像高度和寬度接為28pixels data_sets由灰度圖像組成，通道數為1

def get_fashion_mnist_labels(labels):
    # 返回Fasion-MNIST數據集對應到的文本標籤
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', # fasion-mnist共有十個衣服類別
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels] # 將 labels 中的數字（類別索引）轉換為對應的文字標籤，並返回這些文字標籤組成的列表

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  
    """绘制图像列表并在本地生成窗口显示"""
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # 将所有子图展开成一维数组，方便逐个操作
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 如果是 Tensor，将其转为 NumPy 格式以供绘图
            ax.imshow(img.numpy(), cmap='viridis')  # Fashion-MNIST 是灰度图，添加 `cmap='gray'`
        else:
            # 如果是 PIL 图像或其他格式
            ax.imshow(img, cmap='gray')
        ax.axis('off')  # 隐藏坐标轴
        if titles:
            ax.set_title(titles[i])  # 设置标题
    plt.tight_layout()  # 自动调整子图之间的间距
    plt.show()  # 显式调用以在本地显示图像

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))


# 2.讀取小批量
batch_size = 256
    
def get_dataloader_workers():
    # 使用4個process來讀數據
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers= get_dataloader_workers())

# timer = d2l.Timer()
# for X, y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec') # 讀取訓練數據所需時間


# 3.整合所有組件
def load_data_fation_mnist(batch_size, resize = None):
    # "下載FASION-MNIST數據集，然後加載到記憶體中"
    trans = [transforms.ToTensor()] # 將PIL轉換成tensor型態
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans) # 作用是串聯多個圖片transform的操作
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, # 回傳訓練集、測試集
                           num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=True,
                num_workers=get_dataloader_workers()))
    
train_iter, test_iter = load_data_fation_mnist(batch_size=32,resize=64)
for X,y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

# 如此我們已準備好使用Fashion-MNIST數據集了