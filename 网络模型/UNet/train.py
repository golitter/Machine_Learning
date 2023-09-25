from torch import nn,optim # 优化器
import torch 
from torch.utils.data import DataLoader # 用于加载自定义数据集类
from data import * # 导入自定义类
from net import * # 导入UNet网络模型
from torchvision.utils import save_image # 导入保存图像方法


# 如果有cuda，就用；否则就用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 权重 用于存储和加载训练好的Unet深度学习模型的权重或参数。".pth" 是PyTorch模型文件的命名约定。
weight_path = "params/unet.pth"
# 数据集路径
data_path = r"E:\Undergraduate\School\Scientific_research\ML\Machine_Learning\lab\图形分割\Unet模型总\VOCdevkit\VOC2012"
# 训练结果图像保存路径
save_path = "train_image"

# 在主程序中运行
if __name__ == "__main__":
    # 创建数据加载器对象，MyDataset 是自定义的数据集类，用于加载训练数据，batch_size=2 表示每次训练使用的图像数量为 2。
    data_loader = DataLoader(MyDataset(data_path), batch_size=2, shuffle=True)
    # 实例化UNet网络模型 通过 .to(device) 将模型移动到指定的设备上。
    net = UNet().to(device)

    # 检查是否存在预训练的模型权重文件，如果存在则加载权重到模型中，否则输出提示信息。
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("Successful load weight!")
    else:
        print("Not successful load weight")
    
    # 创建优化器和损失函数对象
    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    # 设置起始训练轮数，并开始训练
    epoch = 1
    while True:
        # 遍历数据加载器中的每个批次，将图像数据和分割图像数据移动到指定设备上。
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # 前向传播计算网络输出结果，并计算训练损失。
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)

            # 梯度清零，反向传播计算梯度。
            opt.zero_grad()
            train_loss.backward()

            # 隔一段时间进行打印信息
            if i%5 == 0:
                print(f"{epoch} {i} - train_loss ==>{train_loss.item()}")

            if i%50 == 0:
                torch.save(net.state_dict(), weight_path)
            
            # 从批次中取出第一张图像、分割图像和网络输出结果。将图像、分割图像和网络输出结果按顺序堆叠，并保存为图像文件。
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]
            
            img = torch.stack([_image, _segment_image, _out_image],dim = 0)
            save_image(img, f"{save_path}/{i}.png")
            
        epoch += 1
