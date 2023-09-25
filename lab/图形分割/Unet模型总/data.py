import os # 路径操作
from torch.utils.data import Dataset # 继承 pytorch中的Dataset
from utils import * # 引入自己写的： 将原始图片和标记图片格式和规格统一
from torchvision import transforms # 用于数据预处理

################################################
#
#   transforms.Compose用于将多个变换操作组合在一起。
# 这里只有一个 ToTensor()，用于将图像数据转为张量格式
#
###############################################
transform = transforms.Compose([
    transforms.ToTensor()
])

##############################################
#
# 通过继承Dataset类，来实现自定义数据集类
#
##############################################
class MyDataset(Dataset):

    # path表示数据集的路径
    def __init__(self, path):
        self.path = path
        # path路径下的目录
        self.name = os.listdir(os.path.join(path, "SegmentationClass"))
    
    # 返回数据集长度
    def __len__(self):
        return len(self.name)
    
    # 实现下标访问
    def __getitem__(self, index):
        segment_name = self.name[index] # xx.png 得到的是：path/SegmentationClass/图像名字
        segment_path = os.path.join(self.path, "SegmentationClass", segment_name) # 得到标记图像路径
        # 得到原始图像路径。segment_name.replace(xxx,kkk) 表示如果标记图像和原始图像格式不同，进行后缀替代，进而得到正确的原始图像路径
        image_path = os.path.join(self.path, "JPEGImages", segment_name.replace("png", "jpg")) 

        # 将图像进行统一规格化
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)

        # 返回经过transform处理后的图像张量
        return transform(image), transform(segment_image)

# 测试是否正确
if __name__ == "__main__":
    data = MyDataset(r"E:\Undergraduate\School\Scientific_research\ML\Machine_Learning\lab\图形分割\Unet模型总\data")
    print(data[0][0].shape)
    print(data[0][1].shape)