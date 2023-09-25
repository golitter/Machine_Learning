from PIL import Image # 用于读取和处理图像

# 用于加载图像并保存图像大小一致
# path表示图像文件的路径，size表示需要调整的目标大小，默认为(256, 256)。
def keep_image_size_open(path, size = (256, 256)):

    # 通过Image.open(path)方法打开path路径的图像，并将返回的图像对象保存在变量img中。
    img = Image.open(path)
    # 获取输入图像的最大边长
    tmp = max(img.size)
    # 创建一个新的空白图像对象mask，大小为(tmp, tmp)，颜色为(0, 0, 0)，即黑色。
    mask = Image.new("RGB", (tmp, tmp), (0,0,0))
    # 将原始图像img粘贴到mask图像对象的左上角，使得原始图像位于黑色背景之上。
    mask.paste(img, (0,0))
    # 将图像调整为目标大小，并返回调整后的图像对象mask。
    mask = mask.resize(size)
    return mask


