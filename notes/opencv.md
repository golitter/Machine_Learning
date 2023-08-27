`cv2`库安装，

```shell
conda install opencv-python
```

> **注意cv2使用时，路径不能有中文**。（不然会一直'None' _
>
> update 
>
> ```python
> # 处理中文路径问题
> def cv_imread(file_path): #使用之前需要导入numpy、cv2库，file_path为包含中文的路径
>     return cv2.imdecode(np.fromfile(file_path, dtype=numpy.uint8), cv2.IMREAD_COLOR)
> 
> ```
>
>  --- [cv2入门函数imread及其相关操作_cv2.imread_trust Tomorrow的博客-CSDN博客](https://blog.csdn.net/liudadaxuexi/article/details/115986619?ops_request_misc=%7B%22request%5Fid%22%3A%22169233659616777224426523%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=169233659616777224426523&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-115986619-null-null.142^v93^insert_down1&utm_term=imread&spm=1018.2226.3001.4187)

cv2简单操作

#### 读入图片

```python
img = cv2.imread(filename[, flags])
"""
filename 表示要读取的图像文件路径。

** cv2的读入顺序是 BGR **

flags 可选参数
	- cv2.IMREAD_COLOR：默认模式，加载彩色图像。图像透明度会被忽略，默认使用 8 位深度的 3 通道图像。
	- cv2.IMREAD_GRAYSCALE：以灰度模式加载图像。图像的透明度将被忽略。
	- cv2.IMREAD_UNCHANGED：以包含 Alpha 通道的模式加载图像。图像的透明度将被保留。
"""
```

e.g.

```python
import cv2
import os
img = cv2.imread("./imgs/cnn.webp",1)
img0 = cv2.imread("./imgs/cnn.webp",0)
cv2.imshow('Original',img)
# print(img) 
cv2.waitKey(0)
cv2.destroyAllWindows()  # 关闭图像窗口

```

![image-20230818134649728](https://cdn.789ak.com/img/image-20230818134649728.png)

![image-20230818134728445](https://cdn.789ak.com/img/image-20230818134728445.png)

#### 显示图片

```python
cv2.imshow("windowname", img_name)
"""
windowname 是该窗口的名字
img_name 是要显示的图片
"""
```

#### 关闭图片

```python
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
cv2.waitKey(0) 和 cv2.destroyAllWindows() 的作用是等待用户按下键盘上的任意键，并在按下键后关闭所有图像窗口。

cv2.waitKey(0) 是一个键盘绑定函数，它会等待用户按下键盘上的一个键。参数 0 表示无限期等待用户按键，直到用户按下键盘上的任意键为止。返回值是用户按键的 ASCII 值（整数类型）。

cv2.destroyAllWindows() 函数用于关闭所有的图像窗口。当你的程序执行到该语句时，所有的图像窗口将会被关闭。
"""
```

