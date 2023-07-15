[60分钟快速入门 PyTorch - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/66543791)

```python
import torch
```



张量表示一个由数值组成的数组，这个数组可能有多个维度。 具有一个轴的张量对应数学上的*向量*（vector）； 具有两个轴的张量对应数学上的*矩阵*（matrix）； 具有两个轴以上的张量没有特殊的数学名称。

### Tensor

Tensor的声明和定义

```python
x = torch.arange(12) # 使用arange创建一个行向量x，值为0~11
x = torch.tensor([1,2,3])
```

可以通过张量的`shape`属性来访问张量（沿每个轴的长度）的*形状* 。

```python
print(x.shape)
```

要想改变一个张量的形状而不改变元素数量和元素值，可以调用`reshape`函数。 例如，可以把张量`x`从形状为（12,）的行向量转换为形状为（3,4）的矩阵。 这个新的张量包含与转换前相同的值，但是它被看成一个3行4列的矩阵。 要重点说明一下，虽然张量的形状发生了改变，但其元素值并没有变。 注意，通过改变张量的形状，张量的大小不会改变。

```python
X2 = x.reshape(2,6)
```

我们不需要通过手动指定每个维度来改变形状。 也就是说，如果我们的目标形状是（高度,宽度）， 那么在知道宽度后，高度会被自动计算得出，不必我们自己做除法。 在上面的例子中，为了获得一个3行的矩阵，我们手动指定了它有3行和4列。 幸运的是，我们可以通过`-1`来调用此自动计算出维度的功能。 即我们可以用`x.reshape(-1,4)`或`x.reshape(3,-1)`来取代`x.reshape(3,4)`。

元素个数

```python
X.numel()
```

tensors 的尺寸大小获取可以采用 `tensor.size()` 方法。`torch.Size` 实际上是**元组(tuple)类型，所以支持所有的元组操作**。

```python
tensor1.size() #
```

#### 操作

加法

```python
torch.add(T1, T2)
T1.add_(T2)
```

**注意**：可以改变 tensor 变量的操作都带有一个后缀 `_`, 例如 `x.copy_(y), x.t_()` 都可以改变 x 变量

除了加法运算操作，对于 Tensor 的访问，和 Numpy 对数组类似，可以使用索引来访问某一维的数据。

```
T1(:,2) # 第三列数据
```

### 自动微分

对于 Pytorch 的神经网络来说，非常关键的一个库就是 `autograd` ，它主要是提供了对 Tensors 上所有运算操作的自动微分功能，也就是计算梯度的功能。它属于 `define-by-run` 类型框架，即反向传播操作的定义是根据代码的运行方式，因此每次迭代都可以是不同的。

`torch.Tensor` 是 Pytorch 最主要的库，当设置它的属性 `.requires_grad=True`，那么就会开始追踪在该变量上的所有操作，而完成计算后，可以调用 `.backward()` 并自动计算所有的梯度，得到的梯度都保存在属性 `.grad` 中。

调用 `.detach()` 方法分离出计算的历史，可以停止一个 tensor 变量继续追踪其历史信息 ，同时也防止未来的计算会被追踪。

而如果是希望防止跟踪历史（以及使用内存），可以将代码块放在 `with torch.no_grad():` 内，这个做法在使用一个模型进行评估的时候非常有用，因为模型会包含一些带有 `requires_grad=True` 的训练参数，但实际上并不需要它们的梯度信息。

对于 `autograd` 的实现，还有一个类也是非常重要-- `Function` 。

`Tensor` 和 `Function` 两个类是有关联并建立了一个非循环的图，可以编码一个完整的计算记录。每个 tensor 变量都带有属性 `.grad_fn` ，该属性引用了创建了这个变量的 `Function` （除了由用户创建的 Tensors，它们的 `grad_fn=None` )。

如果要进行求导运算，可以调用一个 `Tensor` 变量的方法 `.backward()` 。如果该变量是一个标量，即仅有一个元素，那么不需要传递任何参数给方法 `.backward()`，当包含多个元素的时候，就必须指定一个 `gradient` 参数，表示匹配尺寸大小的 tensor，这部分见第二小节介绍梯度的内容。

```python
#  y = x ^ 2
x=torch.tensor(3.0,requires_grad=True)
y=torch.pow(x,2)

#判断x,y是否是可以求导的
print(x.requires_grad)
print(y.requires_grad)

#求导，通过backward函数来实现
y.backward()

#查看导数，也即所谓的梯度
print(x.grad) # tensor(6.)
```



```python
#创建一个二元函数，即z=f(x,y)=x2+y2，x可求导，y设置不可求导
x=torch.tensor(3.0,requires_grad=True)
y=torch.tensor(4.0,requires_grad=False)
z=torch.pow(x,2)+torch.pow(y,2)

#判断x,y是否是可以求导的
print(x.requires_grad) # True
print(y.requires_grad) # False
print(z.requires_grad) # True

#求导，通过backward函数来实现
z.backward()

#查看导数，也即所谓的梯度
print(x.grad) # tensor(6.0)
print(y.grad) # None，y所在的变量相当于常量

```

`x.requires_grad_(True/False)` 设置tensor的可导与不可导，注意后面有一个下划线。但是需要注意的是，我只能够设置叶子变量，即leaf variable的这个方法，否则会出现错误。

需要注意的是：如果出现了复合函数，比如 y是x的函数，z是y的函数，f是z的函数，那么在求导的时候，会使用 f.backwrad()只会默认求f对于叶子变量leaf variable的导数值，而对于中间变量y、z的导数值是不知道的，直接通过x.grad是知道的、y.grad、z.grad的值为none。

总结：

（1）torch.tensor()设置requires_grad关键字参数

（2）查看tensor是否可导，x.requires_grad 属性

（3）设置叶子变量 leaf variable的可导性，x.requires_grad_()方法

（4）自动求导方法 y.backward() ，直接调用backward()方法，只会计算对计算图叶节点的导数。

（5）查看求得的到数值， x.grad 属性

#### 默认的求导规则

在pytorch里面，默认：只能是【标量】对【标量】，或者【标量】对向【量/矩阵】求导。





[(2条消息) pytorch自动求梯度—详解_pytorch自动求梯度原理_浩波的笔记的博客-CSDN博客](https://blog.csdn.net/weixin_44023658/article/details/107417063?ops_request_misc=&request_id=&biz_id=102&utm_term=pytorch 自动计算梯度&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-5-107417063.nonecase&spm=1018.2226.3001.4187)