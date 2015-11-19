#  反向传播算法


---

上一章对神经网络有了基本认识，在参数(权重和偏置)学习过程中我们知道要使用梯度下降算法。然而，计算梯度可并不简单。这一章介绍反向传播算法——用于快速计算梯度。


![](https://ooo.0o0.ooo/2015/11/10/5641ec00a6aff.png)

通过上式可以发现，计算梯度实际上就是计算偏导数:$$\partial C / \partial w$$, $$\partial C / \partial b$$。



## 矩阵形式

在学习反向传播算法之前，先介绍一种参数的矩阵表示方法。

几个要用到的数学符合：$$w_{jk}^{l} $$表示第l-1层第k个神经元与第l层第j个神经元之间的权重。下图以$$w_{24}^{3}$$为例

![](https://ooo.0o0.ooo/2015/11/16/5649fcc7422ee.png)

这样的表示对于初学者是比较别扭的，反正我第一次看到觉得很奇怪，明明是第k个神经元指向第j个神经元，怎么偏偏写成$$jk$$?暂时先不想这个问题，

继续介绍数学符号:$$b_{j}^{l}$$表示第l层第j个神经元的偏置，$$a_{j}^{l}$$表示第l层第j个神经元的激活值（输出值），






![](https://ooo.0o0.ooo/2015/11/16/5649ff9fd3da8.png)

其中$$a_{j}^{l}$$的计算公式：

![](https://ooo.0o0.ooo/2015/11/16/564a0063c9611.png)

为了将上式转为矩阵形式，我们为每一层神经网络定义一个权重矩阵$$w^{l}$$、偏置向量$$b^{l}$$、激活向量$$a^{l}$$。以上图为例，$$w^{3}$$是一个2行4列矩阵。

我们还定义这样的函数操作：如果函数输入是向量，函数对向量中每一点元素进行操作，因此函数输出也是向量。 比如$$f(x)=x^{2}$$,

![](https://ooo.0o0.ooo/2015/11/16/564a03b350b66.png)



下面我们重写(23)

![](https://ooo.0o0.ooo/2015/11/16/564a041ad42a0.png)

式子(25)简单易懂，直观明了，我们定义权重输入$$z^{l}= w^{l}a^{l - 1}+b^{l} $$, 所以(25)又可以写成 $$a^{l}=\sigma (z^{l})$$。



## 关于损失函数的两个假设

计算损失函数的梯度实质上就是计算偏导数$$\partial C/\partial w,\partial C / \partial b$$.
反向传播算法本质上也是计算偏导数，不过在运用反向传播算法之前，我们需要对损失函数作出两个假设。

提醒一下，我们这里使用的是平方损失函数

![](https://ooo.0o0.ooo/2015/11/17/564ac97f96fca.png)

第一个假设：整体损失函数可写成$$C=\frac{1}{n}\sum_{x}C_{x}$$,其中$$C_{x}$$是每一个训练实例的损失函数。
为什么要做这个假设呢？这是因为我们运用反向传播算法时并不是计算整体算是函数对权重/偏置的偏导数，而是计算$$\partial C_{x}/\partial w,\partial C_{x} / \partial b$$,然后取平均值。

第二个假设是损失函数C可以写作神经网络输出的函数：

![](https://ooo.0o0.ooo/2015/11/17/564acc19f13c4.png)

我们使用的平方损失函数就满足这个假设

![](https://ooo.0o0.ooo/2015/11/17/564acc5a79ac2.png)


## Hadamard乘积, s$$\odot$$t

假设s,t是两个同维度向量，Hadamard乘积定义如下：

![](https://ooo.0o0.ooo/2015/11/17/564acec3a81c6.png)

矩阵的Hadamard乘积：

![](https://ooo.0o0.ooo/2015/11/17/564acef5cf5df.png)

Hadamard乘积实质上就是同位置两个元素的点乘。



## 四个基本公式

反向传播算法不但能够计算偏导数，还能帮助我们理解 改变权重和偏置如何影响神经网络的输出。

为此，引入变量$$\delta_{j}^{l}$$表示第l层第个神经元的误差。下面这幅图会帮助我们理解'误差'这个概念。

如图所示，有一个恶魔位于第l层第j个神经元，恶魔的乐趣就是捣蛋，神经元计算好$$z_{j}^{l}$$之后，恶魔偷偷地加上一个$$\Delta z_{j}^{l}$$,原本神经元的输出是$$\sigma (z_{j}^{l})$$,现在变成了$$\sigma(z_{j}^{l} + \Delta z_{j}^{l})$$，并且这个结果会向后传播，影响到最后神经网络的输出，这一点蝴蝶效应会有多大偏差呢？答案是

$$\frac{\partial C}{\partial z_{j}^{l}}\Delta z_{j}^{l}$$。

![](https://ooo.0o0.ooo/2015/11/17/564ad0313c555.png)

定义

![](https://ooo.0o0.ooo/2015/11/17/564ae2c2e7989.png)


### 输出层误差

![](https://ooo.0o0.ooo/2015/11/17/564ae4378d587.png)

不要被第一部分的偏导数吓到，还记得第二个假设么？此处$$C=\frac{1}{2}\sum_{j}(y_{j}-a_{j})^{2}$$,因此$$\partial C/\partial a_{j}^{L} = (a_{j} - y_{j})$$。

BP1 可以写成向量形式：

![](https://ooo.0o0.ooo/2015/11/17/564ae5fd6d43a.png)


![](https://ooo.0o0.ooo/2015/11/17/564ae6681b041.png)


### 相邻层之间误差关系

![](https://ooo.0o0.ooo/2015/11/17/564ae70081d5f.png)

有个以上两个公式，我们就能计算每一层的误差了。先计算输出层误差，然后计算L-1,L-2,...


### 损失函数对偏置的偏导数

![](https://ooo.0o0.ooo/2015/11/17/564ae866d7e70.png)

看到这里，不需要说啥了，数学确实很奇妙。。。

更简洁的向量形式：

![](https://ooo.0o0.ooo/2015/11/17/564ae8ccf1a07.png)


### 损失函数对权重的偏导数

![](https://ooo.0o0.ooo/2015/11/17/564ae926e1170.png)


啥也别说了，继续看向量形式吧：

![](https://ooo.0o0.ooo/2015/11/17/564ae96397971.png)

其中$$a_{in}, \delta _{out}$$可以用下图帮助理解，

![](https://ooo.0o0.ooo/2015/11/17/564aea361546f.png)

注意：上述四个公式中激活函数可以是任意函数，不限于sigmoid函数。

最后，再总结一下四个公式。

![](https://ooo.0o0.ooo/2015/11/17/564aeb4401c16.png)


## **证明**

这一节要对上节提到的四个基本公式进行证明。证明过程很简单，使用链式法则即可。

### BP1


$$\because \delta _{j}^{L} = \frac{\partial C}{\partial z_{j}^{L}} = \sum_{k}\frac{\partial C}{\partial a_{k}^{L}}\frac{\partial a_{k}^{L}}{\partial z_{j}^{L}}$$

$$\because \partial a_{k}^{L}/\partial z_{j}^{L} =0,$$ when $$k\neq j$$

$$\therefore \delta _{j}^{L} = \frac{\partial C}{\partial a_{j}^{L}}\frac{\partial a_{j}^{L}}{\partial z_{j}^{L}}$$

$$\because a_{j}^{L}=\sigma(z_{j}^{L})$$

$$\therefore \frac{\partial C}{\partial a_{j}^{L}} = \sigma '(z_{j}^{L})$$

$$\therefore \delta _{j}^{L}=\frac{\partial C}{\partial a_{j}^{L}}\sigma '(z_{j}^{L})$$


### BP2

$$\delta_{j}^{l} = \frac{\partial C}{\partial z_{j}^{l}} = \sum_{k}\frac{\partial C}{\partial z_{k}^{l+1}}\frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}} = \sum_{k}\delta_{k}^{l+1}\frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}}$$




$$\because z_{k}^{l+1}=\sum_{j}w_{kj}^{l+1}a_{j}^{l}+b_{k}^{l+1}=\sum_{j}w_{kj}^{l+1}\sigma(z_{j}^{l})+b_{k}^{l+1}$$

$$ \frac{\partial z_{k}^{l+1}}{\partial z_{j}^{l}}=w_{kj}^{l+1}\sigma '(z_{j}^{l})$$


$$\therefore \delta_{j}^{l}=\sum_{k}w_{kj}^{l+1}\delta_{k}^{l+1}\sigma '(z_{j}^{l})$$


### BP3

$$\because z_{j}^{l}=\sum_{i}w_{ji}^{l}a_{i}^{l-1}+b_{j}^{l}$$

$$\therefore \frac{\partial C}{\partial b_{j}^{l}} = \frac{\partial C}{\partial z_{j}^{l}}\frac{\partial z_{j}^{l}}{\partial b_{j}^{l}}=\frac{\partial C}{\partial z_{j}^{l}} = \delta_{j}^{l}$$




### BP4





$$ \because z_{j}^{l}=\sum_{k}w_{jk}^{l}a_{k}^{l-1}+b_{j}^{l}$$

$$\therefore \frac{\partial C}{\partial w_{jk}^{l}} = \frac{\partial C}{\partial z_{j}^{l}}\frac{\partial z_{j}^{l}}{\partial w_{jk}^{l}}=\delta_{j}^{l}a_{k}^{l-1}$$




## 反向传播算法

有了上面的计算公式，算法流程很容易得到：

1. **输入**x:设置$$a^{1}$$.
2. **前向计算**: 计算每一层的$$z^{l},a^{l}$$
3. **计算输出层误差**$$\delta^{L}$$:使用BP1
4. **反向计算误差**：从l=L-1,....,2。计算$$\delta^{l}$$,使用BP2
5. **输出**:得到偏导数:$$\frac{\partial C}{\partial w_{jk}^{l}}$$,$$\frac{\partial C}{\partial b_{j}^{l}}$$,使用BP3,BP4.


在实际编程中，处理每一个mini-batch流程如下：

1.输入m个训练实例,mini-batch大小为m

2.对于每一个训练实例x:设置相应的$$a^{x,1}$$,然后进行如下操作:

(a)前向计算:对于每一层l=2,3,...,L,计算$$z^{x,l}=w^{l}a^{x,l-1}$$和$$a^{x,l} = \sigma (z^{x,l})$$
    
(b) 输出误差$$\delta^{x,L}$$: $$\delta^{x,L}=$$$$\nabla_{a}C_{x}\odot\sigma'(z^{x,L})$$

(c)反向传播计算误差:对于l=L-1,L-2,...,2.计算$$\delta^{x,l}=((w^{l+1})^{T}\delta^{x,l+1})\odot\sigma'(z^{x,l})$$

3.梯度下降：对于l=L,L-1,...,2.更新权重:$$w^{l} ->w^{l}-\frac{\eta}{m}\sum_{x}\delta^{x,l}(a^{x,l-1})^{T}$$,更新偏置:$$b^{l}->b^{l}-\frac{\eta}{m}\sum_{x}\delta^{x,l}
$$




##为什么说反向传播算法是fast算法?
快慢是相对的，这里的快指的是针对一个训练实例，只需要一次前向遍历和一次后向遍历就能得到所有的偏导数:$$\frac{\partial C}{\partial w}, \frac{\partial C}{\partial b}$$.











