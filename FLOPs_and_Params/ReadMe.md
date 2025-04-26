## 获取模型的参数量和计算量



参数量和计算量均可以通过thop工具包进行计算。给出通过thop计算标准Image Captioning模型的代码`IC_FLOPs_by_thop.py`和计算ResNet101模型的代码`ResNet101_by_thop.py`



### Image Captioning

在Image Captioning项目的`train.py`中执行训练和推理的`model`实际上是对`transformer.py`中的`Transformer`类进行实例化产生的对象。该对象中的`forward`方法中需要接收形状为`(bs,nog,dim)`的矩阵作为输入。



基于`thop`工具包的特点，在计算参数量和计算量时，需要在原有模型所接收的输入矩阵前添加一个维度，即在下面代码中，需要将`input`的形状设置为`(1,bs,nog,dim)`，否则会报错。

>```python
>flops, params = profile(use_model, inputs=(input))
>```



### ResNet101

获取`ResNet101`模型的参数量和计算量非常简单，直接从`torchvision.models`中引入`resnet101`,随后调用`thop`包中的方法即可。同样的，`ResNet101`模型所需的输入矩阵的形状为`(3,224,224)`，所以调用thop时需要将input的形状设置为`(1,3,224,224)`



### Results

(1) 基于Transformer的Image Captioning模型，其参数量和计算量分别为: 

>FLOPs= 10.974547968G
>
>params= 28.40832M

模型内使用的超参数包括：接收形状为`(49,2048)`的网格特征作为输入，MHA内部维度`d_model=512`，`multi_head=8`，FFN中隐藏层维度`d_ff=2048`，编码和解码层的数量`L=3`。



(2) ResNet101模型的参数量和计算量为：

>FLOPs=44.6M
>
>params=7.8G FLOPs
