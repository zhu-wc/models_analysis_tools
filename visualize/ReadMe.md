## 可视化



### 安装

遵循[here](https://github.com/luo3300612/Visualizer)中的内容在本地python环境中安装`Visualizer`工具包。



### 用法

(1) 修饰注意力函数

通过get_local对准备可视化的注意力函数做修饰。传入修饰器中的参数可以自由命名，随后获取的cache中将会以此为`key`进行保存。

```python
from visualizer import get_local
get_local.activate()

@get_local('attention_map')
def your_attention_function(*args, **kwargs):
    ...
    attention_map = ... 
    ...
    return ...
```

（2）获取cache

在模型被执行后，`get_locah.cache`中的数据即为所求。

```python
from visualizer import get_local
get_local.activate() # 激活装饰器
from ... import model # 被装饰的模型一定要在装饰器激活之后导入！！

# load model and data
...
out = model(data)

cache = get_local.cache # ->  {'your_attention_function': [attention_map]}
```



### 可视化VIT中的注意力

遵循[here](https://github.com/luo3300612/Visualizer)中的内容进行



### 可视化Image Captioning中的注意力

（1）修改`attention.py`的代码

在`ScaledDotProductAttention`类的`forward`方法前添加`@get_local('att')`，同时在`import get_local`后通过`activate()`方法激活装饰器。

（2）修改`eval.py`中的代码

在evaluate_metrics函数中添加代码，在模型被执行后添加如下代码。

```python
cache = get_local.cache
torch.save(cache,'att_map.pth')
sys.exit()
```

（3）`att_map.pth`结构分析

执行单个batch后结束程序，文件中保存的是执行中捕获到的注意力文件。形式上表现为一个长度为120的列表。具体来说，每一个样本生成20个token，即模型重复调用了20次，每次调用后注意力函数在编码器和解码器中分别执行了3次（编码和解码层均设置为3）。

综上所示，在长度为120的列表中，

>step0: 
>
>encoder_layer: 列表中索引为0-2处储存的数据 
>
>decoder_layer: 列表中索引为3-5处储存的数据
>
>step1~step19:
>
>依照上述规律依次排序

（4）获取最后一个解码层中的注意力权重

通过下面的代码，以6个为一组遍历注意力文件中的数据，接着将对应位置处的注意力权重送入绘图函数中做可视化。

```python
for head_num in range(8):
    for step in range(11):
        layer_num = (step+1)*6
        att_map = batch5[layer_num-1][image_num,head_num,:,:]#(1 49)
        visualize_text_to_grid(att_map, image,head_num,step)
```

（5）补充

当实例化的`model`对象被多次执行时（执行了多个`batch`），通过`cache = get_local.cache`获取到的注意力文件也会在列表中累计。即执行一次后，列表长度为120，执行两次后，列表长度为240，以此类推。如果想要可视化第0个batch以后的样本的注意力，将获取到的结果以120为一组划分即可。

(6) 代码

main.py中给出了根据注意力权重列表，对第5个batch中，第2个样本进行可视化的代码。
