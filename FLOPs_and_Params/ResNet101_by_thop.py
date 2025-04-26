import torch
from torchvision.models import resnet101
from thop import profile


model = resnet101()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
print("params=", str(params / 1e6) + '{}'.format("M"))