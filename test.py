import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn

model = retinanet_resnet50_fpn(weights=None)
model.eval()

x = [torch.rand(3, 800, 800)]
out = model(x)

print("Forward pass OK")


