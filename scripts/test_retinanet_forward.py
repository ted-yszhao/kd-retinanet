import torch
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"

model = retinanet_resnet50_fpn(weights=None).to(device)
model.eval()

images = [torch.rand(3, 800, 800, device=device)]

with torch.no_grad():
    outputs = model(images)

print("forward ok")
print("num outputs:", len(outputs))
print("keys:", outputs[0].keys())
print("boxes shape:", outputs[0]["boxes"].shape)
print("labels shape:", outputs[0]["labels"].shape)
print("scores shape:", outputs[0]["scores"].shape)

