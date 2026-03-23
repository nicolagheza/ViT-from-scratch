
import torch

from config import Config
from model.vit import ViT
from data import get_dataloaders

config = Config()

_, testloader = get_dataloaders(config)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = ViT(
    img_size=config.img_size,
    patch_size=config.patch_size,
    in_channels=config.in_channels,
    d_model=config.d_model,
    num_heads=config.num_heads,
    num_layers=config.num_layers,
    num_classes=config.num_classes
).to(device)

model.load_state_dict(torch.load("checkpoint.pth"))

def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    print(f"Accuracy: {100 * correct / total:.2f}%")

evaluate(model, testloader)