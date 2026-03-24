import torch
from config import Config
from model.vit import ViT
from data import get_dataloaders

def evaluate(model, testloader, device):
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
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ViT(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)
    model.load_state_dict(torch.load("checkpoints/best.pth"))
    _, testloader = get_dataloaders(config)
    accuracy = evaluate(model, testloader, device)
    print(f"Accuracy: {accuracy:.2f}%")