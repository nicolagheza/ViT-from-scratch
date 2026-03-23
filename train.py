import torch
import torch.nn as nn

from config import Config
from model.vit import ViT
from data import get_dataloaders

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

config = Config()
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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

trainloader, _ = get_dataloaders(config)

for epoch in range(config.epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
    scheduler.step()

torch.save(model.state_dict(), "checkpoint.pth")
