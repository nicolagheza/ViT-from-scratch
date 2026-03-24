import os
import csv
import time
import torch
import torch.nn as nn
from config import Config
from model.vit import ViT
from data import get_dataloaders
from evaluate import evaluate

def train():
    os.makedirs("checkpoints", exist_ok=True)
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
    trainloader, testloader = get_dataloaders(config)

    with open("training_log.csv", "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "accuracy", "lr"])

    best_accuracy = 0.0
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches
        accuracy = evaluate(model, testloader, device)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"Epoch {epoch+1}/{config.epochs} — Loss: {epoch_loss:.4f} — Acc: {accuracy:.2f}% — LR: {lr:.6f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "checkpoints/best.pth")

        with open("training_log.csv", "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, epoch_loss, accuracy, lr])

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    torch.save(model.state_dict(), "checkpoints/final.pth")

if __name__ == "__main__":
    train()