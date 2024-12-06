import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import models
from UTILS.getDATA import Data

def main():
    # PARAMETER
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # DATA LOADER
    train_loader = DataLoader(Data(), batch_size=BATCH_SIZE, shuffle=True)

    # MODEL: Pretrained MobileNetV2
    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    model.classifier[1] = nn.Linear(model.last_channel, 6)  # Output untuk 6 kelas
    device = "cpu"
    model = model.to(device)

    # OPTIMIZER dan LOSS FUNCTION
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss() # Menggunakan CrossEntropy

    # Menyimpan loss untuk plotting
    loss_all = []

    # TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        loss_train = 0

        for batch, (src, trg) in enumerate(train_loader):
            # Preprocessing untuk batch data
            src = torch.permute(src, (0, 3, 1, 2))  # Ubah (N, H, W, C) menjadi (N, C, H, W)
            src, trg = src.to(device), trg.to(device)

            pred = model(src)
            loss = loss_fn(pred, torch.max(trg, 1)[1])  # Konversi one-hot label ke index

            loss_train += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Rata-rata loss per epoch
        avg_loss = loss_train / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
        loss_all.append(avg_loss)

    # Plot loss training
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_all, color="#931a00", label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.legend()
    plt.savefig("./training_loss.jpg")

    # Simpan model
    torch.save(model.state_dict(), "mobilenet_v2_trained.pth")
    print("Model has been saved to 'mobilenet_v2_trained.pth'.")

if __name__ == "__main__":
    main()