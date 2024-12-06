import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from UTILS.getDATA import Data

def test(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Untuk menyimpan probabilitas kelas

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.permute(0, 3, 1, 2))

            # Simpan probabilitas kelas
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            
            # Simpan prediksi kelas dan label
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.max(labels, 1)[1].cpu().numpy())

    # Hitung metrik evaluasi
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')  # Perbaikan: Menggunakan probabilitas

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(np.unique(all_labels))))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("./confusion_matrix.jpg")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cpu")
    test_loader = DataLoader(Data(), batch_size=8, shuffle=False)

    # Load model
    from torchvision import models
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 6)
    model.load_state_dict(torch.load("mobilenet_v2_trained.pth", map_location=device))
    model = model.to(device)

    test(model, test_loader, device)