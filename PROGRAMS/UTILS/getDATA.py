import os
import cv2 as cv
import torch
import numpy as np
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, folder="D:/TEL-U PWT 2023/KULIAH/SEMESTER 3/IPSD/ASSESSMENT/DATASET/"):
        self.dataset = []
        classes = ['Chickenpox', 'Cowpox', 'Healthy', 'HFMD', 'Measles', 'Monkeypox']
        onehot = np.eye(len(classes))  # Membuat one-hot encoding untuk 6 kelas

        # Loop untuk Original Images
        for fold in range(1, 6):  # Misalkan ada 5 fold
            for class_index, class_name in enumerate(classes):
                train_folder = os.path.join(folder, 'Original Images', 'FOLDS', f'fold{fold}', 'Train', class_name)
                if os.path.exists(train_folder):
                    for image_name in os.listdir(train_folder):
                        image_path = os.path.join(train_folder, image_name)
                        image = cv.imread(image_path)
                        if image is not None:  # Periksa apakah gambar berhasil dibaca
                            image = cv.resize(image, (100, 100)) / 255.0  # Normalisasi gambar
                            self.dataset.append([image, onehot[class_index]])

        # Loop untuk Augmented Images
        for fold in range(1, 6):  # Misalkan ada 5 fold untuk augmented
            for class_index, class_name in enumerate(classes):
                augmented_folder = os.path.join(folder, 'Augmented Images', f'FOLDS_AUG', f'fold{fold}_AUG', 'Train', class_name)
                if os.path.exists(augmented_folder):
                    for image_name in os.listdir(augmented_folder):
                        image_path = os.path.join(augmented_folder, image_name)
                        image = cv.imread(image_path)
                        if image is not None:  # Periksa apakah gambar berhasil dibaca
                            image = cv.resize(image, (100, 100)) / 255.0  # Normalisasi gambar
                            self.dataset.append([image, onehot[class_index]])

        print(f"Total samples: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        features, label = self.dataset[item]
        return (torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))

if __name__ == "__main__":
    data = Data()