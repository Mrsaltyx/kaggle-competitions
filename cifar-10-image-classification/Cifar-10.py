import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

# =======================
# 📂 DATASETS
# =======================

class CustomImageDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None):
        self.labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        base_name = str(self.labels.iloc[idx, 0])
        label = int(self.labels.iloc[idx, 1])
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(self.img_dir, base_name + ext)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
        raise FileNotFoundError(f"Aucune image trouvée pour {base_name}")

class TestImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_names = sorted([
            f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name

# =======================
# 🧠 MODELE CNN
# =======================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# =======================
# 🚀 ENTRAINEMENT + PRED
# =======================

def main():
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Préparation des données
    df = pd.read_csv("trainLabels.csv")
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])
    df[["id", "label_encoded"]].to_csv("encoded_trainLabels.csv", index=False)
    num_classes = len(le.classes_)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = CustomImageDataset("encoded_trainLabels.csv", "train/train", transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TestImageDataset("test/test", transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses, accuracies = [], []

    print("\n🔁 Début de l'entraînement...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        print(f"📘 Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    # 📊 Graphique
    plt.plot(range(1, num_epochs+1), losses, label='Loss')
    plt.plot(range(1, num_epochs+1), accuracies, label='Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Valeur")
    plt.title("Courbe Loss & Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_accuracy_curve.png")
    plt.show()

    # 🧪 Prédictions test
    print("\n🔍 Prédiction sur les images de test...")
    model.eval()
    predictions = {}
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for name, pred in zip(names, preds):
                try:
                    img_id = int(os.path.splitext(name)[0])
                    predictions[img_id] = pred.item()
                except ValueError:
                    print(f"❗ Image ignorée (nom invalide) : {name}")

    print(f"✅ {len(predictions)} images test prédites")

    # 💾 Génération CSV
    try:
        sample_df = pd.read_csv("sampleSubmission.csv")
        ordered_ids = sample_df["id"].tolist()

        missing_ids = [i for i in ordered_ids if i not in predictions]
        if missing_ids:
            print(f"⚠️ Attention : {len(missing_ids)} IDs non trouvés dans les prédictions.")

        predicted_values = [
            predictions[i] if i in predictions else 0
            for i in ordered_ids
        ]
        decoded_labels = le.inverse_transform(predicted_values)

        submission = pd.DataFrame({
            "id": ordered_ids,
            "label": decoded_labels
        })
        submission.to_csv("submission.csv", index=False)
        print("✅ Fichier 'submission.csv' créé avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de la génération du CSV : {e}")

if __name__ == "__main__":
    main()
