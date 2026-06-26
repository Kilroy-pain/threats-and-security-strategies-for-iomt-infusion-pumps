import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Simulated dataset for anomaly detection in IoMT infusion pumps
class InfusionPumpDataset(Dataset):
    def __init__(self, num_samples=1000):
        np.random.seed(42)
        self.data = np.random.normal(0, 1, (num_samples, 10))  # Normal operational data
        self.labels = np.zeros(num_samples)  # 0 for normal
        # Inject anomalies
        num_anomalies = int(0.1 * num_samples)
        anomaly_data = np.random.normal(5, 1, (num_anomalies, 10))  # Anomalous data
        anomaly_labels = np.ones(num_anomalies)  # 1 for anomaly
        self.data = np.vstack((self.data, anomaly_data))
        self.labels = np.hstack((self.labels, anomaly_labels))
        # Shuffle dataset
        indices = np.arange(len(self.labels))
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Simple Autoencoder for anomaly detection
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder(model, dataloader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, _ in dataloader:
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

def evaluate_autoencoder(model, dataloader, threshold):
    model.eval()
    anomalies = []
    with torch.no_grad():
        for data, labels in dataloader:
            reconstructed = model(data)
            loss = torch.mean((reconstructed - data) ** 2, dim=1)
            anomalies.extend((loss > threshold).cpu().numpy())
    return np.array(anomalies)

if __name__ == '__main__':
    # Hyperparameters
    input_dim = 10
    batch_size = 32
    learning_rate = 0.001
    epochs = 20
    anomaly_threshold = 0.1

    # Prepare dataset and dataloaders
    dataset = InfusionPumpDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = Autoencoder(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the autoencoder
    train_autoencoder(model, dataloader, criterion, optimizer, epochs)

    # Evaluate the model
    test_dataset = InfusionPumpDataset(num_samples=200)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    anomalies = evaluate_autoencoder(model, test_dataloader, anomaly_threshold)

    # Print results
    print(f"Detected anomalies: {np.sum(anomalies)} out of {len(test_dataset)} samples")