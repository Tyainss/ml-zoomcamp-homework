

### Question 1

# nn.BCEWithLogitsLoss()
# This is because it's a binary classification, with only a single output neuron


### Question 2

import torch
import torch.nn as nn

class HairTypeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution + ReLU + MaxPool
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3  # same as (3, 3)
                # default: stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (2, 2)
        )

        # After conv+pool, we need to know the flattened size.
        # Input: (3, 200, 200)
        # Conv (3x3, no padding): output H/W = 200 - 3 + 1 = 198
        # MaxPool(2x2): output H/W = 198 / 2 = 99
        # Channels: 32
        # => flatten size = 32 * 99 * 99
        flattened_size = 32 * 99 * 99

        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu = nn.ReLU()

        # Output layer: 1 neuron (logit for binary classification)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv_block(x)
        # flatten: (batch_size, 32, 99, 99) -> (batch_size, 32*99*99)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)

        # IMPORTANT: we return raw logits here (no sigmoid)
        # BCEWithLogitsLoss will handle sigmoid internally
        x = self.out(x)
        return x


# Example of creating the model and moving it to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HairTypeCNN().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Total parameters: 20073473


### Question 3

import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transforms for both train and test (no augmentation yet)
train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

data_dir = "data"

# Datasets: ImageFolder will read subfolders as classes
train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"),
    transform=train_transforms,
)

validation_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "test"),
    transform=train_transforms,
)

# DataLoaders: batch_size=20, shuffle=True for train, False for test
train_loader = DataLoader(
    train_dataset,
    batch_size=20,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=20,
    shuffle=False,
)

# Train for 10 epochs and store history
num_epochs = 10
history = {"acc": [], "loss": [], "val_acc": [], "val_loss": []}

for epoch in range(num_epochs):
    # ----- TRAINING -----
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # BCEWithLogitsLoss expects float labels with shape (batch_size, 1)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass: raw logits
        outputs = model(images)

        # Loss (includes sigmoid internally)
        loss = criterion(outputs, labels)

        # Backprop + update
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * images.size(0)

        # Compute training accuracy
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history["loss"].append(epoch_loss)
    history["acc"].append(epoch_acc)

    # ----- VALIDATION -----
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history["val_loss"].append(val_epoch_loss)
    history["val_acc"].append(val_epoch_acc)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
        f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}"
    )

# Median of training accuracy over the 10 epochs (for Q3)
train_accs = np.array(history["acc"])
median_train_acc = np.median(train_accs)
print("Median train accuracy over 10 epochs:", median_train_acc)

# Median train accuracy over 10 epochs: 0.810625


### Question 4

train_losses = np.array(history["loss"])
std_train_loss = np.std(train_losses)  # population std is fine here
print("Train losses:", train_losses)
print("Standard deviation of training loss over 10 epochs:", std_train_loss)

# Standard deviation of training loss over 10 epochs: 0.16867895056903376


### Question 5

train_transforms_aug = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Recreate ONLY the training dataset and dataloader with augmentation
aug_train_dataset = datasets.ImageFolder(
    root=os.path.join(data_dir, "train"),
    transform=train_transforms_aug,
)

aug_train_loader = DataLoader(
    aug_train_dataset,
    batch_size=20,     # same batch size as before
    shuffle=True,
)

# We KEEP the same model, optimizer, criterion, and validation_loader
# Train for 10 more epochs with augmented data
num_epochs_aug = 10
aug_val_losses = []

for epoch in range(num_epochs_aug):
    # ----- TRAIN on augmented train data -----
    model.train()
    for images, labels in aug_train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # ----- EVALUATE on validation (test) set -----
    model.eval()
    val_running_loss = 0.0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # accumulate validation loss
            val_running_loss += loss.item() * images.size(0)
            total_val += images.size(0)

    val_epoch_loss = val_running_loss / len(validation_dataset)
    aug_val_losses.append(val_epoch_loss)

    print(
        f"[Augmented] Epoch {epoch + 1}/{num_epochs_aug}, "
        f"Val Loss: {val_epoch_loss:.4f}"
    )

# Mean test loss over the 10 augmented epochs (for Q5)
aug_val_losses = np.array(aug_val_losses)
mean_aug_test_loss = np.mean(aug_val_losses)
print("Mean test (validation) loss over 10 augmented epochs:", mean_aug_test_loss)

# Mean test (validation) loss over 10 augmented epochs: 0.5312283328942844

### Question 6


num_epochs_aug_q6 = 10
aug_val_accs = []

for epoch in range(num_epochs_aug_q6):
    # ----- TRAIN on augmented train data -----
    model.train()
    for images, labels in aug_train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # ----- EVALUATE on validation (test) set -----
    model.eval()
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)

            # compute probabilities and predictions for accuracy
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_acc = correct_val / total_val
    aug_val_accs.append(val_epoch_acc)

    print(
        f"[Augmented-Q6] Epoch {epoch + 1}/{num_epochs_aug_q6}, "
        f"Val Acc: {val_epoch_acc:.4f}"
    )

# Average test (validation) accuracy for the last 5 epochs (6 to 10)
aug_val_accs = np.array(aug_val_accs)
last5_mean_acc = np.mean(aug_val_accs[-5:])
print("Mean validation accuracy over last 5 augmented epochs (6–10):", last5_mean_acc)

# Mean validation accuracy over last 5 augmented epochs (6–10): 0.7960199004975126