import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dataset location relative to backend/
DATASET_DIR = Path(
    "../archive (3)/chest_xray/chest_xray"
)

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

# Where trained model will be saved
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.pth"

# Training configuration
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224

print("=" * 60)
print("EXPLAINABLE MEDICAL AI - MODEL TRAINING")
print("=" * 60)

print("\nDevice:", DEVICE)

print("\nDataset paths:")
print("Train:", TRAIN_DIR.resolve())
print("Validation:", VAL_DIR.resolve())
print("Test:", TEST_DIR.resolve())


# ============================================================
# 2. VERIFY DATASET
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR.resolve()}"
    )

if not VAL_DIR.exists():
    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR.resolve()}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR.resolve()}"
    )

print("\nDataset directories found successfully.")


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Small augmentation for training
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


validation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=validation_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=validation_transform
)


# ============================================================
# 5. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nClasses:")
print(train_dataset.classes)

print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nNumber of images:")
print("Training   :", len(train_dataset))
print("Validation :", len(val_dataset))
print("Testing    :", len(test_dataset))


# Make sure class mapping is the same
if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError(
        "Training and validation class mappings are different."
    )

if train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise ValueError(
        "Training and test class mappings are different."
    )


# ============================================================
# 6. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 7. CREATE RESNET18 MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING RESNET18")
print("=" * 60)

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# Get number of input features
number_of_features = model.fc.in_features


# Replace ImageNet classifier
model.fc = nn.Linear(
    number_of_features,
    2
)


# Move model to CPU/GPU
model = model.to(DEVICE)


print("\nModel:")
print(model.fc)

print("\nOutput classes:")
print("0 =", train_dataset.classes[0])
print("1 =", train_dataset.classes[1])


# ============================================================
# 8. LOSS FUNCTION AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 9. VALIDATION FUNCTION
# ============================================================

def evaluate_model(model, data_loader):

    model.eval()

    all_predictions = []
    all_labels = []

    running_loss = 0.0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = (
        running_loss / len(data_loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return (
        average_loss,
        accuracy,
        all_labels,
        all_predictions
    )


# ============================================================
# 10. TRAINING LOOP
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for batch_number, (images, labels) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    training_loss = (
        running_loss / len(train_loader)
    )

    training_accuracy = (
        100 * correct / total
    )

    # Validation
    validation_loss, validation_accuracy, _, _ = (
        evaluate_model(
            model,
            val_loader
        )
    )

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss      : {training_loss:.4f}"
    )

    print(
        f"Training Accuracy  : {training_accuracy:.2f}%"
    )

    print(
        f"Validation Loss    : {validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )


# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy, y_true, y_pred = (
    evaluate_model(
        model,
        test_loader
    )
)


# Calculate metrics
precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\nTest Results:")

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Accuracy      : {test_accuracy * 100:.2f}%"
)

print(
    f"Precision     : {precision * 100:.2f}%"
)

print(
    f"Recall        : {recall * 100:.2f}%"
)

print(
    f"F1 Score      : {f1 * 100:.2f}%"
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(cm)


print("\nClass order:")

for index, class_name in enumerate(
    train_dataset.classes
):
    print(
        index,
        "->",
        class_name
    )


# ============================================================
# 13. SAVE MODEL
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    "\nModel saved successfully:"
)

print(
    MODEL_PATH.resolve()
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"\nModel: ResNet18"
)

print(
    f"Classes: {train_dataset.classes}"
)

print(
    f"Training images: {len(train_dataset)}"
)

print(
    f"Validation images: {len(val_dataset)}"
)

print(
    f"Test images: {len(test_dataset)}"
)

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

print(
    f"Test Precision: {precision * 100:.2f}%"
)

print(
    f"Test Recall: {recall * 100:.2f}%"
)

print(
    f"Test F1: {f1 * 100:.2f}%"
)

print(
    f"\nSaved model: {MODEL_PATH}"
)

print("\nUS-B06 model training completed.")